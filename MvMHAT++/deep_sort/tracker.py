# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
import config as C

# 代表着一个视角下的tracker
# 单视角下的tracker执行顺序如下：predict->pre_update->linear_assignment.spatial_association->update
class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=80, n_init=3, next_id=[1]):
        # metric为nn_matching中的余弦距离指标
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        # tracks只存储当前时刻confirmed和Tentative的轨迹，不包括deleted
        self.tracks = []
        # 该视角下新分配轨迹应该的id
        self.next_id = next_id
        # 记录经过级联匹配和iou匹配后匹配的（轨迹id，检测目标下标）对
        self.matches = []
        # 当rematch=true时，用来存储经过级联匹配和iou匹配后匹配的（轨迹id，检测目标下标）对
        self.matches_backup = []
        # 存储在tracks中多视角视频中一个视角都没出现的轨迹的id
        self.unmatched_tracks = []
        # 存储经过级联匹配、iou匹配、跨视角匹配后还没有匹配的检测结果
        self.unmatched_detections = []
        # 级联匹配和iou匹配后该视角没匹配的检测目标，通过多视角检测目标匹配，最终匹配的（轨迹，检测目标）对
        self.possible_matches = []
        # 是detection类对象的list，用来存储其中某一帧该视角下的所有检测目标
        self.detections = None

        # 新增加
        self.last_frame_trackid=None
        self.this_frame_detect_id=None
        self.time_assign_matrix=None
        #----

    def predict(self):
        """Propagate track state distributions one time step forward.
        使用卡尔曼滤波对轨迹状态进行预测，应在级联匹配和iou匹配前调用该函数
        This function should be called once every time step, before `update`.
        """
        # 对当前时刻所有confirmed和Tentative的轨迹进行卡尔曼滤波预测
        for track in self.tracks:
            track.predict(self.kf)

    def pre_update(self, re_matching):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        # 进行级联匹配和iou匹配，若rematch=true会对最终的匹配结果进行一定修改
        # 当rematch=true时，matches为空
        self.matches, self.unmatched_tracks, self.unmatched_detections, self.matches_backup = self._match(re_matching)
        # 当rematch=false时，该行代码才有意义
        # to_abs_idx将原匹配对（轨迹对应数组下标，检测目标对应数组下标）-->（轨迹对应id，检测目标对应数组下标）
        self.matches = self.to_abs_idx(self.matches)
        self.matches_backup = self.to_abs_idx(self.matches_backup)
        # 将没有匹配轨迹的存储方式由原来的数组对应下标转化为轨迹id
        self.unmatched_tracks = [self.tracks[i].track_id for i in self.unmatched_tracks]
        # return matches, unmatched_tracks, unmatched_detections

    # 暂时不清该函数去重track_id的意义
    def to_abs_idx(self, idx_pairs):
        abs_idx_pairs = []
        for pair in idx_pairs:
            abs_idx_pairs.append((self.tracks[pair[0]].track_id, pair[1]))
        del_ls = []
        for i, pair_i in enumerate(abs_idx_pairs):
            for j, pair_j in enumerate(abs_idx_pairs):
                if i < j:
                    if pair_i[0] == pair_j[0]:
                        del_ls.append(pair_j)
        ret = [i for i in abs_idx_pairs if i not in del_ls]
        return ret

    # 用于根据pre_update后的matches、unmatched_tracks、unmatched_detections、matches_backup进行相应的更新操作
    def update(self):
        # Update track set.

        if C.inference_mode=="without_association":
            # 跨视角关联不能用来找回该视角中存在的轨迹，只能用来创建新轨迹（即新出现目标）
            for match in self.possible_matches[::-1]:
                for track in self.tracks:
                    if match[0] == track.track_id:
                        self.possible_matches.remove(match)
                        break

        # 这是对于re_match=true进行的更新，因为此时matches_backup才不为空
        # 对于经过级联匹配和iou匹配后的匹配对matches_backup，如果其轨迹或检测目标在possible_matches中也出现了，则以possible_matches的为准，删去matches_backup中的
        for bmatch in self.matches_backup[::-1]:
            for match in self.possible_matches:
                if bmatch[0] == match[0] or bmatch[1] == match[1]:
                    self.matches_backup.remove(bmatch)
                    break

        # 对rematch=false有意义，因为此时matches不为空
        # 注：match和pmatch一定不会有相同的检测目标，只可能有相同的轨迹
        # 若match和pmatch有相同的轨迹，则以matches为准，即去掉possible_matches中的
        for pmatch in self.possible_matches[::-1]:
            for match in self.matches:
                if pmatch[0] == match[0] or pmatch[1]==match[1]:
                    self.possible_matches.remove(pmatch)
                    break

        # 此时matches,matches_backup和possible_matches都已整理完毕，可以进行合并
        if C.inference_mode=="association_and_tracking" or C.inference_mode=="without_association":
            self.matches = self.matches+self.matches_backup+self.possible_matches
        elif C.inference_mode=="without_tracking":
            self.matches = self.matches+self.possible_matches
        else:
            print('Wrong inference mode!')

        # 对matches中匹配对进行更新
        for track_idx,detection_idx in self.matches:
            # state=1标志着possible_matches着匹配对中的轨迹不在tracks中，则需放入tracks；=0表示在tracks中，直接调用track.update函数
            state=1
            for track in self.tracks:
                if track.track_id == track_idx:
                    track.update(self.kf,self.detections[detection_idx])
                    state=0
            if state:
                self._associate_track(self.detections[detection_idx],track_idx)

        # 计算未匹配detection
        self.unmatched_detections=list(set(range(len(self.detections)))-set(v for _,v in self.matches))
        # 对unmatches_detections进行处理
        for detection_idx in self.unmatched_detections:
            track_idx = self._initiate_track(self.detections[detection_idx])
            self.matches.append((track_idx, detection_idx))


        # TODO 只是用来实验新的可能(不采用)
        # for pair in self.matches[::-1]:
        #     for one_track in self.tracks:
        #         if one_track.track_id==pair[0]:
        #             if not one_track.is_confirmed():
        #                 self.matches.remove(pair)
        #             break


    # 用于对unmatched_tracks进行相应操作
    def last_update(self):

        # 对没有匹配成功的轨迹更新（根据track的状态决定是否标为删除态）
        # 注：只有tracks中的轨迹在该帧的各个视角下都没有被匹配时，才会在unmatched_tracks中，所以该轨迹只要在其中任何一个视角中被匹配，都不会被删除
        for track_idx in self.unmatched_tracks:
            for track in self.tracks:
                if track.track_id == track_idx:
                    track.mark_missed()

        # tracks中只保留Tentative和confirmed的轨迹
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        # 只对confirmed的轨迹更新特征向量集，所以只取confirmed状态的轨迹的id
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            # 新增加
            if track.features!=[]:
                track.last_feature=track.features[-1]
            #-------------
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, re_matching):
        # 获得门控代价矩阵，首先要获得余弦/欧式距离矩阵，再通过gate_cost_matrix函数得到门控代价矩阵
        # tracks表示所有待匹配轨迹（track对象），dets表示该帧所有检测目标（detection对象）
        # track_indices代表所有参与计算代价矩阵的轨迹在tracks中的对应下标，detection_indices代表所有参与计算代价矩阵的检测目标在dets中的对应下标
        def gated_metric(tracks, dets, track_indices, detection_indices):
            # 获取所有参与比较的检测目标的特征向量
            features = np.array([dets[i].feature for i in detection_indices])
            # 获得所有参与比较的轨迹的！！！id！！！
            targets = np.array([tracks[i].track_id for i in track_indices])
            # 调用nn_matching中的distance函数，得到轨迹与所有检测目标的距离（代价）特征矩阵（只考虑特征间相似度，不考虑检测框位置距离）
            cost_matrix = self.metric.distance(features, targets)
            # 将上一步生成的cost_matrix通过gate_cost_matrix函数得到门控代价矩阵
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            # 返回门控代价矩阵
            return cost_matrix

        # 存储tracks中confirmed的轨迹的对应下标
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        # 存储tracks中Tentative(unconfirmed)的轨迹的对应下标
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        # 进行级联匹配(只有confirmed的轨迹才参加），注detection_indices没有传参数，即代表detections中所有检测结果都参与级联匹配
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, self.detections, confirmed_tracks,last_frame_trackid=self.last_frame_trackid,this_frame_detect_id=self.this_frame_detect_id,time_assign_matrix=self.time_assign_matrix,kalman_use=self.kf)

        # 进行iou匹配，参与匹配的轨迹包括tracks中unconfirmed的轨迹和！！！！上一轮匹配成功！！！！但这一轮级联比较没匹配成功的轨迹（这是因为iou匹配的特性，跨帧的话iou会变小，但有可能是同一个人）
        # 获取参与iou比较的tracks中轨迹对应下标
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        # unmatched_tracks_a中轨迹不参与iou匹配，因此这些轨迹肯定是本一轮的失配轨迹
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        # 进行iou匹配，参与匹配的检测目标就是级联匹配没成功的检测目标
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                self.detections, iou_track_candidates, unmatched_detections)
        # 本轮的匹配成功对=级联匹配成功对并上iou匹配成功对
        matches = matches_a + matches_b
        # 用于rematch时存储上面已经匹配成功的matches中的轨迹
        unmatched_tracks_c = []
        # 用于rematch时存储上面已经匹配成功的matches中的检测目标
        unmatched_detections_b = []
        # 用于rematch时存储上面已经匹配成功的matches中的匹配对
        matches_backup = []
        # re_matching是bool类型，true表示进行重匹配，不进行级联匹配和iou匹配？
        if re_matching:
            unmatched_tracks_c = [i[0] for i in matches]
            unmatched_detections_b = [i[1] for i in matches]
            matches_backup = matches[:]
            # 已匹配的匹配对清空
            matches = []
        # 本轮没匹配成功的轨迹包括级联匹配失败和iou匹配失败的轨迹，当rematch=true是还要考虑已匹配成功的轨迹
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b + unmatched_tracks_c))
        # 本轮没匹配成功的检测目标包括经过iou匹配后还没匹配成的检测目标，当rematch=true是还要考虑已匹配成功的检测目标
        unmatched_detections += unmatched_detections_b
        return matches, unmatched_tracks, unmatched_detections, matches_backup


    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self.next_id[0], self.n_init, self.max_age,
            detection.feature))
        idx = self.next_id[0]
        self.next_id[0] += 1
        return idx

    def _associate_track(self, detection, track_idx):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, track_idx, self.n_init, self.max_age,
            detection.feature))
