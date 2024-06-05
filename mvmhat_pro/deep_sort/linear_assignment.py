# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from Tools.linear_assignment import sklearn_linear_assignment as linear_assignment
from . import kalman_filter
import config as C

# 用于将该视角没有匹配上的检测目标通过多视角的目标所对应的轨迹进行匹配，匹配结果存在possible_matches
def spatial_association(tracker, view):
    # 得到该视角i与所有视角（包括本视角）的检测目标的匹配矩阵P，是每个视角j-->i和j视角检测目标匹配矩阵的映射
    matching_dict = tracker.matching_mat[view]
    # 用来存储该视角跨视角关联上的（轨迹，检测目标）对，注是（轨迹！！！id！！！，检测目标下标）的形式
    possible_matches = []
    # 用于存储已匹配的轨迹id，防止不同detection匹配成同一个id轨迹的情况出现
    have_used_trackid=[]
    # 该视角本帧经过级联匹配和iou匹配后仍没有匹配上的检测目标，即需要参与跨视角匹配的检测目标（是数组下标形式）
    unmatches = tracker.mvtrack_dict[view].unmatched_detections
    # 从后往前遍历所有该视角目前还没有匹配的检测目标
    for det_id in unmatches[::-1]:
        # isTracked标记着该检测目标是否已经有一个轨迹（其他视角的检测目标）匹配上了，初始为-1，匹配后变为1
        isTracked = -1
        # 遍历该多视角视频中的所有视角
        for view_tgt in tracker.view_ls:
            # 若该检测目标已经匹配上，则退出循环，进行下一个检测目标的匹配
            if isTracked == 1:
                break
            # 若视角就是该视角，则跳出本次循环
            if view == view_tgt:
                continue
            # 该检测目标没被匹配，且现在的视角不是该视角
            else:
                # 得到该检测目标（det_id）与view_tgt视角的所有检测目标的匹配情况
                matching_row = matching_dict[view_tgt][det_id]
                associated_num = np.sum(matching_row)
                # view_tgt视角下没有检测目标与该检测目标（det_id）匹配
                if associated_num == 0:
                    continue
                # view_tgt视角下有多个检测目标与该检测目标（det_id）匹配，说明结果不可信，不要（实际该情况永远不会发生）
                elif associated_num > 1:
                    print('too many association')
                # view_tgt视角下只有一个检测目标与该检测目标（det_id）匹配
                else:
                    # associated_id为view_tgt视角下与该检测目标（det_id）匹配的检测目标在view_tgt视角中detections中的数组下标
                    associated_id = np.where(matching_row == 1)[0][0]
                    # 遍历view_tgt视角下所有匹配成功的（轨迹，检测目标）对
                    for match in tracker.mvtrack_dict[view_tgt].matches:
                        # 若associated_id对应的检测目标与轨迹匹配成功，则det_id也应与该轨迹配对
                        if associated_id == match[1] and not match[0] in have_used_trackid:
                            # 标记det_id检测目标匹配完成
                            isTracked = 1
                            # 该轨迹与det_id检测目标匹配成功
                            possible_matches.append((match[0], det_id))
                            have_used_trackid.append(match[0])
                            # 从该视角中（view视角中）没有匹配的检测目标中去除det_id检测目标
                            # tracker.mvtrack_dict[view].unmatched_detections.remove(det_id)
    # 将得到的view视角中可能匹配对写回到view视角的tracker中
    tracker.mvtrack_dict[view].possible_matches = possible_matches

def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections

# 级联匹配
def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None,last_frame_trackid=None,this_frame_detect_id=None,time_assign_matrix=None,kalman_use=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        #新增加
        if level==0:
            truly_have_track_id=[tracks[indice].track_id for indice in track_indices_l]
            truly_have_detectid=[detections[indice].id for indice in unmatched_detections]
            assert(last_frame_trackid==truly_have_track_id and this_frame_detect_id==truly_have_detectid)

            # 新增卡尔曼滤波对检测框距离过滤
            # if C.inference_time_consider_filter:
                # 提取出相似度小于阈值的不合法指派
                # print('go here!')
                # matches_l=[]
                # similarity_false_mask=time_assign_matrix<C.inference_time_simliarity_threshold
                # dis_matrix=gate_distance_matrix(kalman_use,tracks,detections,track_indices_l,unmatched_detections)
                # dis_fasle_mask=dis_matrix==0.0
                # time_assign_matrix=time_assign_matrix*dis_matrix
                # cost_matrix=-time_assign_matrix
                # cost_matrix[similarity_false_mask]=1+1e-5
                # cost_matrix[dis_fasle_mask]=1+1e-5
                # indices = linear_assignment(cost_matrix)
                # for row, col in indices:
                #     track_idx = track_indices_l[row]
                #     detection_idx = unmatched_detections[col]
                #     if cost_matrix[row, col]<1:
                #         matches_l.append((track_idx, detection_idx))
            # ----------
            # else:
            #     indice_matrix = np.array([[(track_ind, detect_ind) for detect_ind in unmatched_detections] for track_ind in track_indices_l])
            #     mask=time_assign_matrix==1.0
            #     matches_l=indice_matrix[mask].tolist()
            # unmatched_detections=list(set(unmatched_detections)-set(v for _,v in matches_l))
            # print('')

            # TODO 暂时这么写
            # matches_l_2=[]
            # similarity_false_mask = time_assign_matrix < C.inference_time_simliarity_threshold
            # dis_matrix=gate_distance_matrix(kalman_use,tracks,detections,track_indices_l,unmatched_detections)
            # dis_fasle_mask = dis_matrix == 0.0
            # cost_matrix=-time_assign_matrix
            # cost_matrix[similarity_false_mask] = 1 + 1e-5
            # cost_matrix[dis_fasle_mask]=1+1e-5
            # indices = linear_assignment(cost_matrix)
            # for row, col in indices:
            #     track_idx = track_indices_l[row]
            #     detection_idx = unmatched_detections[col]
            #     if cost_matrix[row, col]<1:
            #         matches_l_2.append((track_idx, detection_idx))


            indice_matrix = np.array([[(track_ind, detect_ind) for detect_ind in unmatched_detections] for track_ind in track_indices_l])
            dis_matrix = gate_distance_matrix(kalman_use, tracks, detections, track_indices_l, unmatched_detections)
            time_assign_matrix = time_assign_matrix * dis_matrix
            mask = time_assign_matrix == 1.0
            matches_l_2 = indice_matrix[mask]
            matches_l_2=[tuple(i) for i in matches_l_2]
            # matches_l_1是原级联匹配结果
            # matches_l_2是指派矩阵结果
            matches_l_1, _, _ = min_cost_matching(distance_metric, max_distance, tracks, detections,track_indices_l, unmatched_detections)

            # case1
            # matches_l由两者取交集得到
            if C.inference_time_assign_type=='intersection':
                matches_l=[pair1 for pair1 in matches_l_1 if pair1 in matches_l_2]

            # case2
            # matches_l由取并集得到，并以原级联匹配结果为准
            elif C.inference_time_assign_type=='union_matching_cascade':
                for pair_2 in matches_l_2[::-1]:
                    for pair_1 in matches_l_1:
                        if pair_2[0]==pair_1[0] or pair_2[1]==pair_1[1]:
                            matches_l_2.remove(pair_2)
                            break
                matches_l=matches_l_1+matches_l_2

            # case3
            # matches_l由取并集得到，并以指派矩阵结果为准
            elif C.inference_time_assign_type=='union_assign_matrix':
                for pair_1 in matches_l_1[::-1]:
                    for pair_2 in matches_l_2:
                        if pair_1[0]==pair_2[0] or pair_1[1]==pair_2[1]:
                            matches_l_1.remove(pair_1)
                            break
                matches_l=matches_l_1+matches_l_2

            elif C.inference_time_assign_type=='no_use':
                matches_l=matches_l_1

            elif C.inference_time_assign_type=='only_assign_matrix':
                matches_l=matches_l_2

            else:
                print('please enter right inference time assign type!')


            # if matches_l_2 != matches_l_1:
            #     print(matches_l_1)
            #     print(matches_l_2)
            #     print(matches_l)
            unmatched_detections = list(set(unmatched_detections) - set(v for _, v in matches_l))
            # TODO------后续需修改



        else:
        #-------
            # 返回的都是tracks和detection中数组下标
            matches_l, _, unmatched_detections = min_cost_matching(distance_metric, max_distance, tracks, detections,track_indices_l, unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=C.INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix


# 新增卡尔曼滤波对检测框距离进行过滤
def gate_distance_matrix(kf,tracks, detections, track_indices, detection_indices,only_position=False):
    matrix=np.ones((len(track_indices),len(detection_indices)))
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track=tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        matrix[row,gating_distance>gating_threshold]=0.0
    return matrix