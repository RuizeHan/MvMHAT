import torch
import torch.nn.functional as f
from deep_sort.detection import Detection
from sklearn import preprocessing as sklearn_preprocessing
from application_util import preprocessing
from sklearn.utils.extmath import softmax
from . import linear_assignment
from application_util import visualization
from Tools.linear_assignment import sklearn_linear_assignment
import cv2
import numpy as np
import config as C
from all_loss import gen_S
from matchSVT import SVT

class Update():
    def __init__(self, seq, mvtracker, display, model=None):
        # model指深度匈牙利网络
        self.model=model
        # 存储多视角视频在每个视角下的信息，seq是个dict{view:info}
        self.seq = seq
        # 存储多视角视频的所有视角编号名称
        self.view_ls = mvtracker.view_ls
        # mvtracker为mvtracker类对象，用于多视角下跟踪
        self.tracker = mvtracker
        self.display = display
        # 定义检测框的最小置信度
        self.min_confidence = 0.8
        # 定义检测框的最大重叠度
        self.nms_max_overlap = 1.0
        # 定义检测框最小高度
        self.min_detection_height = 0
        # 自适应温度softmax的两个参数
        self.delta = 0.5
        self.epsilon = 0.1
        # self.epsilon = 0.5
        # result是dict，{view:该视角下每帧中每个轨迹的标注框位置}
        self.result = {key: [] for key in self.view_ls}
        matrix = [[]]

    # 按照帧号保留该帧检测框高度>min_height的框
    def create_detections(self, detection_mat, frame_idx, min_height=0):
        # detection_mat是关于检测结果的矩阵，每一行是一个检测结果包括：帧号、框id、框位置（tlwh）、框的置信度、框对应的特征向量等
        # 若没有检测结果，则该帧也一定没有检测结果
        if len(detection_mat) == 0:
            return []
        # 获得每一行检测结果所对应的帧号
        frame_indices = detection_mat[:, 0].astype(int)
        # 只要frame_idx帧的检测结果
        mask = frame_indices == frame_idx

        detection_list = []
        # 取出frame_idx帧的检测结果
        for row in detection_mat[mask]:
            bbox, confidence, feature, id = row[2:6], row[6], row[10:], row[1]
            if bbox[3] < min_height:
                continue
            # 只要检测框高度>min_height的框，生成Detection类对象
            detection_list.append(Detection(bbox, confidence, feature, id))
        return detection_list

    # 对于create_detections函数得到的检测结果进行置信度筛选和nms抑制，该函数需给出明确的视角view
    def select_detection(self, frame_idx, view):
        detections = self.create_detections(
            self.seq[view]["detections"], frame_idx, self.min_detection_height)
        # 只保留view视角下frame_idx帧中检测框置信度>=min_confidence的框
        detections = [d for d in detections if d.confidence >= self.min_confidence]

        # Run non-maxima suppression.
        # 获取保留下的每个框的位置
        boxes = np.array([d.tlwh for d in detections])
        # 获取保留下的每个框的置信度
        scores = np.array([d.confidence for d in detections])
        # 去重检测框中重叠面积大且置信度低的框，返回保留下标
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        return detections

    def frame_matching(self, frame_idx):
        # 该函数计算所有视角中任意两个视角i和视角j中所有检测目标的匹配矩阵P，并存为两个dict，首先是{i->dicti},其中dicti为{j->Pij}
        def gen_X(features,dict_last_frame_track_num):
            # 将features中每个视角特征向量矩阵标准化，实则无需此操作，因为features已经标准化
            features = [sklearn_preprocessing.normalize(i, axis=1) for i in features]
            # all_blocks_X是一个dict，view->list
            all_blocks_X = {view: [] for view in self.view_ls}

            if self.model == None:
                # x是一个视角下该帧所有检测目标的特征向量的array，view_x是对应视角的编号名称
                for x, view_x in zip(features, self.view_ls):
                    # row_blocks_X是一个dict，view->list
                    row_blocks_X = {view: [] for view in self.view_ls}
                    # y是一个视角下该帧所有检测目标的特征向量的array，view_y是对应视角的编号名称
                    for y, view_y in zip(features, self.view_ls):
                        # 计算x中所有特征向量与y中所有特征向量间的余弦相似度，S12即为相似度矩阵
                        S12 = np.dot(x, y.transpose(1, 0))

                        # 计算温度自适应softmax中的𝜏
                        scale12 = np.log(self.delta / (1 - self.delta) * S12.shape[1]) / self.epsilon
                        # 利用温度自适应softmax使S12中每行元素的和为1，即变为概率
                        S12 = softmax(S12 * scale12)
                        # 将概率小于0.5的认定为两个检测目标肯定不是同一个人
                        S12[S12 < 0.5] = 0
                        # assign_ls为使用匈牙利算法后的匹配结果，-S12代表相似度越大，代价（距离）越小
                        assign_ls = sklearn_linear_assignment(- S12)
                        # X_12即为论文中的P矩阵，初始矩阵中数值全为0
                        X_12 = np.zeros((S12.shape[0], S12.shape[1]))
                        # 对于每个匹配对，只有S12中对应值不为0，才把X_12对应位置赋为1
                        for assign in assign_ls:
                            if S12[assign[0], assign[1]] != 0:
                                X_12[assign[0], assign[1]] = 1
                        # row_blocks_X是dict，view_y->X_12构成其中一个映射
                        row_blocks_X[view_y] = X_12
                    # all_blocks_X是dict，view_x->row_blocks_X构成其中一个映射
                    all_blocks_X[view_x] = row_blocks_X
            elif self.model:
                self.model.eval()
                with torch.no_grad():
                    features=[torch.from_numpy(i).to(dtype=torch.float32).cuda() for i in features]
                    each_len=[i.shape[0] for i in features]
                    # x是一个视角下该帧所有检测目标的特征向量的array，view_x是对应视角的编号名称
                    pred_all=[]
                    for x in features:
                        # row_blocks_X是一个dict，view->list
                        # row_blocks_X = {view: [] for view in self.view_ls}
                        pred_row=[]
                        # y是一个视角下该帧所有检测目标的特征向量的array，view_y是对应视角的编号名称
                        for y in features:
                            # 计算x中所有特征向量与y中所有特征向量间的余弦相似度，S12即为相似度矩阵
                            S12=torch.mm(x,y.transpose(1,0))
                            # S_norm=(S12+1)/2
                            # dis_matrix=(1-S_norm).unsqueeze(0)
                            dis_matrix=S12.unsqueeze(0)
                            if C.use_softmax=='temperature_softmax':
                                scale=np.log(C.delta/(1-C.delta)*dis_matrix.shape[2])/C.epsilon
                                dis_matrix=f.softmax(dis_matrix*scale,dim=2)

                            elif C.use_softmax=='row_col_softmax':
                                row_softmax=f.softmax(dis_matrix,dim=2)
                                col_softmax=f.softmax(dis_matrix,dim=1)
                                dis_matrix=row_softmax*col_softmax

                            elif C.use_softmax=='row_col_temperature_softmax':
                                scale=np.log(C.delta/(1-C.delta)*dis_matrix.shape[2])/C.epsilon
                                row_softmax=f.softmax(dis_matrix*scale,dim=2)
                                scale=np.log(C.delta/(1-C.delta)*dis_matrix.shape[1])/C.epsilon
                                col_softmax=f.softmax(dis_matrix*scale,dim=1)
                                dis_matrix=row_softmax*col_softmax

                            else:
                                print('not use softmax!')

                            # self.model.hidden_row=self.model.init_hidden(1)
                            # self.model.hidden_col = self.model.init_hidden(1)
                            # pred=self.model(dis_matrix).squeeze(0)
                            pred_row.append(dis_matrix.squeeze(0))
                        pred_all.append(pred_row)
                    # 将预测结果拼成大的矩阵
                    one_pred = torch.cat([torch.cat(row, dim=1) for row in pred_all], dim=0)
                    # 使用dhn网络得到最终的预测指派矩阵
                    if C.stan_type == 'rnn':
                        self.model.hidden_row = self.model.init_hidden(1)
                        self.model.hidden_col = self.model.init_hidden(1)
                    # pred为dhn网络输出的预测结果
                    pred = self.model(one_pred.unsqueeze(0)).squeeze(0)

                    # 此处可对输出的预测结果另行处理
                    # TODO 尝试1：将预测结果变为对称阵
                    pred=(pred+pred.T)/2

                    # TODO 尝试2：将预测结果核范数最小化+对称化
                    # pred=SVT(pred,each_len)


                    #--------------------


                    row = torch.split(pred, each_len, dim=0)
                    dis_row_col = [list(torch.split(i, each_len, dim=1)) for i in row]

                    # 新增加
                    first_index=0
                    view_x_num=0
                    view_index=[]

                    while view_x_num < len(self.view_ls):
                        # 生成时序上的指派矩阵
                        if dict_last_frame_track_num[self.view_ls[view_x_num]]!=0:
                            pred=dis_row_col[first_index][first_index+1]
                            S12 = pred.cpu().numpy()

                            if C.inference_time_consider_filter:
                                self.tracker.mvtrack_dict[self.view_ls[view_x_num]].time_assign_matrix=S12
                            else:
                                # 将概率小于0.5的认定为两个检测目标肯定不是同一个人
                                S12[S12 < C.inference_global_threshold] = 0
                                # assign_ls为使用匈牙利算法后的匹配结果，-S12代表相似度越大，代价（距离）越小
                                assign_ls = sklearn_linear_assignment(- S12)
                                # X_12即为论文中的P矩阵，初始矩阵中数值全为0
                                X_12 = np.zeros((S12.shape[0], S12.shape[1]))
                                # 对于每个匹配对，只有S12中对应值不为0，才把X_12对应位置赋为1
                                for assign in assign_ls:
                                    if S12[assign[0], assign[1]] != 0:
                                        X_12[assign[0], assign[1]] = 1
                                self.tracker.mvtrack_dict[self.view_ls[view_x_num]].time_assign_matrix=X_12
                            first_index += 1
                        else:
                            self.tracker.mvtrack_dict[self.view_ls[view_x_num]].time_assign_matrix=None


                        view_index.append(first_index)
                        first_index+=1
                        view_x_num+=1

                    # 生成视角间的指派矩阵
                    for i_x,index_x in enumerate(view_index):
                        row_blocks_X = {view: [] for view in self.view_ls}
                        for i_y,index_y in enumerate(view_index):
                            pred=dis_row_col[index_x][index_y]
                            S12=pred.cpu().numpy()
                            # 将概率小于0.5的认定为两个检测目标肯定不是同一个人
                            S12[S12 < C.inference_global_threshold] = 0
                            # assign_ls为使用匈牙利算法后的匹配结果，-S12代表相似度越大，代价（距离）越小
                            assign_ls = sklearn_linear_assignment(- S12)
                            # X_12即为论文中的P矩阵，初始矩阵中数值全为0
                            X_12 = np.zeros((S12.shape[0], S12.shape[1]))
                            # 对于每个匹配对，只有S12中对应值不为0，才把X_12对应位置赋为1
                            for assign in assign_ls:
                                if S12[assign[0], assign[1]] != 0:
                                    X_12[assign[0], assign[1]] = 1
                            row_blocks_X[self.view_ls[i_y]]=X_12
                        all_blocks_X[self.view_ls[i_x]]=row_blocks_X
                    #----

                    # 原本的
                    # for i,view_x in zip(range(len(dis_row_col)),self.view_ls):
                    #     row_blocks_X是一个dict，view->list
                    #     row_blocks_X = {view: [] for view in self.view_ls}
                    #     for j,view_y in zip(range(len(dis_row_col)),self.view_ls):
                    #         pred=dis_row_col[i][j]
                    #         S12 = pred.cpu().numpy()
                            # 将概率小于0.5的认定为两个检测目标肯定不是同一个人
                            # S12[S12 < 0.5] = 0
                            # assign_ls为使用匈牙利算法后的匹配结果，-S12代表相似度越大，代价（距离）越小
                            # assign_ls = sklearn_linear_assignment(- S12)
                            # X_12即为论文中的P矩阵，初始矩阵中数值全为0
                            # X_12 = np.zeros((S12.shape[0], S12.shape[1]))
                            # 对于每个匹配对，只有S12中对应值不为0，才把X_12对应位置赋为1
                            # for assign in assign_ls:
                            #     if S12[assign[0], assign[1]] != 0:
                            #         X_12[assign[0], assign[1]] = 1
                            # row_blocks_X是dict，view_y->X_12构成其中一个映射
                            # row_blocks_X[view_y] = X_12
                        # all_blocks_X是dict，view_x->row_blocks_X构成其中一个映射
                        # all_blocks_X[view_x] = row_blocks_X

            return all_blocks_X

        # print("Matching frame %05d" % frame_idx)

        all_view_features = []
        all_view_id = []
        dict_last_frame_track_num={}
        # 遍历多视角视频中所有视角
        for view in self.view_ls:
            view_feature = []
            view_id = []
            # view视角的tracker对象中detections变量存储着该视角frame_idx帧所有的检测目标
            self.tracker.mvtrack_dict[view].detections = self.select_detection(frame_idx, view)
            # 遍历该视角该帧的所有检测目标，并将对应检测目标的特征向量和id分别放入view_feature和view_id
            for detection in self.tracker.mvtrack_dict[view].detections:
                view_feature.append(detection.feature)
                view_id.append(detection.id)

            #新增加
            self.tracker.mvtrack_dict[view].this_frame_detect_id=view_id



            #新增加
            last_time_feature=[]
            last_frame_trackid=[]
            track_indices = list(range(len(self.tracker.mvtrack_dict[view].tracks)))
            track_indices_l = [k for k in track_indices if self.tracker.mvtrack_dict[view].tracks[k].time_since_update == 0 and self.tracker.mvtrack_dict[view].tracks[k].is_confirmed()]
            for index in track_indices_l:
                last_time_feature.append(self.tracker.mvtrack_dict[view].tracks[index].last_feature)
                last_frame_trackid.append(self.tracker.mvtrack_dict[view].tracks[index].track_id)
            self.tracker.mvtrack_dict[view].last_frame_trackid=last_frame_trackid
            dict_last_frame_track_num[view]=len(last_frame_trackid)

            if last_time_feature!=[]:
                last_time_feature=np.stack(last_time_feature)
                all_view_features.append(last_time_feature)


            #------



            # 若该视角该帧下有检测目标
            if view_feature != []:
                # np.stack对单array没有用，所以view_feature、view_id和原来一样，只不过变成了array
                view_feature = np.stack(view_feature)
                view_id = np.stack(view_id)
                # 将view_feature中的每个特征向量标准化，使其模长=1
                # view_feature = sklearn_preprocessing.normalize(view_feature, norm='l2', axis=1)
                # 将该视角该帧下所有标准化后的特征向量放入all_view_features中，all_view_features是list，其中每个视角所有的特征向量构成1个array
                all_view_features.append(view_feature)
            # 若该视角该帧下没有检测目标
            else:
                # 若没有检测目标，则该视角为array([[0,...,0]])，即只有一个特征向量且全为0
                all_view_features.append(np.array([[0] * 1000]))
            # 若该视角该帧下有检测目标，则view_id为array([int,int,...]);若无，则view_id为[]
            # all_view_id实际并没有用
            all_view_id.append(view_id)
        # match_mat包含了该帧下各视角间所有检测目标的匹配情况
        match_mat = gen_X(all_view_features,dict_last_frame_track_num)
        # 更新mvtracker在该帧各个视角间所有检测目标的匹配矩阵P
        self.tracker.update(match_mat)

    # 对frame_idx帧进行轨迹与目标间的时序匹配（级联、iou）以及目标与目标间的关联匹配(利用匹配矩阵P)
    def frame_callback(self, frame_idx):
        # 当当前视频帧号是10（C.RENEW_TIME）的倍数时，re_matching=True；否则re_matching=False
        if C.RENEW_TIME:
            re_matching = frame_idx % C.RENEW_TIME == 0
        else:
            re_matching = 0
        # 遍历多视角视频中的每个视角
        for view in self.view_ls:
            # 对该视角当前时刻所有confirmed和Tentative的轨迹进行卡尔曼滤波预测
            self.tracker.mvtrack_dict[view].predict()
            # 每个视角在该帧进行级联匹配和iou匹配
            # 当re_matching=False时，matches是级联匹配和iou匹配匹配成功的匹配对，matches_backup是空
            # 当re_matching=True时，matches是空，matches_backup是级联匹配和iou匹配匹配成功的匹配对，同时unmatched_tracks和unmatched_detections也会算上之前匹配成功的轨迹和目标
            # pre_update会对各视角tracker的self.matches,unmatched_tracks,unmatched_detections,matches_backup进行更新，其中涉及轨迹的会以轨迹id的形式表现，涉及detection的会以数组下标形式表现
            # 第一个视角固定认为时序匹配的匹配结果正确，保证所有视角里至少有一个视角matches不为空，使spatial_association有意义
            if view == self.view_ls[0]:
                self.tracker.mvtrack_dict[view].pre_update(False)
            else:
                if C.inference_mode == 'association_and_tracking':
                    self.tracker.mvtrack_dict[view].pre_update(re_matching)
                elif C.inference_mode == 'without_tracking':
                    self.tracker.mvtrack_dict[view].pre_update(True)
                elif C.inference_mode == 'without_association':
                    self.tracker.mvtrack_dict[view].pre_update(False)
                else:
                    print('Wrong inference mode!')

        # 对所有视角进行视角间检测目标的匹配，得到各个视角的possible_matches，并且从unmatched_detections中去掉可能匹配对中的目标
        # 空间关联只考虑各视角没匹配的detection可能根据其它视角能否有相匹配的轨迹id，而不考虑已匹配上的detection与其他视角detection是同一目标时两者关联轨迹是否是同一个；也不考虑两者不是同一目标时两者关联轨迹是否不同
        # 同时在算possible_matches时，也与第一个匹配的detection关联的轨迹id配对（非最优解）
        # TODO 此处可以改进匹配轨迹id策略
        # print('')
        for view in self.view_ls:
            linear_assignment.spatial_association(self.tracker, view)
            # 对所有视角的tracker调用各自的update函数
            self.tracker.mvtrack_dict[view].update()

        track_ls = []
        # 获取所有视角中的全部匹配对，包括matches、matches_backup、possible_matches
        for view in self.view_ls:
            track_ls += self.tracker.mvtrack_dict[view].matches

        # 获取所有视角中全部匹配对中的所有轨迹id
        track_ls = [i[0] for i in track_ls]
        for view in self.view_ls:
            for track_ in track_ls:
                if track_ in self.tracker.mvtrack_dict[view].unmatched_tracks:
                    # 只要所有视角中有一个视角中有该轨迹，则该视角就不把该轨迹放入unmatched_tracks中，即认为该轨迹在该视角的后续帧中很有可能出现，该轨迹不会调用轨迹删除函数
                    self.tracker.mvtrack_dict[view].unmatched_tracks.remove(track_)

        for view in self.view_ls:
            self.tracker.mvtrack_dict[view].last_update()







    def frame_display(self, vis, frame_idx, view):

        # Update visualization.
        # 若进行可视化(display=true)
        if self.display:
            # 获得view视角下第frame_idx帧图片的位置，并根据位置读取图片
            image = cv2.imread(
                self.seq[view]["image_filenames"][frame_idx - self.seq[view]["min_frame_idx"]], cv2.IMREAD_COLOR)
            vis.set_image(image.copy(), view, str(frame_idx))
            # 画出view视角下第frame_idx帧所有轨迹的位置
            vis.draw_trackers(self.tracker.mvtrack_dict[view].tracks)

        # Store results.
        # 将该视角下该帧的所有轨迹位置存储下来
        for track in self.tracker.mvtrack_dict[view].tracks:
            # 只有confirmed状态和该视角最多失配1次轨迹才画出轨迹，记录这些轨迹的位置放入result
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            self.result[view].append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    def run(self):
        if self.display:
            visualizer = visualization.Visualization(self.seq, update_ms=5)
        else:
            visualizer = visualization.NoVisualization(self.seq)
        print('start inference...')
        visualizer.run(self.frame_matching, self.frame_callback, self.frame_display)
