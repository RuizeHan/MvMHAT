from __future__ import absolute_import
from .tracker import Tracker
from deep_sort import nn_matching

class MVTracker:
    def __init__(self, view_ls):
        # 用于存放各个视角的tracker类对象，是view-->tracker的映射
        self.mvtrack_dict = {}
        # 定义特征向量余弦距离的阈值
        self.max_cosine_distance = 0.2
        # 不限制每个轨迹存储的最大特征向量数
        self.nn_budget = None
        # 存储每帧各个视角间所有检测目标的匹配矩阵P
        self.matching_mat = None
        # 新创建轨迹应该具有的id号
        self.next_id = [1]
        # 该多视角视频的所有视角，是str的list
        self.view_ls = view_ls
        for view in view_ls:
            self.mvtrack_dict[view] = Tracker(nn_matching.NearestNeighborDistanceMetric(
        "cosine", self.max_cosine_distance, self.nn_budget), next_id=self.next_id)

    def update(self, matching_mat):
        # 更新该帧各个视角间所有检测目标的匹配矩阵P
        self.matching_mat = matching_mat