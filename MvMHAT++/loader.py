from torch.utils.data import Dataset
from collections import defaultdict
import os
import numpy as np
import cv2
import random
import re
import config as C
import sys
import torch
from torchvision import transforms


class Loader(Dataset):
    def __init__(self, views=2, frames=2, mode='train', dataset='1', detection_mode='origin'):
        self.views = views
        self.mode = mode
        self.dataset = dataset
        self.down_sample = 1
        # 数据集在2080ti上的位置
        # self.root_dir = '/dataset/gyy/mot/dataset'
        # 数据集在3090上的位置
        # self.root_dir = '/HDDs/sdd1/wff/dataset'
        # 数据集在新3090上位置
        self.root_dir = C.dataset_root_dir+'dataset'
        if sys.platform == 'win32':
            self.root_dir = 'E:\dataset\MVMOT\dataset'
        self.dataset_dir = os.path.join(self.root_dir, dataset)
        self.detection_mode=detection_mode

        self.my_transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5000, 0.5000, 0.5000],
                                                                   std=[0.5000, 0.5000, 0.5000])])
        # if self.mode == 'train':
        #     self.cut_dict = {
        #         '1': [0, 800],
        #         '2': [0, 800],
        #         '3': [0, 1000],
        #         '4': [0, 800],
        #         '5': [0, 1000],
        #         '6': [0, 1000],
        #         '7': [0, 2000],
        #         '8': [1000, 3000],
        #         '9': [2000, 4000],
        #         '10': [3000, 5000],
        #         '11': [500, 1500],
        #         '12': [500, 1500],
        #         '13': [1000, 3000],
        #     }
        # else:
        #     self.cut_dict = {
        #         '1': [800, 1200],
        #         '2': [800, 1200],
        #         '3': [1000, 1500],
        #         '4': [800, 1200],
        #         '5': [1000, 1500],
        #         '6': [1000, 1500],
        #         '7': [2000, 2500],
        #         '8': [3000, 4000],
        #         '9': [4000, 5000],
        #         '10': [5000, 6000],
        #         '11': [1500, 2000],
        #         '12': [1500, 2000],
        #         '13': [3000, 4000],
        #     }
        # 训练集划分
        if self.mode == 'train':
            self.cut_dict = {
                '1': [0, 800],
                '2': [0, 800],
                '3': [0, 1000],
                '4': [0, 800],
                '5': [0, 1000],
                '6': [0, 1000],
                '7': [0, 2000],
                '8': [1000, 3000],
                '9': [2000, 4000],
                '10': [3000, 5000],
                # '11': [2000, 4000],
                # '12': [500, 510],
                '12': [500, 1500],
                '13': [500, 1500],
                '14': [1000, 3000],
                # '15': [0, 795]
            }
        # 测试集划分
        else:
            self.cut_dict = {
                # '1': [800, 820],
                '1': [800, 1200],
                '2': [800, 1200],
                '3': [1000, 1500],
                '4': [800, 1200],
                '5': [1000, 1500],
                '6': [1000, 1500],
                '7': [2000, 2500],
                '8': [3000, 4000],
                '9': [4000, 5000],
                '10': [5000, 6000],
                # '11': [2000, 4001],
                '12': [1500, 2000],
                # '12': [1500, 1510],
                # '12': [0, 500],
                '13': [1500, 2000],
                '14': [3000, 4000],
                # '15': [0, 795]
            }

        if self.mode == 'train':
            self.frames = frames
            self.isShuffle = C.DATASET_SHUFFLE
            self.isCut = 1
        elif self.mode == 'test':
            self.frames = 1
            self.isShuffle = 0
            self.isCut = 1

        self.view_ls = os.listdir(self.dataset_dir)[:views]
        self.img_dict = self.gen_path_dict(False)
        self.anno_dict = self.gen_anno_dict()

    def gen_path_dict(self, drop_last: bool):
        path_dict = defaultdict(list)
        for view in self.view_ls:
            dir = os.path.join(self.dataset_dir, view, 'images')
            path_ls = os.listdir(dir)
            # path_ls.sort(key=lambda x: int(x[:-4]))
            path_ls.sort(key=lambda x: int(re.search(r"\d*", x).group()))
            path_ls = [os.path.join(dir, i) for i in path_ls]
            if self.isCut:
                start, end = self.cut_dict[self.dataset][0], self.cut_dict[self.dataset][1]
                path_ls = path_ls[start:end]
            if drop_last:
                path_ls = path_ls[:-1]
            cut = len(path_ls) % self.frames
            if cut:
                path_ls = path_ls[:-cut]
            if self.isShuffle:
                random.seed(self.isShuffle)
                random.shuffle(path_ls)
            path_dict[view] += path_ls
        # 再次分割，path_dict格式为 视角名称：list的映射，该list又是很多的list，每个list包含self.frame个帧的图片地址，用于保证之后训练时有：1.同视角不同帧2.同帧不同视角的图片
        path_dict = {view: [path_dict[view][i:i+self.frames] for i in range(0, len(path_dict[view]), self.frames)] for view in path_dict}
        return path_dict

    def gen_anno_dict(self):
        anno_dict = {}
        for view in self.view_ls:
            anno_view_dict = defaultdict(list)
            if self.mode == 'train':
                anno_path = os.path.join(self.dataset_dir, view, 'gt_det', 'anno.txt')
            elif self.mode == 'test':
                if self.detection_mode=='origin':
                    anno_path = os.path.join(self.dataset_dir, view, 'gt_det', 'det_res.txt')
                elif self.detection_mode=='yolox':
                    anno_path = '/HDDs/hdd3/wff/documents/bytetrack/results/mvmhat_mot17/'+self.dataset+'_'+view+'.txt'
            with open(anno_path, 'r') as anno_file:
                # print(anno_path)
                anno_lines = anno_file.readlines()
                for anno_line in anno_lines:
                    # if self.mode == 'train':
                    #     anno_line_ls = anno_line.split(',')
                    # else:
                    #     anno_line_ls = anno_line.split(' ')
                    # print(anno_line)
                    anno_line_ls = anno_line.split(',')
                    # anno_key即为帧号
                    anno_key = str(int(anno_line_ls[0]))
                    anno_view_dict[anno_key].append(anno_line_ls)
            anno_dict[view] = anno_view_dict
        # anno_dict的格式为 视角名称：dict，其中dict格式为 帧号：list ，list中包括着很多的list，其中每个list是该帧图片中某一个人的位置标注
        return anno_dict

    def read_anno(self, path: str):
        path_split = path.split('/')
        if sys.platform == 'win32':
            path_split = path.split('\\')
        view = path_split[-3]
        frame = str(int(re.search(r"\d*", path_split[-1]).group()))
        # annos是一个list包含着该帧中所有人物的标注
        annos = self.anno_dict[view][frame]
        bbox_dict = {}
        for anno in annos:
            bbox = anno[2:6]
            bbox = [max(int(float(i)),0) for i in bbox]
            bbox_dict[anno[1]] = bbox
        # bbox_dict返回一个dict，格式为人物id:标注框四个点坐标（int格式）
        return bbox_dict

    def crop_img(self, frame_img, bbox_dict):
        img = cv2.imread(frame_img)
        c_img_ls = []
        bbox_ls = []
        label_ls = []
        for key in bbox_dict:
            bbox = bbox_dict[key]
            bbox = [0 if i < 0 else i for i in bbox]
            # c_img_ls.append(img[bbox[0]:bbox[2], bbox[1]:bbox[3], :])
            # crop原始是高，宽，通道数
            crop = img[bbox[1]:bbox[3] + bbox[1], bbox[0]:bbox[2] + bbox[0], :]
            # 将crop的宽高统一变成224，同时变为 通道数，高，宽
            if C.model_type=='resnet':
                crop = cv2.resize(crop, (224, 224)).transpose(2, 0, 1).astype(np.float32)
                crop = torch.tensor(crop)
            elif C.model_type=='transformer':
                crop = self.my_transform(cv2.resize(crop, (224, 224)))
            else:
                print('wrong model type!')
            # c_img_ls存储着剪裁且resize后的图片
            c_img_ls.append(crop)
            # bbox_ls存储着该人框的位置
            bbox_ls.append(bbox)
            # label_ls存储着该人的id
            label_ls.append(key)
        return torch.stack(c_img_ls), bbox_ls, label_ls, frame_img

    def __len__(self):
        # return self.len
        # 返回所有视角中最少组的数目作为len
        return min([len(self.img_dict[i]) for i in self.view_ls] + [10000])

    def __getitem__(self, item):
        ret = []
        # img_ls中存储着所有视角中第item对的图像地址集
        img_ls = [self.img_dict[view][item] for view in self.view_ls]

        for img_view in img_ls:
            view_ls = []
            # img是图片的完整路径
            for img in img_view:
                anno = self.read_anno(img)
                if anno == {}:
                    # 训练过程中若某一帧没有人，则用上一帧的代替
                    if self.mode == 'train':
                        return self.__getitem__(item - 1)
                    # 训练时若该帧没有人则跳过
                    else:
                        view_ls.append([])
                        continue
                # self.crop_img虽然返回四个元素但会自动组成一个tuple
                view_ls.append(self.crop_img(img, anno))
            ret.append(view_ls)
        # 返回的ret是个list，里面共有总共视角数个list，每个list中有self.frame个tuple，该tuple是四元tuple，包括截取人图片，框位置，人id和图片路径
        return ret



if __name__ == '__main__':
    a = Loader(views=4, frames=2, mode='train', dataset='1',detection_mode='origin')
    # for i in range(a[0][0][0][0].shape[0]):
    #     pic = a[0][0][0][0][i, :, :, :].transpose(1, 2, 0)
    #     cv2.imwrite('/HDDs/hdd3/wff/documents/dhn_tracking_network2_origin_inference/ourdatatset_test' + str(i) + '.png',pic)
    for i in enumerate(a):
        pass
    print('finished!')






