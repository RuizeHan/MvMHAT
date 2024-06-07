from torch.utils.data import Dataset
from collections import defaultdict
import os
import numpy as np
import cv2
import random
import re
import config as C
import sys
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

class MMPTrack_Loader(Dataset):
    def __init__(self,views=4,frames=2,mode='train',dataset='cafe_shop_0',time='64am',detection_mode='origin'):
        if dataset=='retail_0':
            self.views=[1,4,5,6]
        else:
            self.views=[i for i in range(1,views+1)]
        self.mode=mode
        self.dataset=dataset
        self.root_dir=C.dataset_root_dir+'MMPTracking/'
        # self.root_dir='/data1/wff/dataset/MMP-MvMHAT/'
        self.images_path=self.root_dir+mode+'/images/'+time+'/'+dataset+'/'
        self.detection_mode=detection_mode
        # 训练时所使用的gt检测框
        self.labels_path=self.root_dir+mode+'/labels/'+time+'/'+dataset+'/'
        # 测试时所使用的检测框
        # TODO here
        self.dets_path='/HDDs/hdd3/wff/documents/YOLOX-main/YOLOX_outputs/yolox_x/label_res/64pm/'

        self.my_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=torch.tensor([0.5000, 0.5000, 0.5000]),
                                                                     std=torch.tensor([0.5000, 0.5000, 0.5000]))])


        # 训练集划分（暂定）
        if self.mode=='train':
            self.cut_dict={
                'cafe_shop_0':[0,2000],
                'industry_safety_0':[0,2000],
                'lobby_0':[0,2000],
                'office_0':[0,2000],
                'retail_0':[0,2000]
            }
        # 测试集划分（暂定）
        else:
            self.cut_dict={
                'cafe_shop_0': [0, 1000],
                'industry_safety_0': [0, 1000],
                'lobby_0': [0, 1000],
                'office_0': [0, 1000],
                'retail_0': [0, 1000]
            }

        if self.mode=='train':
            self.frames=frames
            self.isShuffle=C.DATASET_SHUFFLE

        elif self.mode=='test':
            self.frames=1
            self.isShuffle=False

        self.img_dict=self.gen_path_dict()
        self.anno_dict=self.gen_anno_dict()


    def gen_path_dict(self):
        start,end=self.cut_dict[self.dataset][0],self.cut_dict[self.dataset][1]
        path_dict= {}
        for view_id in self.views:
            path_ls=[]
            for frame_id in range(start, end+1):
                path_ls.append(self.images_path+'rgb_'+str(frame_id).zfill(5)+'_'+str(view_id)+'.jpg')
            cut=len(path_ls) % self.frames
            if cut:
                path_ls=path_ls[:-cut]
            if self.isShuffle:
                random.seed(self.isShuffle)
                random.shuffle(path_ls)
            path_dict[str(view_id)]=path_ls
        path_dict={view:[path_dict[view][i:i+self.frames] for i in range(0,len(path_dict[view]),self.frames)] for view in path_dict.keys()}
        return path_dict

    def gen_anno_dict(self):
        start, end = self.cut_dict[self.dataset][0], self.cut_dict[self.dataset][1]
        anno_dict={}
        for view_id in self.views:
            if self.detection_mode=='origin':
                anno_view_dict = {}
            elif self.detection_mode=='yolox':
                anno_view_dict=defaultdict(dict)
            if self.mode=='train':
                anno_path=self.labels_path
            elif self.mode=='test':
                # TODO 修改here
                # anno_path=self.dets_path
                # 测试暂时先用gt标注
                if self.detection_mode=='origin':
                    anno_path=self.labels_path
                elif self.detection_mode=='yolox':
                    anno_path=self.dets_path
            if self.detection_mode=='origin':
                for frame_id in range(start,end+1):
                    with open(anno_path+'rgb_'+str(frame_id).zfill(5)+'_'+str(view_id)+'.json','r') as f:
                        anno_json=json.load(f)
                        my_anno_dict={}
                        for k,v in anno_json.items():
                            if int(float(v[0]))<640 and int(float(v[1]))<360:
                                my_anno_dict[k]=[max(0,int(float(v[0]))),max(0,int(float(v[1]))),min(640,int(float(v[2]))),min(360,int(float(v[3])))]
                        anno_view_dict[str(frame_id).zfill(5)]=my_anno_dict
                anno_dict[str(view_id)]=anno_view_dict

            elif self.detection_mode=='yolox':
                with open(anno_path+self.dataset+'_'+str(view_id)+'.txt', 'r') as anno_file:
                    anno_lines = anno_file.readlines()
                    id=1
                    for anno_line in anno_lines:
                        anno_line_ls = anno_line.split(',')
                        # anno_key即为帧号
                        anno_key = str(int(anno_line_ls[0])).zfill(5)
                        pid=anno_line_ls[1]
                        bbox=anno_line_ls[2:6]
                        bbox=[max(int(float(i)),0) for i in bbox]
                        bbox[2] += bbox[0]
                        bbox[3] += bbox[1]
                        anno_view_dict[anno_key][str(id)]=bbox
                        id+=1
                anno_dict[str(view_id)] = anno_view_dict
        return anno_dict

    def read_anno(self,img_path):
        img_name=img_path.split('/')[-1]
        frame_id=img_name.split('_')[-2]
        view_id=img_name.split('_')[-1].split('.')[0]
        anno=self.anno_dict[view_id][frame_id]
        return anno

    def crop_img(self, frame_img, bbox_dict):
        img = cv2.imread(frame_img)
        c_img_ls = []
        bbox_ls = []
        label_ls = []
        for key in bbox_dict:
            bbox = bbox_dict[key]
            # c_img_ls.append(img[bbox[0]:bbox[2], bbox[1]:bbox[3], :])
            # crop原始是高，宽，通道数
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            # 将crop的宽高统一变成224，同时变为 通道数，高，宽
            if C.model_type == 'resnet':
                crop = cv2.resize(crop, (224, 224)).transpose(2, 0, 1).astype(np.float32)
                crop = torch.tensor(crop)
            elif C.model_type == 'transformer':
                crop = self.my_transform(cv2.resize(crop, (224, 224)))
            else:
                print('wrong model type!')
            # c_img_ls存储着剪裁且resize后的图片
            c_img_ls.append(crop)
            # bbox_ls存储着该人框的位置
            bbox_ls.append(bbox)
            # label_ls存储着该人的id
            label_ls.append(key)
        if self.mode=='test':
            return torch.stack(c_img_ls),bbox_ls,label_ls,frame_img
        elif self.mode=='train':
            # 对标签顺序打乱
            assert(len(c_img_ls)==len(bbox_ls)==len(label_ls))
            shuffle_index=[i for i in range(len(label_ls))]
            random.shuffle(shuffle_index)
            random_img_ls=[]
            random_bbox_ls=[]
            random_label_ls=[]
            for i in range(len(label_ls)):
                random_img_ls.append(c_img_ls[shuffle_index[i]])
                random_bbox_ls.append(bbox_ls[shuffle_index[i]])
                random_label_ls.append(label_ls[shuffle_index[i]])
            return torch.stack(random_img_ls), random_bbox_ls, random_label_ls, frame_img


    def __len__(self):
        return min([len(self.img_dict[k]) for k in self.img_dict.keys()])

    def __getitem__(self, item):
        ret=[]
        img_ls=[self.img_dict[view][item] for view in self.img_dict.keys()]
        for img_view in img_ls:
            view_ls=[]
            for img in img_view:
                anno=self.read_anno(img)
                if anno=={}:
                    if self.mode=='train':
                        return self.__getitem__(item-1)
                    else:
                        view_ls.append([])
                        continue
                view_ls.append(self.crop_img(img, anno))
            ret.append(view_ls)
        return ret

if __name__ == '__main__':
    train_times=['63am','64am']
    test_times=['64pm']
    # TODO recover
    # datasets_viewnum_dict={'cafe_shop_0':4,'industry_safety_0':4,'lobby_0':4,'office_0':5,'retail_0':6}
    datasets_viewnum_dict = {'cafe_shop_0':4,'industry_safety_0':4,'lobby_0':4,'office_0':5}
    # 检查所有训练数据
    # print('checking train data!')

    # --------------new_begin--------------------
    # datasets = []
    # for time in train_times:
    #     for dataset_name in datasets_viewnum_dict.keys():
    #         datasets.append(MMPTrack_Loader(views=datasets_viewnum_dict[dataset_name], frames=C.FRAMES, mode='train',
    #                                         dataset=dataset_name, time=time))
    # datasets = ConcatDataset(datasets)
    # dataset_train = DataLoader(datasets, num_workers=0, pin_memory=True, shuffle=C.LOADER_SHUFFLE)
    # for step_i, data in tqdm(enumerate(dataset_train), total=len(dataset_train)):
    #     pass
    # -------------new_end-----------------------

    # for time in train_times:
    #     for dataset in datasets_viewnum_dict.keys():
    #         a=MMPTrack_Loader(views=datasets_viewnum_dict[dataset],mode='train',dataset=dataset,time=time)
    #         for i in range(a[0][0][0][0].shape[0]):
            #     pic=a[0][0][0][0][i,:,:,:].transpose(1, 2, 0)
            #     cv2.imwrite('/HDDs/hdd3/wff/documents/dhn_tracking_network2_origin_inference/mmptrack_test'+str(i)+'.png',pic)
            # for i in enumerate(a):
            #     pass
            # print(time+' '+dataset+' finished!')

    # 检查所有测试数据
    print('checking test data!')

    # ------------new_begin-----------------------------
    # datasets = []
    # for time in test_times:
    #     for dataset_name in datasets_viewnum_dict.keys():
    #         datasets.append(MMPTrack_Loader(views=datasets_viewnum_dict[dataset_name], frames=1, mode='test',
    #                                         dataset=dataset_name, time=time, detection_mode='origin'))
    # datasets = ConcatDataset(datasets)
    # dataset_test = DataLoader(datasets, num_workers=4, pin_memory=True, shuffle=False)
    # for step_i, data in tqdm(enumerate(dataset_test), total=len(dataset_test)):
    #     pass
    # ------------new_end-------------------------------

    count=0
    for time in test_times:
        for dataset in datasets_viewnum_dict.keys():
            a = MMPTrack_Loader(views=datasets_viewnum_dict[dataset], mode='test', dataset=dataset, time=time,detection_mode='origin')
            for i in enumerate(a):
                count+=1
            print(time+' '+dataset+' finished!')
    print(count)

