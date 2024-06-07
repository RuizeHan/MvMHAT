import os
from collections import defaultdict
import numpy as np
from Tools.linear_assignment import sklearn_linear_assignment
import math

dataset_view_num_dict={
    'cafe_shop_0': 4,
    'industry_safety_0': 4,
    'lobby_0': 4,
    'office_0': 5,
}

for i in range(1,15):
    if i==6 or i==8 or i==10:
        dataset_view_num_dict[str(i)]=3
    elif i!=11:
        dataset_view_num_dict[str(i)]=4

test_frame_dict={
    # todo recover
    # 'cafe_shop_0': [0, 7521],
    # 'industry_safety_0': [0, 7361],
    # 'lobby_0': [0, 6325],
    # 'office_0': [0, 6300],
    'cafe_shop_0': [0, 1000],
    'industry_safety_0': [0, 1000],
    'lobby_0': [0, 1000],
    'office_0': [0, 1000],
    'retail_0': [0, 1000],
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
    '12': [1500, 2000],
    '13': [1500, 2000],
    '14': [3000, 4000]
}


def calc_f1_loss(y_true , y_pred):

    tp = np.sum(y_true * y_pred)
    tn = np.sum((1 - y_true) * (1 - y_pred))
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)
    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)

def get_view_frame_id_bbox_dict(pred_base_path,gt_base_path,dataset_name):
    gt_view_dict={}
    pred_view_dict={}
    for view in range(dataset_view_num_dict[dataset_name]):
        view_id=str(view+1)
        # 先生成gt dict
        gt_txt_path=gt_base_path+dataset_name+'_'+view_id+'.txt'
        with open(gt_txt_path,'r') as f:
            frame_dict=defaultdict(list)
            lines=f.readlines()
            for line in lines:
                line=line.split(',')
                frame_id=int(line[0])
                pid=float(line[1])
                x,y,w,h=float(line[2]),float(line[3]),float(line[4]),float(line[5])
                frame_dict[frame_id].append([pid,x,y,w,h])
        gt_view_dict[view_id]=frame_dict

        # 生成pred dict
        pred_txt_path=pred_base_path+dataset_name+'_'+view_id+'.txt'
        with open(pred_txt_path,'r') as f:
            frame_dict=defaultdict(list)
            lines=f.readlines()
            for line in lines:
                line=line.split(',')
                frame_id = int(line[0])
                pid = float(line[1])
                x, y, w, h = float(line[2]), float(line[3]), float(line[4]), float(line[5])
                frame_dict[frame_id].append([pid, x, y, w, h])
        pred_view_dict[view_id]=frame_dict
    return pred_view_dict, gt_view_dict

def get_pred_gt_match(pred_view_dict,gt_view_dict,dataset_name,iou_threshold=0.5):
    # 上pred id-> 下gt id
    pred_gt_match_dict={}
    start_frame,end_frame=test_frame_dict[dataset_name][0],test_frame_dict[dataset_name][1]
    for view in range(dataset_view_num_dict[dataset_name]):
        view_id=str(view+1)
        frame_dict={}
        for frame_id in range(start_frame,end_frame+1):
            if frame_id in pred_view_dict[view_id].keys():
                pred_mat=np.array(pred_view_dict[view_id][frame_id])
            else:
                pred_mat=np.array([])
            if frame_id in gt_view_dict[view_id].keys():
                gt_mat=np.array(gt_view_dict[view_id][frame_id])
            else:
                gt_mat=np.array([])
            if len(pred_mat)==0 and len(gt_mat)>0:
                gt_id=gt_mat[:,0]
                pred_id=np.array([-1]*len(gt_id))
                pred_gt_match=np.vstack((pred_id,gt_id))
            elif len(gt_mat)==0 and len(pred_mat)>0:
                pred_id=pred_mat[:,0]
                gt_id=np.array([-1]*len(pred_id))
                pred_gt_match=np.vstack((pred_id,gt_id))
            elif len(gt_mat)==0 and len(pred_mat)==0:
                pred_gt_match=np.array([])
            else:
                # 列表示pred,行表示gt
                frame_iou_mat=[]
                pred_id=pred_mat[:,0]
                gt_id=gt_mat[:,0]
                for one_pred in pred_mat:
                    frame_iou_mat.append(iou(one_pred[1:],gt_mat[:,1:]))
                frame_iou_mat=np.array(frame_iou_mat)
                frame_iou_mat[frame_iou_mat<=iou_threshold]=0
                assign_ls = sklearn_linear_assignment(-frame_iou_mat)
                pair_pred=[]
                pair_gt=[]
                for as_one in assign_ls:
                    if frame_iou_mat[as_one[0], as_one[1]] > 0:
                        pair_pred.append(pred_id[as_one[0]])
                        pair_gt.append(gt_id[as_one[1]])
                pred_left=list(set(pred_id)-set(pair_pred))
                gt_left=list(set(gt_id)-set(pair_gt))
                pair_pred+=pred_left
                pair_gt+=[-1]*len(pred_left)
                pair_pred+=[-1]*len(gt_left)
                pair_gt+=gt_left
                pred_gt_match=np.vstack((pair_pred,pair_gt))
            assert (len(pred_gt_match)==0 or (len(pred_gt_match[0])==len(pred_gt_match[1])))
            frame_dict[frame_id]=pred_gt_match
        pred_gt_match_dict[view_id]=frame_dict

    return pred_gt_match_dict

def cal_time_view_mat_simliarity(start_frame,end_frame,frame_len,pred_gt_match_dict,step_ratio=0.5):
    step_start_frame=start_frame
    # 拼接顺序如下：视角1帧1，视角1帧2；视角2帧1，视角2帧2......
    num=0
    result=0
    while step_start_frame <= end_frame:
        # 构成view个视角，frame_len帧的矩阵
        step_end_frame=min(step_start_frame+frame_len-1, end_frame)
        pred_id=np.array([])
        gt_id=np.array([])
        for view_id in pred_gt_match_dict.keys():
            for frame_id in range(step_start_frame,step_end_frame+1):
                if len(pred_gt_match_dict[view_id][frame_id])>0:
                    one_pred=pred_gt_match_dict[view_id][frame_id][0]
                    one_gt=pred_gt_match_dict[view_id][frame_id][1]
                    pred_id=np.hstack((pred_id,one_pred))
                    gt_id=np.hstack((gt_id,one_gt))
        step_pred_matrix = np.float32(pred_id.reshape(1, len(pred_id)) == pred_id.reshape(len(pred_id), 1))
        step_gt_matrix = np.float32(gt_id.reshape(1,len(gt_id))==gt_id.reshape(len(gt_id),1))
        for index in range(len(gt_id)):
            if pred_id[index]==-1:
                # TODO 对FN用0填值
                step_pred_matrix[index,:]=0
                step_pred_matrix[:,index]=0
            if gt_id[index]==-1:
                # TODO 对FP用0填值
                step_gt_matrix[index,:]=0
                step_gt_matrix[:,index]=0

        # 接下来计算pred矩阵和gt矩阵的相似度
        # TODO 方法1：iou
        # jiao_mat=step_gt_matrix*step_pred_matrix
        # bing_mat=np.float32((step_gt_matrix+step_pred_matrix)>0)
        # result=np.sum(jiao_mat)/np.sum(step_gt_matrix)

        # TODO 方法2：F1
        result+=calc_f1_loss(y_true=step_gt_matrix,y_pred=step_pred_matrix)
        num+=1

        step_start_frame=step_start_frame+math.floor(frame_len*step_ratio)
    return result/num

def cal_new_metric(pred_base_path,gt_base_path,dataset_name):
    method_list=os.listdir(pred_base_path)
    if dataset_name == 'mvmhat':
        dataset_name_list = [str(i) for i in range(1, 15) if i != 11]
    elif dataset_name == 'MMPTrack':
        dataset_name_list = ['cafe_shop_0', 'industry_safety_0', 'lobby_0', 'office_0']
    else:
        print('wrong dataset!')
        return

    for method in method_list:
        print('method: '+ method)
        time_window_30_results=0
        time_window_5_results=0
        time_window_10_results=0

        for dataset_name in dataset_name_list:
            # print('processing dataset '+dataset_name)
            start_frame,end_frame=test_frame_dict[dataset_name][0],test_frame_dict[dataset_name][1]
            pred_view_dict, gt_view_dict = get_view_frame_id_bbox_dict(pred_base_path+method+'/', gt_base_path, dataset_name)
            pred_gt_match_dict = get_pred_gt_match(pred_view_dict, gt_view_dict, dataset_name)

            temp30=cal_time_view_mat_simliarity(start_frame,end_frame,30,pred_gt_match_dict)
            temp5=cal_time_view_mat_simliarity(start_frame, end_frame, 5, pred_gt_match_dict)
            temp10=cal_time_view_mat_simliarity(start_frame, end_frame, 10, pred_gt_match_dict)
            time_window_30_results += temp30
            time_window_5_results += temp5
            time_window_10_results += temp10

            # print(dataset_name + ' time 5: ' + str(temp5))
            # print(dataset_name + ' time 10: ' + str(temp10))
            # print(dataset_name + ' time 30: ' + str(temp30))

        print('Time window 5 f1 result: '+str(time_window_5_results/len((dataset_name_list))))
        print('Time window 10 f1 result: ' + str(time_window_10_results / len((dataset_name_list))))
        print('Time window 30 f1 result: ' + str(time_window_30_results / len((dataset_name_list))))
        print('')

if __name__ == '__main__':
    # TODO dataset='mvmhat', 'MMPTrack'
    dataset='MMPTrack'
    # TODO 用于测试mvmhat数据集
    # pred_base_path = '/HDDs/sdd1/wff/work1_result/mvmhat/'
    # TODO 用于测试MMPTrack数据集
    # pred_base_path ='/data/wff/documents/tmp/'
    # pred_base_path ='/data/wff/documents/tmp_mmp/'
    pred_base_path = '/HDDs/sdd1/wff/work1_result/MMPTrack/'
    gt_base_path='/data1/wff/documents/dataset_compine_gt/'
    # gt_base_path='/HDDs/sdd1/wff/dataset_compine_gt/'
    # TODO recover
    cal_new_metric(pred_base_path, gt_base_path, dataset)
    # cal_new_metric(pred_base_path,gt_base_path,pred_base_path.split('/')[-2])

    # pred_view_dict, gt_view_dict=get_view_frame_id_bbox_dict(pred_base_path,gt_base_path,'4')
    # pred_gt_match_dict=get_pred_gt_match(pred_view_dict,gt_view_dict,'4')
    # cal_time_view_mat_simliarity(800,1200,5,pred_gt_match_dict)