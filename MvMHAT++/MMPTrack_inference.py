from __future__ import division, print_function, absolute_import
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from mmptrack_loader import MMPTrack_Loader
import torchvision.models as models
from deep_sort.mvtracker import MVTracker
from deep_sort.update import Update
from torch.cuda.amp import autocast as autocast
import config as C
from collections import defaultdict
import argparse
import sys
from dhn import Munkrs
from resnet_dhn_model import resnet_dhn_model


parser = argparse.ArgumentParser()
# parser.add_argument('--model', default=C.INF_ID)
parser.add_argument('--use_final_matrix', default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = C.TRAIN_GPUS
# resume = "./models/" + args.model + '.pth'
# TODO here
datasets_viewnum_dict = {'cafe_shop_0': 4, 'industry_safety_0': 4, 'lobby_0': 4, 'office_0': 5}
# datasets_viewnum_dict = { 'industry_safety_0': 4}

def read_loader(dataset_name):
    dataset = MMPTrack_Loader(views=datasets_viewnum_dict[dataset_name], frames=1, mode='test', dataset=dataset_name, time='64pm',detection_mode='origin')
    dataset_loader = DataLoader(dataset, num_workers=4, pin_memory=True)
    dataset_info = {
        'view': [str(i) for i in dataset.views],
        'seq_len': len(dataset),
        'start': dataset.cut_dict[dataset_name][0],
        'end': dataset.cut_dict[dataset_name][1]
    }
    return dataset_info, dataset_loader

def gather_seq_info_multi_view(dataset,dataset_info, dataset_test, model):

    groundtruth = None
    seq_dict = {}
    coffidence = 1
    print('loading dataset '+dataset+'...')

    image_filenames = defaultdict(list)
    detections = defaultdict(list)
    for data_i, data in tqdm(enumerate(dataset_test), total=len(dataset_test)):
        for view_i, view in enumerate(dataset_info['view']):
            data_pack = data[view_i][0]
            if data_pack == []:
                continue
            # if data_i > 50 and view_i == 3: break
            # img是该帧所有人的裁剪图片，box每个人的标注框位置，lbl是每个人的id（并非与前帧对应），scn该帧的图片路径
            img, box, lbl, scn = data[view_i][0]
            model.eval()
            with torch.no_grad():
                img = img.squeeze(0).cuda()
                # with autocast():
                img = model(img)
            image_filenames[view].append(scn[0])
            for feature, bndbox, id in zip(img, box, lbl):
                if sys.platform == 'win32':
                    # index = int(scn[0].split('\\')[-1][:-4]) - 1
                    index = int(scn[0].split('\\')[-1].split('_')[1])
                else:
                    # index = int(scn[0].split('/')[-1][:-4]) - 1
                    index=int(scn[0].split('/')[-1].split('_')[1])

                bndbox[2]-=bndbox[0]
                bndbox[3]-=bndbox[1]
                bndbox = [int(i) for i in bndbox]
                id = int(id[0])
                det = [index] + [id] + bndbox + [coffidence] + [0, 0, 0] + feature.detach().cpu().numpy().tolist()
                # det格式为图片id，人物id，标注框的4个位置，置信度（固定为1），三个0（应该没什么用），1000维的该人物特征向量
                detections[view].append(det)

    for view_i, view in enumerate(dataset_info['view']):
        seq_dict[view] = {
        "sequence_name": 'test',
        "image_filenames": image_filenames[view],
        "detections": np.array(detections[view]),
        "groundtruth": groundtruth,
        "image_size": (3, 360, 640),
        "min_frame_idx": dataset_info['start'],
        "max_frame_idx": dataset_info['end'] - 1,
        "feature_dim": 1000,
        "update_ms": 10
        }
    return seq_dict


def run(output_file, display, dataset, model):
    dataset_info, dataset_loader = read_loader(dataset)
    seq_mv = gather_seq_info_multi_view(dataset,dataset_info, dataset_loader, model[0])

    # if args.use_final_matrix:
    #     output_file += 'use_final_matrix/'
    # else:
    #     output_file += 'not_use_final_matrix/'

    mvtracker = MVTracker(dataset_info['view'])
    if len(model)==2:
        updater = Update(seq=seq_mv, mvtracker=mvtracker, display=display, model=model[1])
    else:
        updater = Update(seq=seq_mv, mvtracker=mvtracker, display=display, model=None)

    # 跑完所有视角所有帧
    updater.run()
    for view in updater.view_ls:
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        # if args.use_final_matrix:
        #     output_path=output_file + dataset + '_' + view +'_use_final_matrix.txt'
        # else:
        #     output_path=output_file + dataset + '_' + view +'_not_use_final_matrix.txt'
        output_path = output_file + dataset + '_' + view + '.txt'
        f = open(output_path, 'w')
        # 输出路径
        print(output_path)
        for row in updater.result[view]:
            # 前六个值为frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,-1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
        f.close()

if __name__ == "__main__":
    print('Inference global threshold is: '+str(C.inference_global_threshold))
    if C.inference_time_consider_filter:
        print('Inference time assign use Kalman filtering!')
        print('Inference time simliarity threshold is: '+str(C.inference_time_simliarity_threshold))
    else:
        print('Inference time assign does not use Kalman filtering!')
    resnet_dhn_model = resnet_dhn_model(resume=True, use_softmax='temperature_softmax')
    # 测试文件夹中指定epoch的模型
    model_list = [name for name in os.listdir(C.test_folder)]
    # TODO 该输出路径要改
    output_dir=C.test_folder.split('/')[-2]

    if C.INF_ID is None:
        print('Inference from epoch:'+str(C.start_epoch)+' to epoch:'+str(C.end_epoch))
        for epoch_i in range(C.start_epoch,C.end_epoch+1):
            name_list=[name for name in model_list if name.split('.')[0].split('_')[-2]==str(epoch_i)]
            if len(name_list)>0:
                model_name=name_list[0]
                ckp=torch.load(C.test_folder+model_name)['model']
                resnet_dhn_model.load_state_dict(ckp)
                print('model: '+model_name.split('.')[0]+'.'+model_name.split('.')[1])
                for dataset_name in datasets_viewnum_dict.keys():
                    # run(output_file="/HDDs/hdd3/wff/documents/dhn_tracking_network2_origin_inference/MMPTrack_output/" + output_dir+'/'+model_name.split('.')[0]+'.'+model_name.split('.')[1] + "/", display=C.DISPLAY, dataset=dataset_name,model=[resnet_dhn_model.resnet_model, resnet_dhn_model.dhn_model])
                    run(output_file="./MMPTrack_output/" + output_dir + "_" + C.inference_mode + '/'+model_name.split('.')[0]+'.'+model_name.split('.')[1] + "_" + C.inference_mode + "/", display=C.DISPLAY,
                        dataset=dataset_name, model=[resnet_dhn_model.feature_extractor, resnet_dhn_model.dhn_model])
            else:
                continue
    else:
        # TODO 测试单个模型
        ckp = torch.load(C.test_folder + C.INF_ID + '.pth')['model']
        resnet_dhn_model.load_state_dict(ckp)
        print('model: '+C.INF_ID)
        for dataset_name in datasets_viewnum_dict.keys():
            run(output_file="./MMPTrack_output/" + C.INF_ID + "_" + C.inference_mode + "/", display=C.DISPLAY,
                dataset=dataset_name, model=[resnet_dhn_model.feature_extractor, resnet_dhn_model.dhn_model])

