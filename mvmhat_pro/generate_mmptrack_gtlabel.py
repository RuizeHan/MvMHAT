import json
import os
from collections import defaultdict

# TODO recover
# datasets_viewnum_dict={'cafe_shop_0':4,'industry_safety_0':4,'lobby_0':4,'office_0':5,'retail_0':6}
dataset_frame={
    'cafe_shop_0': [0, 7521],
    'industry_safety_0': [0, 7361],
    'lobby_0': [0, 6325],
    'office_0': [0, 6300],
    }
datasets_viewnum_dict={'cafe_shop_0': 4, 'industry_safety_0': 4, 'lobby_0': 4, 'office_0': 5}
# label_section=[0,7361]
label_path='/HDDs/sdd1/wff/MMPTracking/test/labels/64pm/'
write_path='/HDDs/sdd1/wff/MMPTracking/test_gt_longterm_labels/'
pseduo_write_path='/HDDs/sdd1/wff/MMPTracking/test_pseduo_longterm_labels/'

def generate_gt_label(start_frame,end_frame,dataset_name,write_path):
    for view in range(1,datasets_viewnum_dict[dataset_name]+1):
        view_dict=defaultdict(list)
        start,end=start_frame,end_frame
        for frame_id in range(start,end+1):
            json_path=label_path+dataset_name+'/rgb_'+str(frame_id).zfill(5)+'_'+str(view)+'.json'
            with open(json_path,'r') as f:
                anno_json=json.load(f)
                for k,v in anno_json.items():
                    if int(float(v[0])) < 640 and int(float(v[1])) < 360:
                        bbox=[max(0, int(float(v[0]))), max(0, int(float(v[1]))),min(640, int(float(v[2]))), min(360, int(float(v[3])))]
                        bbox[2]-=bbox[0]
                        bbox[3]-=bbox[1]
                        view_dict[k].append([frame_id]+[int(k)]+bbox+[-1,-1,-1,-1])
        # 写文件
        with open(write_path+dataset_name+'_'+str(view)+'.txt','w',encoding="utf-8") as f:
            ids_key=[int(i) for i in list(view_dict.keys())]
            ids_key.sort()
            for i in ids_key:
                id_list=view_dict[str(i)]
                for one in id_list:
                    for index,index_one in enumerate(one):
                        one[index]=str(index_one)
                    line=",".join(one)+'\n'
                    f.write(line)

def generate_pseduo_label(start_frame,end_frame,dataset_name,pseduo_write_path):
    for view in range(1,datasets_viewnum_dict[dataset_name]+1):
        view_list=[]
        for frame_id in range(start_frame,end_frame+1):
            json_path=label_path+dataset_name+'/rgb_'+str(frame_id).zfill(5)+'_'+str(view)+'.json'
            with open(json_path,'r') as f:
                anno_json=json.load(f)
                ids_key=[int(i) for i in list(anno_json.keys())]
                ids_key.sort()
                for k in ids_key:
                    bbox=anno_json[str(k)]
                    if int(float(bbox[0])) < 640 and int(float(bbox[1])) < 360:
                        bbox=[max(0, int(float(bbox[0]))), max(0, int(float(bbox[1]))),min(640, int(float(bbox[2]))), min(360, int(float(bbox[3])))]
                        bbox[2]-=bbox[0]
                        bbox[3]-=bbox[1]
                        view_list.append([frame_id]+[k]+bbox+[-1,-1,-1,-1])
        # 开始写文件
        with open(pseduo_write_path + dataset_name + '_' + str(view) + '.txt', 'w', encoding="utf-8") as f:
            for one in view_list:
                for index, index_one in enumerate(one):
                    one[index] = str(index_one)
                line = ",".join(one) + '\n'
                f.write(line)

if __name__ == '__main__':
    if not os.path.exists(write_path):
        os.mkdir(write_path)
    if not os.path.exists(pseduo_write_path):
        os.mkdir(pseduo_write_path)
    for dataset_name in datasets_viewnum_dict.keys():
        generate_gt_label(dataset_frame[dataset_name][0],dataset_frame[dataset_name][1],dataset_name,write_path)
        generate_pseduo_label(dataset_frame[dataset_name][0],dataset_frame[dataset_name][1],dataset_name,pseduo_write_path)