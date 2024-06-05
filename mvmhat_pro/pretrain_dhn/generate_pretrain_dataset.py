import numpy as np
import random
import math
from tqdm import tqdm


# 对于标签情况要有5%概率干扰矩阵是错误的
def generate_gt_label(view_num,total_person_num_min,total_person_num_max,each_view_num_ratio_min,out_num_max,in_num_max):
    # 3-6个视角(TODO 改为3-4个视角），每个视角5-15人
    each_num=[]
    each_id=[]
    # 先生成场景中的总人数
    total_person_num=random.randint(total_person_num_min,total_person_num_max)
    each_view_person_num_min=math.floor(total_person_num*each_view_num_ratio_min)
    # id_list=np.array([(i+1) for i in range(total_person_num)])
    id_list = [(i + 1) for i in range(total_person_num)]
    # 拼接顺序如下：视角1帧1，视角1帧2；视角2帧1，视角2帧2......
    for i in range(view_num):
        # 先生成时刻1时刻的id_list
        time1_viewi_num=random.randint(each_view_person_num_min,total_person_num)
        time1_select_list=id_list.copy()
        random.shuffle(time1_select_list)
        time1_viewi_idlist=time1_select_list[:time1_viewi_num]
        # 生成时刻2的id_list
        time2_select_list=time1_viewi_idlist.copy()
        out_view_num=random.randint(0,min(out_num_max,time1_viewi_num))
        select_in_idlist=list(set(id_list)-set(time1_viewi_idlist))
        random.shuffle(select_in_idlist)
        in_view_num=random.randint(0,min(in_num_max,len(select_in_idlist)))
        random.shuffle(time2_select_list)
        after_out_idlist=time2_select_list[out_view_num:]
        time2_viewi_idlist=after_out_idlist+select_in_idlist[:in_view_num]
        random.shuffle(time2_viewi_idlist)
        each_id.extend(time1_viewi_idlist)
        each_id.extend(time2_viewi_idlist)
        each_num.append(len(time1_viewi_idlist))
        each_num.append(len(time2_viewi_idlist))

    each_id=np.array(each_id)
    GT_matrix = np.float32(each_id.reshape(1,len(each_id)) == each_id.reshape(len(each_id),1))
    return GT_matrix,each_num


def generate_noise_matrix(GT_matrix,each_num,all_zero_noise_ratio,one_noise_min,one_noise_max,diagonal_one_noise_ratio,false_ratio,zero_false_case_max,one_false_case_max):
    # np.split特性，each_num需要是切分的位置；tensor不需要
    sum=0
    each_len=[]
    for i in each_num[:-1]:
        sum+=i
        each_len.append(sum)
    #
    row = np.split(GT_matrix, each_len, axis=0)
    dis_row_col = [list(np.split(i, each_len, axis=1)) for i in row]
    for index_i in range(len(dis_row_col)):
        for index_j in range(len(dis_row_col)):
            one=dis_row_col[index_i][index_j]
            GT=one.copy()
            for i in range(one.shape[0]):
                hang=one[i]
                sum=np.sum(hang)
                # 全0情况
                if sum==0.0:
                    random_case=random.randint(1,100000)
                    # 落入极端情况
                    if random_case <= 100000*false_ratio:
                        index=[i for i in range(len(hang))]
                        random.shuffle(index)
                        max_index=index[0]
                        max=random.uniform(0.4,zero_false_case_max)
                        one[i][max_index]=max
                        left=1.0-max
                        for j in range(1,len(hang)-1):
                            noise=random.uniform(0,left)
                            one[i][index[j]]=noise
                            left-=noise
                        one[i][index[-1]]=left

                    # 没落入极端情况
                    else:
                        one[i]=1.0/(len(hang))
                        index=[i for i in range(len(hang))]
                        random.shuffle(index)
                        neg_num=random.randint(1,len(hang)-1)
                        # 原方法
                        #left=random.uniform(0,all_zero_noise_ratio/len(hang))
                        # left=all_zero_noise_ratio/len(hang)*random.random()
                        # neg_left=-left
                        # 生成正浮动噪声
                        # for j in range(add_num-1):
                        #     noise=left*random.random()
                        #     one[i][index[j]]+=noise
                        #     left-=noise
                        # one[i][index[add_num-1]]+=left
                        # 生成负浮动噪声
                        # for j in range(add_num,len(hang)-1):
                        #     noise=neg_left*random.random()
                        #     one[i][index[j]]+=noise
                        #     neg_left-=noise
                        # one[i][index[len(hang)-1]]+=neg_left
                        # 新方法
                        # 生成负浮动
                        left=0
                        for j in range(neg_num):
                            # noise=all_zero_noise_ratio*random.random()/len(hang)
                            noise = random.uniform(0,all_zero_noise_ratio/len(hang))
                            one[i][index[j]]-=noise
                            left+=noise
                        # 生成正浮动
                        for j in range(neg_num,len(hang)-1):
                            # noise=left*random.random()
                            noise = random.uniform(0,left)
                            one[i][index[j]]+=noise
                            left-=noise
                        one[i][index[-1]]+=left
                # 含1的情况
                else:
                    if index_i==index_j:
                        # left=diagonal_one_noise_ratio*random.random()
                        left = random.uniform(0,diagonal_one_noise_ratio)
                    else:
                        # left=one_noise_ratio*random.random()
                        random_case = random.randint(1, 100000)
                        # 落入极端情况
                        if random_case <= 100000 * false_ratio:
                            left=random.uniform(0.5,one_false_case_max)
                        # 没落入极端情况
                        else:
                            left = random.uniform(one_noise_min,one_noise_max)
                    id=np.argmax(one[i])
                    one[i][id]-=left
                    index=[i for i in range(len(hang)) if i!= id]
                    random.shuffle(index)
                    for j in range(len(index)-1):
                        # noise=left*random.random()
                        noise = random.uniform(0,left)
                        one[i][index[j]]+=noise
                        left-=noise
                    one[i][index[-1]]+=left
            # print('')
    return GT_matrix



if __name__ == '__main__':
    # 可定义参数
    # TODO here
    # 定义存储地址
    saveto_path='./pretrain_datatset/'
    # 定义生成数据类型:[train,test]
    data_type='test'


    if data_type == 'train':
        # 用于控制生成的矩阵数量
        total_matrix_num = 10000
    elif data_type == 'test':
        total_matrix_num = 2000

    # 定义场景中人数范围
    total_person_num_min = 9
    total_person_num_max = 15
    # 用于定义每个视角人数占总人数的最小比例
    each_view_num_ratio_min=0.6
    # 用于定义连续两帧人数进出的最大人数
    out_num_max=2
    in_num_max=2

    # 用来控制全0情况时噪声波动上限
    all_zero_noise_ratio=0.9
    # 用来控制含1情况时噪声波动上下限
    one_noise_min=0.05
    one_noise_max=0.3
    # 用于控制处于大矩阵对角线上矩阵中含1的情况时的噪声浮动上限
    diagonal_one_noise_ratio=0.2
    # 对于含1情况或全0情况产生极端错误干扰输入的概率
    # TODO here
    false_ratio=0.1
    # 全0情况生成极端干扰数据值的最大上限
    zero_false_case_max=0.75
    # 含1情况生成极端干扰数据值的最大上限
    one_false_case_max=0.65

    GT_matrix_list=[]
    noise_matrix_list=[]
    each_num_list=[]

    # for step_i, data in tqdm(enumerate(dataset_train), total=len(dataset_train)):
    print('Generating pretrain dataset!!')
    for i in tqdm(range(total_matrix_num),total=total_matrix_num):
        # print(i+1)
        view_num=random.randint(3, 4)
        GT_matrix, each_num = generate_gt_label(view_num, total_person_num_min, total_person_num_max, each_view_num_ratio_min,out_num_max, in_num_max)
        noise_martix=generate_noise_matrix(GT_matrix.copy(),each_num,all_zero_noise_ratio,one_noise_min,one_noise_max,diagonal_one_noise_ratio,false_ratio,zero_false_case_max,one_false_case_max)
        GT_matrix_list.append(GT_matrix)
        noise_matrix_list.append(noise_martix)
        each_num_list.append(np.array(each_num))
        # print('')
    # print('')
    np.save(saveto_path+data_type+'_GT_matrix_false_ratio_'+str(int(false_ratio*100)).zfill(3)+'.npy', np.array(GT_matrix_list, dtype=object))
    np.save(saveto_path+data_type+'_noise_matrix_false_ratio_'+str(int(false_ratio*100)).zfill(3)+'.npy', np.array(noise_matrix_list, dtype=object))
    np.save(saveto_path+data_type+'_each_len_false_ratio_'+str(int(false_ratio*100)).zfill(3)+'.npy', np.array(each_num_list,dtype=object))
    print('pretrain data save to '+saveto_path+data_type+'_GT_matrix_false_ratio_'+str(int(false_ratio*100)).zfill(3)+'.npy , '+saveto_path+data_type+'_noise_matrix_false_ratio_'+str(int(false_ratio*100)).zfill(3)+'.npy and '
          +saveto_path+data_type+'_each_len_false_ratio_'+str(int(false_ratio*100)).zfill(3)+'.npy!!')