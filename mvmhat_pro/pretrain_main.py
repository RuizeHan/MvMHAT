# 注：每次在服务器端运行时需要从文件夹中拷贝到外面后运行！！！
import os
import config as C
os.environ["CUDA_VISIBLE_DEVICES"] = C.TRAIN_GPUS
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from dhn import Munkrs
from Tools.linear_assignment import sklearn_linear_assignment
from dhn_transformer import STAN_transformer
from all_loss import weighted_binary_focal_entropy,symmetry_loss,nuclear_norm,one_pred_row_col_loss


class pretrain_Loader(Dataset):
    def __init__(self, dir_path, GT_matrix_name, noise_matrix_name, len_name, train_data_len):

        self.GT_matrix_path=dir_path+GT_matrix_name+'.npy'
        self.noise_matrix_path=dir_path+noise_matrix_name+'.npy'
        self.len_path=dir_path+len_name+'.npy'
        self.GT_matrix=np.load(self.GT_matrix_path, allow_pickle=True)[:train_data_len]
        self.noise_matrix=np.load(self.noise_matrix_path,allow_pickle=True)[:train_data_len]
        self.len_list=np.load(self.len_path,allow_pickle=True)[:train_data_len]

    def __len__(self):
        return len(self.GT_matrix)

    def __getitem__(self, item):
        GT_one=torch.from_numpy(self.GT_matrix[item])
        noise_one=torch.from_numpy(self.noise_matrix[item])
        len_one=self.len_list[item]

        return GT_one,noise_one,len_one

def test(dhn_model,dataset_test):
    dhn_model.eval()
    total_loss=0
    origin_gt_similarity=[]
    origin_self_similarity=[]
    Hungarian_gt_similarity=[]
    Hungarian_self_similarity = []
    print('test!......')
    for step_i, data in tqdm(enumerate(dataset_test),total=len(dataset_test)):
        with torch.no_grad():
            noise_matrix=data[1].cuda()
            gt_matrix=data[0].squeeze(0).cuda()
            each_len=data[2].squeeze(0).cpu().numpy().tolist()
            if C.stan_type == 'rnn':
                dhn_model.hidden_row=dhn_model.init_hidden(1)
                dhn_model.hidden_col=dhn_model.init_hidden(1)
            pred=dhn_model(noise_matrix).squeeze(0)
            # 对预测矩阵中值进行修正，防止求交叉熵时梯度爆炸
            final_pred = torch.clamp(pred, min=0.0001, max=0.9999)
            # 交叉熵损失
            cross_entropy = C.cross_entropy * weighted_binary_focal_entropy(final_pred, gt_matrix, use_weights=True)
            total_loss += cross_entropy.item()
            # print('cross_entropy:' + str(cross_entropy))
            # 对称损失
            symmetry = C.symmetry_loss * symmetry_loss(final_pred)
            total_loss += symmetry.item()
            # print('symmetry_loss:' + str(symmetry))
            # 核范数损失
            nuc_loss = C.nuclear_norm * nuclear_norm(final_pred)
            total_loss += nuc_loss.item()
            # print('nuclear_norm:' + str(nuc_loss))
            # 行列和约束损失
            sum = C.sum_loss * one_pred_row_col_loss(final_pred, len(each_len))
            total_loss += sum.item()
            # print('sum_loss:' + str(sum))

            # 使用预测矩阵直接计算相似度
            origin_gt_similarity.append(torch.sum(final_pred*gt_matrix)/torch.sum(gt_matrix))
            origin_self_similarity.append(torch.sum(final_pred*gt_matrix)/torch.sum(final_pred))
            # 拆分小矩阵，每个小矩阵使用匈牙利算法生成指派矩阵计算相似度
            row = torch.split(final_pred, each_len, dim=0)
            split_pred = [list(torch.split(i, each_len, dim=1)) for i in row]
            final_result_all=[]
            for index_i in range(len(each_len)):
                final_result_row=[]
                for index_j in range(len(each_len)):
                    one=split_pred[index_i][index_j].cpu().numpy()
                    # 将概率小于0.5的认定为两个检测目标肯定不是同一个人
                    one[one < 0.5] = 0
                    # assign_ls为使用匈牙利算法后的匹配结果，-S12代表相似度越大，代价（距离）越小
                    assign_ls = sklearn_linear_assignment(-one)
                    # X_12即为论文中的P矩阵，初始矩阵中数值全为0
                    X_12 =torch.zeros((one.shape[0], one.shape[1]))
                    # 对于每个匹配对，只有S12中对应值不为0，才把X_12对应位置赋为1
                    for assign in assign_ls:
                        if one[assign[0], assign[1]] != 0:
                            X_12[assign[0], assign[1]] = 1
                    # row_blocks_X是dict，view_y->X_12构成其中一个映射
                    final_result_row.append(X_12)
                # all_blocks_X是dict，view_x->row_blocks_X构成其中一个映射
                final_result_all.append(final_result_row)
            # final_result即为各小矩阵经过匈牙利算法预测后的最终矩阵
            final_result=torch.cat([torch.cat(row, dim=1) for row in final_result_all], dim=0).cuda()
            Hungarian_gt_similarity.append(torch.sum(final_result*gt_matrix)/torch.sum(gt_matrix))
            Hungarian_self_similarity.append(torch.sum(final_result*gt_matrix)/torch.sum(final_result))
            # print('')
    avg_loss=total_loss/(step_i+1)
    print('avg loss: '+str(avg_loss))
    origin_gt_similarity_min=min(origin_gt_similarity).item()
    print('origin_gt_similarity min: '+str(origin_gt_similarity_min))
    origin_gt_similarity_avg = get_avg(origin_gt_similarity).item()
    print('origin_gt_similarity avg: '+str(origin_gt_similarity_avg))
    origin_self_similarity_min=min(origin_self_similarity).item()
    print('origin_self_similarity min: '+str(origin_self_similarity_min))
    origin_self_similarity_avg = get_avg(origin_self_similarity).item()
    print('origin_self_similarity avg: '+str(origin_self_similarity_avg))
    Hungarian_gt_similarity_min=min(Hungarian_gt_similarity).item()
    print('Hungarian_gt_similarity min: '+str(Hungarian_gt_similarity_min))
    Hungarian_gt_similarity_avg = get_avg(Hungarian_gt_similarity).item()
    print('Hungarian_gt_similarity avg: '+str(Hungarian_gt_similarity_avg))
    Hungarian_self_similarity_min=min(Hungarian_self_similarity).item()
    print('Hungarian_self_similarity min: '+str(Hungarian_self_similarity_min))
    Hungarian_self_similarity_avg = get_avg(Hungarian_self_similarity).item()
    print('Hungarian_self_similarity avg: '+str(Hungarian_self_similarity_avg))
    # print('')
    return

def get_avg(list):
    sum=0
    for item in list:
        sum+=item
    return sum/len(list)


def train(epoch,dhn_model,dataset_train,optimizer):
    dhn_model.train()
    epoch_loss = 0
    print('epoch: '+str(epoch))
    for step_i, data in tqdm(enumerate(dataset_train), total=len(dataset_train)):
        step_loss=0
        optimizer.zero_grad()
        # with autocast():

        noise_matrix = data[1].cuda()
        gt_matrix=data[0].squeeze(0).cuda()
        len_one=len(data[2].squeeze(0))

        # 使用dhn网络得到最终的预测指派矩阵
        if C.stan_type == 'rnn':
            dhn_model.hidden_row = dhn_model.init_hidden(1)
            dhn_model.hidden_col = dhn_model.init_hidden(1)
        pred = dhn_model(noise_matrix).squeeze(0)

        # 对预测矩阵中值进行修正，防止求交叉熵时梯度爆炸
        final_pred = torch.clamp(pred, min=0.0001, max=0.9999)
        # 交叉熵损失
        cross_entropy = C.cross_entropy * weighted_binary_focal_entropy(final_pred, gt_matrix, use_weights=True)
        step_loss += cross_entropy
        # print('cross_entropy:' + str(cross_entropy))
        # 对称损失
        symmetry = C.symmetry_loss * symmetry_loss(final_pred)
        step_loss += symmetry
        # print('symmetry_loss:' + str(symmetry))
        # 核范数损失
        nuc_loss = C.nuclear_norm * nuclear_norm(final_pred)
        step_loss += nuc_loss
        # print('nuclear_norm:' + str(nuc_loss))
        # 行列和约束损失
        sum = C.sum_loss * one_pred_row_col_loss(final_pred, len_one)
        step_loss += sum
        # print('sum_loss:' + str(sum))

        epoch_loss += step_loss.item()
        # print(step_loss.item())
        if epoch >= 0:
            step_loss.backward()
            optimizer.step()
    return epoch_loss / (step_i + 1)



if __name__ == '__main__':
    # TODO here
    false_ratio=0.1


    # 参数定义
    RE_PATH=0
    dir_path='./pretrain_datatset/'
    # RE_PATH = '/HDDs/hdd3/wff/documents/dhn_tracking_network2/models/pretrain_dhn_first_try/pretrain_dhn_8_0.027538955308683218.pth'
    # EX_dir='/HDDs/hdd3/wff/documents/dhn_tracking_network2/models/pretrain_dhn_false_ratio_0.3/'
    EX_dir='./models/pretrain_transformer_based_dhn_nhead_4_layer_2_with_2d_postion_embed_false_ratio_'+str(int(false_ratio*100)).zfill(3)+'/'
    EX_PATH=EX_dir+'pretrain_transformer_based_dhn_nhead_4_layer_2_with_2d_postion_embed_false_ratio_'+str(int(false_ratio*100)).zfill(3)

    if not os.path.exists(EX_dir):
        os.mkdir(EX_dir)
    # 训练数据集
    datasets=pretrain_Loader(dir_path=dir_path,GT_matrix_name=('train_GT_matrix_false_ratio_'+str(int(false_ratio*100)).zfill(3)),noise_matrix_name=('train_noise_matrix_false_ratio_'+str(int(false_ratio*100)).zfill(3)),len_name=('train_each_len_false_ratio_'+str(int(false_ratio*100)).zfill(3)),train_data_len=10000)
    dataset_train = DataLoader(datasets, num_workers=0, pin_memory=True, shuffle=C.LOADER_SHUFFLE)
    # 测试数据集
    datasets2=pretrain_Loader(dir_path=dir_path,GT_matrix_name=('test_GT_matrix_false_ratio_'+str(int(false_ratio*100)).zfill(3)),noise_matrix_name=('test_noise_matrix_false_ratio_'+str(int(false_ratio*100)).zfill(3)),len_name=('test_each_len_false_ratio_'+str(int(false_ratio*100)).zfill(3)),train_data_len=2000)
    dataset_test=DataLoader(datasets2,num_workers=0,pin_memory=True,shuffle=False)
    if C.stan_type == 'rnn':
        dhn_model=Munkrs(element_dim=C.element_dim, hidden_dim=C.hidden_dim, target_size=C.target_size, biDirenction=C.bidrectional, minibatch=C.batch_size, is_cuda=C.is_cuda, is_train=True).cuda()
    elif C.stan_type == 'transformer':
        dhn_model = STAN_transformer(input_dim=C.input_dim, hidden_dim=C.stan_hidden_dim, output_dim=C.output_dim, n_head=C.n_head, num_layers=C.num_layers).cuda()
    print(dhn_model)
    optimizer = torch.optim.Adam(dhn_model.parameters(), lr=C.LEARNING_RATE, betas=(0.9, 0.999),eps=1e-08, weight_decay=0)

    if RE_PATH:
        print('train continue! Saved model name: ' + RE_PATH.split('/')[-1] + ' , ' +
              'Output model name: ' + EX_PATH.split('/')[-1] + ' , ' +
              'lr: ' + str(C.LEARNING_RATE) + ' , ' +
              'network: dhn')

        checkpoint_path = RE_PATH
        ckp_origin = torch.load(checkpoint_path)
        dhn_model.load_state_dict(ckp_origin['model'])
        max_loss = ckp_origin['loss']
        # 新增学习速率下降
        last_epoch = max_loss
        start_epoch = ckp_origin['epoch'] + 1
        optimizer.load_state_dict(ckp_origin['optimizer'])
        # for params_group in optimizer.param_groups:
        #     params_group['lr']=C.LEARNING_RATE
        for params_group in optimizer.param_groups:
            print(params_group['lr'])
    else:
        print('train from start! Output model name: ' + EX_PATH.split('/')[-1] + ' , ' +
              'lr: ' + str(C.LEARNING_RATE) + ' , ' +
              'network: dhn')

        for params_group in optimizer.param_groups:
            print(params_group['lr'])

        max_loss = 1e8
        # 新增学习速率下降
        last_epoch = 1e8
        start_epoch = 0


        # TODO
        # 训练前先存模型
        print('save model before train!')
        torch.save(
            {
                'epoch': -1,
                'loss': -1,
                'optimizer': optimizer.state_dict(),
                'model': dhn_model.state_dict(),
            },
            EX_PATH + '_-1_-1.pth'
        )




    now = 0
    for epoch_i in range(start_epoch, C.PRETRAIN_MAX_EPOCH):

        # TODO 查看模型的参数是否相同
        # before_dhn=[v.clone().detach() for _,v in dhn_model.state_dict().items()]

        epoch_loss = train(epoch_i,dhn_model,dataset_train,optimizer)
        # test(dhn_model,dataset_test)

        # TODO 查看模型参数是否相同
        # after_dhn=[v.clone().detach() for _,v in dhn_model.state_dict().items()]

        # dhn_num = 0
        # for i in range(len(before_dhn)):
        #     if (torch.sum(before_dhn[i] != after_dhn[i]) != 0):
        #         dhn_num += 1
        # if dhn_num == 0:
        #     print('dhn all same!')
        # else:
        #     print('dhn change!')

        print("epoch loss:" + "%.6f" % epoch_loss)
        #
        # 新增调节学习速率下降
        if epoch_loss > last_epoch:
            now += 1
            if now >= C.patience:
                for params_group in optimizer.param_groups:
                    params_group['lr'] = C.REDUCTION_LEARNING_RATE
                now = 0
                last_epoch = epoch_loss
        else:
            last_epoch = epoch_loss
            now = 0

        if epoch_loss < max_loss:
            max_loss = epoch_loss
            print('save model')
            torch.save(
                {
                    'epoch': epoch_i,
                    'loss': epoch_loss,
                    'optimizer': optimizer.state_dict(),
                    'model': dhn_model.state_dict(),
                },
                EX_PATH + '_' + str(epoch_i) + '_' + str(epoch_loss) + '.pth'
            )
        # 开始测试
        test(dhn_model, dataset_test)