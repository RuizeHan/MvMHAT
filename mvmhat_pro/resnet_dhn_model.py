import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import timm
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.nn.functional import softmax
import torch.nn.functional as f
import config as C
import random
from dhn import Munkrs
from dhn_transformer import STAN_transformer, STAN_FC
from Tools.linear_assignment import sklearn_linear_assignment
from all_loss import gen_S,pairwise_loss,triplewise_loss,weighted_binary_focal_entropy,symmetry_loss,nuclear_norm,get_matrix_sum_row_column,one_pred_row_col_loss,self_similarity,pairwise_loss_consider_zero_case,triplewise_loss_consider_zero_case
os.environ["CUDA_VISIBLE_DEVICES"] = C.TRAIN_GPUS


class resnet_dhn_model(nn.Module):
    def __init__(self, resume,use_softmax='temperature_softmax'):
        super(resnet_dhn_model,self).__init__()
        if C.model_type == 'resnet':
            self.feature_extractor = models.resnet50(pretrained=False).cuda()
            print('use resnet50 feature extractor!')
        elif C.model_type == 'transformer':
            self.feature_extractor = timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=True).cuda()
            print('use VIT-S feature extractor!')
        else:
            print('wrong model type!')
        if C.stan_type == 'rnn':
            self.dhn_model=Munkrs(element_dim=C.element_dim, hidden_dim=C.hidden_dim, target_size=C.target_size, biDirenction=C.bidrectional, minibatch=C.batch_size, is_cuda=C.is_cuda, is_train=True).cuda()
        elif C.stan_type == 'transformer':
            self.dhn_model=STAN_transformer(input_dim=C.input_dim, hidden_dim=C.stan_hidden_dim, output_dim=C.output_dim, n_head=C.n_head, num_layers=C.num_layers).cuda()
        elif C.stan_type == 'fc':
            self.dhn_model=STAN_FC(hidden_dim=C.stan_hidden_dim).cuda()
        else:
            print('wrong STAN model type!')
        print(self.dhn_model)
        self.checkpoint_path = './models/resnet50-19c8e357.pth'
        self.use_softmax=use_softmax
        # 王云dhn位置
        # self.pretrain_dhn_path='./models/pretrain_dhn.pth'
        if C.stan_type == 'rnn':
            self.pretrain_dhn_path = './models/pretrain_rnn_based_stan_error_rate_010.pth'
        elif C.stan_type == 'transformer':
            # TODO here
            # false_ratio=0.1的预训练dhn
            self.pretrain_dhn_path = './models/pretrain_transformer_based_stan_error_rate_010.pth'
        elif C.stan_type == 'fc':
            self.pretrain_dhn_path = None

        # self.pretrain_dhn_path = '/HDDs/hdd3/wff/documents/dhn_tracking_network2_origin_inference/models/pretrain_dhn_false_ratio_0.3/pretrain_dhn_false_ratio_0.3_13_0.04309321042168886.pth'
        if resume == 0:
            if C.model_type=='resnet':
                if C.only_train_dhn:
                    resnet_path='./models/all_model.pth'
                    print('loading pretrained resnet!')
                    pretrained_dict=torch.load(resnet_path)['model']
                    pretrained_dict={k[13:]:v for k,v in pretrained_dict.items()}
                    self.feature_extractor.load_state_dict(pretrained_dict)
                else:
                    self.feature_extractor.load_state_dict(torch.load(self.checkpoint_path))
            if C.use_pretrained_dhn and self.pretrain_dhn_path is not None:
                print('loading pretrained dhn model: '+self.pretrain_dhn_path.split('/')[-1]+'!')
                # 加载原王云师姐预训练dhn的方法
                # self.dhn_model=torch.load(self.pretrain_dhn_path)
                # 加载自己预训练dhn的方法
                pretrained_dhn_dict=torch.load(self.pretrain_dhn_path)['model']
                self.dhn_model.load_state_dict(pretrained_dhn_dict)

    def forward(self,image_crop,each_len,GT_matrix):
        step_loss=0
        if C.only_train_dhn:
            self.feature_extractor.eval()
            feature_map=self.feature_extractor(image_crop)
            norm_feature=f.normalize(feature_map,dim=-1)
            S=torch.mm(norm_feature,norm_feature.transpose(1, 0)).unsqueeze(0)
            # S_norm=(S+1)/2
            # dis_matrix=(1-S_norm).unsqueeze(0)
            # dis_matrix=(-S).unsqueeze(0)
            row = torch.split(S, each_len, dim=1)
            dis_row_col = [list(torch.split(i, each_len, dim=2)) for i in row]
            pred_all=[]
        else:
            feature_map = self.feature_extractor(image_crop)
            norm_feature = f.normalize(feature_map, dim=-1)
            S = torch.mm(norm_feature, norm_feature.transpose(1, 0)).unsqueeze(0)
            # S_norm=(S+1)/2
            # dis_matrix=(1-S_norm).unsqueeze(0)
            # dis_matrix=(-S).unsqueeze(0)
            row = torch.split(S, each_len, dim=1)
            dis_row_col = [list(torch.split(i, each_len, dim=2)) for i in row]
            pred_all = []
        # 用于存储伪标签
        pseduo_label_all=[]

        for i in range(len(dis_row_col)):
            pred_row=[]
            pseduo_label_row = []
            for j in range(len(dis_row_col)):
                # dis_one=dis_row_col[i][j].unsqueeze(0).contiguous()
                dis_one = dis_row_col[i][j].contiguous()

                if self.use_softmax=='temperature_softmax':
                    # 为了防止dhn网络输出值很小且没有差异性，使用temperature softmax
                    scale12 = np.log(C.delta / (1 - C.delta) * dis_one.shape[2]) / C.epsilon
                    # 利用温度自适应softmax使S12中每行元素的和为1，即变为概率
                    dis_one = f.softmax(dis_one * scale12,dim=2)

                elif self.use_softmax=='row_col_softmax':
                    row_softmax = f.softmax(dis_one, dim=2)
                    col_softmax = f.softmax(dis_one, dim=1)
                    dis_one = row_softmax * col_softmax

                elif self.use_softmax == 'row_col_temperature_softmax':
                    scale = np.log(C.delta / (1 - C.delta) * dis_one.shape[2]) / C.epsilon
                    row_softmax = f.softmax(dis_one * scale, dim=2)
                    scale = np.log(C.delta / (1 - C.delta) * dis_one.shape[1]) / C.epsilon
                    col_softmax = f.softmax(dis_one * scale, dim=1)
                    dis_one = row_softmax * col_softmax
                    # print('go here!')

                # TODO 改行可以注释掉
                else:
                    print('not use softmax!')
                pred_row.append(dis_one.squeeze(0))
                if not C.use_gt_label:
                    if C.pseduo_label_type == 'Hungarian':
                        pseduo_matrix=dis_one.squeeze(0).detach().clone().cpu().numpy()
                        pseduo_matrix[pseduo_matrix<0.5]=0
                        assign_ls=sklearn_linear_assignment(-pseduo_matrix)
                        assign_matrix=np.zeros((pseduo_matrix.shape[0],pseduo_matrix.shape[1]))
                        for assign in assign_ls:
                            if pseduo_matrix[assign[0],assign[1]]!=0:
                                assign_matrix[assign[0],assign[1]]=1
                        pseduo_label_row.append(torch.from_numpy(assign_matrix).to(torch.float32).cuda())

                    elif C.pseduo_label_type == 'random_label':
                        pseduo_matrix=dis_one.squeeze(0).detach().clone()
                        one_mask=pseduo_matrix>=0.7
                        zero_mask=pseduo_matrix<=0.3
                        pseduo_matrix[one_mask]=1
                        pseduo_matrix[zero_mask]=0
                        for index_i in range(pseduo_matrix.shape[0]):
                            for index_j in range(pseduo_matrix.shape[1]):
                                if pseduo_matrix[index_i][index_j]<0.7 and pseduo_matrix[index_i][index_j]>0.3:
                                    pseduo_matrix[index_i][index_j]=random.randint(0,1)
                        pseduo_label_row.append(pseduo_matrix)

                    elif C.pseduo_label_type == 'self_label':
                        pseduo_matrix=dis_one.squeeze(0).detach().clone()
                        pseduo_label_row.append(pseduo_matrix)

                    else:
                        print('Wrong pseduo label type!!')

            pred_all.append(pred_row)
            if not C.use_gt_label:
                pseduo_label_all.append(pseduo_label_row)
        if not C.only_train_dhn:
            if 'pairwise' in C.LOSS:
                if C.consider_dialog01:
                    step_loss += C.pairwise_loss_weight * pairwise_loss_consider_zero_case(pred_all)
                    # print('consider zero case pairwise_loss:'+str(C.pairwise_loss_weight * pairwise_loss_consider_zero_case(pred_all)))
                else:
                    step_loss += C.pairwise_loss_weight*pairwise_loss(pred_all)
                    # print('pairwise_loss:'+str(C.pairwise_loss_weight*pairwise_loss(pred_all)))
                # print('go pairwise loss!')

            if 'triplewise' in C.LOSS:
                if C.consider_dialog01:
                    step_loss += C.triplewise_loss_weight * triplewise_loss_consider_zero_case(pred_all)
                    # print('consider zero case triplewise_loss:' + str(C.triplewise_loss_weight * triplewise_loss_consider_zero_case(pred_all)))
                else:
                    step_loss += C.triplewise_loss_weight*triplewise_loss(pred_all)
                    # print('triplewise_loss:' + str(C.triplewise_loss_weight*triplewise_loss(pred_all)))
                # print('go triplewise loss!')

            # if 'sum_loss' in C.LOSS:
            #     step_loss+=C.sum_loss*get_matrix_sum_row_column(pred_all,len(each_len)*len(each_len))
            #     print('sum_loss:' + str(C.sum_loss*get_matrix_sum_row_column(pred_all,len(each_len)*len(each_len))))

        # 将预测结果拼成大的矩阵
        one_pred = torch.cat([torch.cat(row, dim=1) for row in pred_all], dim=0)
        if not C.use_gt_label:
            pseduo_label=torch.cat([torch.cat(row,dim=1) for row in pseduo_label_all],dim=0)
        # 使用dhn网络得到最终的预测指派矩阵
        if C.stan_type == 'rnn':
            self.dhn_model.hidden_row = self.dhn_model.init_hidden(1)
            self.dhn_model.hidden_col = self.dhn_model.init_hidden(1)
        pred = self.dhn_model(one_pred.unsqueeze(0)).squeeze(0)

        # 对预测矩阵中值进行修正，防止求交叉熵时梯度爆炸，对于自监督可以不要该步骤
        final_pred=torch.clamp(pred,min=0.0001,max=0.9999)

        if C.use_gt_label:
            # 只是用来纠错
            cross_entropy=C.cross_entropy * weighted_binary_focal_entropy(final_pred, GT_matrix, use_weights=True)
            #
            step_loss += cross_entropy
            # print('cross_entropy:' + str(C.cross_entropy*weighted_binary_focal_entropy(final_pred,GT_matrix,True)))
        else:
            if C.pseduo_label_type == 'Hungarian' or C.pseduo_label_type == 'random_label':
                cross_entropy=C.pseduo_label_weight*weighted_binary_focal_entropy(final_pred,pseduo_label,use_weights=True)
            elif C.pseduo_label_type == 'self_label':
                cross_entropy=C.pseduo_label_weight*self_similarity(final_pred,pseduo_label)
            else:
                print('Wrong pseduo label type!!')

            step_loss+=cross_entropy
            # print('pseduo_cross_entropy:'+str(cross_entropy))
        if 'symmetry_loss' in C.LOSS:
            # 只是用来纠错
            symmetry=C.symmetry_loss * symmetry_loss(final_pred)
            #
            step_loss += symmetry
            # print('symmetry_loss:' + str(C.symmetry_loss*symmetry_loss(final_pred)))
        if 'nuclear_norm' in C.LOSS:
            # print('go nuclear loss!')
            step_loss += C.nuclear_norm * nuclear_norm(final_pred)
            # print('nuclear_norm:' + str(C.nuclear_norm*nuclear_norm(final_pred)))
        if 'sum_loss' in C.LOSS:
            # 只是用来纠错
            sum=C.sum_loss*one_pred_row_col_loss(final_pred,len(each_len))
            step_loss+=sum
            # print('sum_loss:' + str(C.sum_loss*one_pred_row_col_loss(final_pred,len(each_len))))
        # step_loss+=0.000001
        # print(step_loss)
        if torch.sum(torch.isnan(step_loss)) != 0:
            print('pred nan number:')
            print(torch.sum(torch.isnan(final_pred)))
            min=0.001
            for i in range(5):
                print('matrix value < '+str(min)+':')
                print(torch.sum(final_pred<min))
                min*=0.1
            print('cross_entropy:')
            print(cross_entropy)
            print('symmetry_loss:')
            print(symmetry)
            print('sum loss:')
            print(sum)

        return step_loss

# a=resnet_dhn_model(resume=C.RE_ID)
# print(a)
# for k,v in a.state_dict().items():
#     print(v)