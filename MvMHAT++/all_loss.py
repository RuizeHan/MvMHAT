import torch
import torch.nn.functional as f
import numpy as np
import config as C
import os
import math
os.environ["CUDA_VISIBLE_DEVICES"] = C.TRAIN_GPUS


def gen_S(feature_ls: list):
    norm_feature = [f.normalize(i, dim=-1) for i in feature_ls]
    all_blocks_S = []
    for idx, x in enumerate(norm_feature):
        row_blocks_S = []
        for idy, y in enumerate(norm_feature):
            S = torch.mm(x, y.transpose(0, 1))
            S_norm=(S+1)/2
            row_blocks_S.append(S_norm)
        all_blocks_S.append(row_blocks_S)
    return all_blocks_S

def pairwise_loss_consider_zero_case(all_pred):
    # print('go pair')
    loss_num = 0
    loss_sum = 0
    for i in range(len(all_pred)):
        for j in range(len(all_pred)):
            if i < j:
                loss_num += 1
                pred = all_pred[i][j]
                if pred.shape[0] < pred.shape[1]:
                    S21 = pred
                    S12 = all_pred[j][i]
                    x_index=j
                    y_index=i
                else:
                    S12 = pred
                    S21 = all_pred[j][i]
                    x_index=i
                    y_index=j

                S1221_hat = torch.mm(S12, S21)
                n = S1221_hat.shape[0]
                I = torch.eye(n).cuda()
                pos = S1221_hat * I
                neg = S1221_hat * (1 - I)
                neg2=neg+100*I
                loss = 0
                # 用于获取理想情况下对角线值应该是0或1
                one_mask=(torch.max(all_pred[x_index][y_index],1)[0])>=C.have_target_threshold
                # 每行最大值
                neg_max_1=torch.max(neg,1)[0]
                # 每列最大值
                neg_max_0=torch.max(neg,0)[0]
                # 每行最小值
                neg_min_1=torch.min(neg2,1)[0]
                # 每列最小值
                neg_min_0=torch.min(neg2,0)[0]
                diag=torch.diag(pos)
                for index_i in range(n):
                    if one_mask[index_i]:
                        loss += f.relu(neg_max_1[index_i]+C.MARGIN-diag[index_i])
                        loss += f.relu(neg_max_0[index_i]+C.MARGIN-diag[index_i])
                    else:
                        loss += (f.relu(neg_max_1[index_i]-diag[index_i]-C.MARGIN2)+f.relu(diag[index_i]-neg_min_1[index_i]-C.MARGIN2))/2
                        loss += (f.relu(neg_max_0[index_i]-diag[index_i]-C.MARGIN2)+f.relu(diag[index_i]-neg_min_0[index_i]-C.MARGIN2))/2

                loss /= 2 * n
                loss_sum += loss
    return loss_sum / loss_num

def triplewise_loss_consider_zero_case(all_pred):
    # print('go triple')
    loss_num = 0
    loss_sum = 0
    for i in range(len(all_pred)):
        for j in range(len(all_pred)):
            if i < j:
                for k in range(len(all_pred)):
                    if k != i and k != j:
                        loss_num += 1
                        S12_ = all_pred[i][k]
                        S23_ = all_pred[k][j]
                        S = torch.mm(S12_, S23_)
                        if S.shape[0] < S.shape[1]:
                            S21 = S
                            S12 = all_pred[j][i]
                            x_index=j
                            y_index=i
                            z_index=k
                        else:
                            S12 = S
                            S21 = all_pred[j][i]
                            x_index=i
                            y_index=k
                            z_index=j

                        S1221_hat = torch.mm(S12, S21)
                        n = S1221_hat.shape[0]
                        I = torch.eye(n).cuda()
                        pos = S1221_hat * I
                        neg = S1221_hat * (1 - I)
                        neg2=neg+100*I
                        loss = 0
                        # 用于获取理想情况下对角线应该是0或1
                        mask1=(torch.max(all_pred[x_index][y_index],1)[0])>=C.have_target_threshold
                        mask2=(torch.max(all_pred[x_index][z_index],1)[0])>=C.have_target_threshold
                        one_mask=torch.logical_and(mask1,mask2)
                        neg_max_1 = torch.max(neg, 1)[0]
                        neg_max_0 = torch.max(neg, 0)[0]
                        neg_min_1 = torch.min(neg2, 1)[0]
                        neg_min_0 = torch.min(neg2, 0)[0]
                        diag = torch.diag(pos)
                        for index_i in range(n):
                            if one_mask[index_i]:
                                loss += f.relu(neg_max_1[index_i] + C.MARGIN - diag[index_i])
                                loss += f.relu(neg_max_0[index_i] + C.MARGIN - diag[index_i])
                            else:
                                loss += (f.relu(neg_max_1[index_i] - diag[index_i] - C.MARGIN2)+f.relu(diag[index_i] - neg_min_1[index_i] - C.MARGIN2))/2
                                loss += (f.relu(neg_max_0[index_i] - diag[index_i] - C.MARGIN2)+f.relu(diag[index_i] - neg_min_0[index_i] - C.MARGIN2))/2

                        loss /= 2 * n
                        loss_sum += loss
    return loss_sum / loss_num

def pairwise_loss(all_pred):
    loss_num = 0
    loss_sum = 0
    for i in range(len(all_pred)):
        for j in range(len(all_pred)):
            if i < j:
                loss_num += 1
                pred = all_pred[i][j]
                if pred.shape[0] < pred.shape[1]:
                    S21 = pred
                    S12 = all_pred[j][i]
                else:
                    S12 = pred
                    S21 = all_pred[j][i]

                S1221_hat = torch.mm(S12, S21)
                n = S1221_hat.shape[0]
                I = torch.eye(n).cuda()
                pos = S1221_hat * I
                neg = S1221_hat * (1 - I)
                loss = 0
                loss += torch.sum(f.relu(torch.max(neg, 1)[0] + C.MARGIN - torch.diag(pos)))
                loss += torch.sum(f.relu(torch.max(neg, 0)[0] + C.MARGIN - torch.diag(pos)))
                loss /= 2 * n
                # loss += torch.sum(f.relu(torch.sum(neg, axis=1) + C.MARGIN - torch.diag(pos)))
                # loss += torch.sum(f.relu(torch.sum(neg, axis=0) + C.MARGIN - torch.diag(pos)))
                # loss /= (2 * n * n)
                loss_sum += loss
    return loss_sum / loss_num

def triplewise_loss(all_pred):
    loss_num = 0
    loss_sum = 0
    for i in range(len(all_pred)):
        for j in range(len(all_pred)):
            if i < j:
                for k in range(len(all_pred)):
                    if k != i and k != j:
                        loss_num += 1
                        S12_ = all_pred[i][k]
                        S23_ = all_pred[k][j]
                        S = torch.mm(S12_, S23_)
                        if S.shape[0] < S.shape[1]:
                            S21 = S
                            S12 = all_pred[j][i]
                        else:
                            S12 = S
                            S21 = all_pred[j][i]

                        S1221_hat = torch.mm(S12, S21)
                        n = S1221_hat.shape[0]
                        I = torch.eye(n).cuda()
                        pos = S1221_hat * I
                        neg = S1221_hat * (1 - I)
                        loss = 0
                        loss += torch.sum(f.relu(torch.max(neg, 1)[0] + C.MARGIN - torch.diag(pos)))
                        loss += torch.sum(f.relu(torch.max(neg, 0)[0] + C.MARGIN - torch.diag(pos)))
                        loss /= 2 * n
                        # loss += torch.sum(f.relu(torch.sum(neg, axis=1) + C.MARGIN - torch.diag(pos)))
                        # loss += torch.sum(f.relu(torch.sum(neg, axis=0) + C.MARGIN - torch.diag(pos)))
                        # loss /= (2 * n * n)
                        loss_sum += loss
    return loss_sum / loss_num

# weighted classification loss #
def weighted_binary_focal_entropy(output, target, use_weights=True, gamma=2):
    if use_weights:
        total_number=target.shape[0]*target.shape[1]
        num_positive=target.sum().item()
        # 用于计算带权交叉熵中负样本前面的权重系数
        weight2negative = float(num_positive) / total_number

        # case all zeros, then weight2negative = 1.0
        if weight2negative <= 0.0:
            weight2negative = 1.0
        # case all ones, then weight2negative = 0.0
        if num_positive == total_number:
            weight2negative = 0.0

        weights = torch.tensor([weight2negative, 1.0 - weight2negative], dtype=torch.float32).unsqueeze(0).contiguous()
        weights=weights.cuda()

        # weights.shape=(1,2)
        # weight[:,1] is for positive class, label = 1
        # weight[:,0] is for negative class, label = 0
        # target是label,output是DHN预测结果
        loss = weights[:, 1].item() *(1 - output) ** gamma * (target * torch.log(output)) + \
             weights[:, 0].item() *output ** gamma * ((1 - target) * torch.log(1 - output))
    else:
        loss = target[1, :] * torch.log(output[1, :]) + target[0, :] * torch.log(output[0, :])

    return torch.neg(torch.mean(loss))

def symmetry_loss(output):
    output_T=torch.transpose(output,1, 0)
    temp=output-output_T
    try:
        L=torch.norm(temp,p=2,dim=None,keepdim=False,out=None,dtype=None).cuda()
    except:
        L=0.0
    return L

def nuclear_norm(output):
    try:
        L=torch.norm(output, p='nuc', dim=None, keepdim=False, out=None, dtype=None).cuda()
    except:
        L=0.0
    return L

# 暂时不用
def get_matrix_sum_row_column(output,matrix_num):
    loss=0
    # output=torch.squeeze(output)
    # n=output.shape[0]
    for i in range(len(output)):
        for j in range(len(output)):
            one=output[i][j]
            temp_loss=0
            one_shape=one.shape[0]*one.shape[1]
            matrix_row=one.sum(axis=1)
            matrix_column=one.sum(axis=0)
            # for k in range(len(matrix_row)):
                # if matrix_row[k]>1.0:
                #     temp_loss+=0.5*(matrix_row[k]-1.0)*(matrix_row[k]-1.0)

                # elif i==j and matrix_row[k]<1.0:
                #     temp_loss += 0.5 * (matrix_row[k] - 1.0) * (matrix_row[k] - 1.0)
                # elif matrix_row[k]<0.0:
                #     temp_loss += 0.5 * matrix_row[k]*matrix_row[k]

            temp_loss+=torch.sum(0.5*(matrix_row-1.0)*(matrix_row-1.0))
            # for k in range(len(matrix_column)):
                # if matrix_column[k]>1.0:
                #     temp_loss += 0.5 * (matrix_column[k] - 1.0) * (matrix_column[k] - 1.0)

                # elif i==j and matrix_column[k]<1.0:
                #     temp_loss += 0.5 * (matrix_column[k] - 1.0) * (matrix_column[k] - 1.0)
                # elif matrix_column[k]<0.0:
                #     temp_loss += 0.5 * matrix_column[k] * matrix_column[k]

            temp_loss+=torch.sum(0.5*(matrix_column-1.0)*(matrix_column-1.0))
            loss+=(temp_loss/one_shape)
    # return loss/(0.5*n*n)
    return loss/matrix_num

def sum_loss(pred,max_num,dim):
    dim_sum=torch.sum(pred,dim=dim)
    # 通过下面计算loss的方法有如下结果：
    # (1)sum>max_num时,loss=sum-max_num;
    # (2)1<=sum<=max_num时,loss=0;
    # (3)sum<1时,loss=1-sum.
    sum=torch.abs(dim_sum-(max_num+1)/2)-(max_num-1)/2
    loss=torch.max(sum,torch.zeros_like(sum))
    return torch.mean(loss)

def one_pred_row_col_loss(pred,max_num):
    row_col_loss=sum_loss(pred,max_num,0)+sum_loss(pred,max_num,1)
    return row_col_loss

def self_similarity(pred,pseduo_label):
    temp=pred-pseduo_label
    len=pred.shape[0]
    return (torch.norm(temp,p=2,dim=None,keepdim=False,out=None,dtype=None)/len).cuda()