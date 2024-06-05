import os
import config as C
os.environ["CUDA_VISIBLE_DEVICES"] = C.TRAIN_GPUS
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from mmptrack_loader import MMPTrack_Loader
from torch.cuda.amp import autocast as autocast
from resnet_dhn_model import resnet_dhn_model
from dhn import Munkrs

def train(epoch):
    resnet_dhn_model.train()
    epoch_loss = 0
    print('epoch: '+str(epoch))
    for step_i, data in tqdm(enumerate(dataset_train), total=len(dataset_train)):
        optimizer.zero_grad()
        # 只是为了纠错
        # torch.autograd.set_detect_anomaly(True)
        image_ls = []
        label_ls=[]
        each_len=[]
        # with autocast():
        for view_i in range(len(data)):
            for frame_i in range(C.FRAMES):
                img, box, lbl, scn = data[view_i][frame_i]
                image_ls.extend(img.squeeze(0))
                if C.use_gt_label:
                    label_ls.extend(lbl)
                each_len.append(len(lbl))
        # image_ls共包含view_num*C.frame个帧中所有人对应的剪裁图片
        image_ls=torch.stack(image_ls,0).cuda()
        if C.use_gt_label:
            # 生成gt标签,GT_matrix即为dis_matrix对应的正确指派矩阵
            labels = [int(i[0]) for i in label_ls]
            labels = torch.tensor(labels).cuda()
            GT_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            # 通过网络生成每个图片间的预测指派矩阵
            step_loss=resnet_dhn_model(image_ls,each_len,GT_matrix)
        else:
            step_loss = resnet_dhn_model(image_ls, each_len, None)
        epoch_loss += step_loss.item()
        # print(step_loss.item())
        if epoch >= 0:
            # 只是为了纠错
            # with torch.autograd.detect_anomaly():
            step_loss.backward()
            optimizer.step()
        # if(step_i==200):
        #     print(one_pred)
        #     print(pred)
    return epoch_loss / (step_i + 1)

if __name__ == '__main__':
    train_times = ['63am', '64am']
    datasets_viewnum_dict = {'cafe_shop_0': 4, 'industry_safety_0': 4, 'lobby_0': 4, 'office_0': 5}
    datasets = []
    for time in train_times:
        for dataset_name in datasets_viewnum_dict.keys():
            datasets.append(MMPTrack_Loader(views=datasets_viewnum_dict[dataset_name], frames=C.FRAMES, mode='train', dataset=dataset_name,time=time))
    datasets = ConcatDataset(datasets)
    dataset_train = DataLoader(datasets, num_workers=0, pin_memory=True, shuffle=C.LOADER_SHUFFLE)

    resnet_dhn_model=resnet_dhn_model(resume=C.RE_ID,use_softmax='temperature_softmax')

    # optimizer = torch.optim.Adam([{'params': resnet_model.parameters()}, {'params': dhn_model.parameters()}],
    #                              lr=C.LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    if C.only_train_dhn:
        optimizer = torch.optim.Adam(resnet_dhn_model.dhn_model.parameters(),lr=C.LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        optimizer=torch.optim.Adam(resnet_dhn_model.parameters(),lr=C.LEARNING_RATE,betas=(0.9,0.999),eps=1e-08,weight_decay=0)

    if C.RE_ID:
        print('train continue! Saved model name: ' + C.RE_ID + ' , ' +
              'Output model name: '+C.EX_ID+' , '+
              'loss: ' + " ".join(C.LOSS) + ' , ' +
              'lr: ' + str(C.LEARNING_RATE) + ' , ' +
              'network: ' + C.model_type + ' dhn')

        checkpoint_path = C.TRAIN_RESUME
        ckp_origin=torch.load(checkpoint_path)
        resnet_dhn_model.load_state_dict(ckp_origin['model'])
        max_loss=ckp_origin['loss']
        # 新增学习速率下降
        last_epoch = max_loss
        start_epoch=ckp_origin['epoch']+1
        optimizer.load_state_dict(ckp_origin['optimizer'])
        # for params_group in optimizer.param_groups:
        #     params_group['lr']=C.LEARNING_RATE
        for params_group in optimizer.param_groups:
            print(params_group['lr'])
    else:
        print('train from start! Output model name: ' + C.EX_ID + ' , ' +
              'loss: ' + " ".join(C.LOSS) + ' , ' +
              'lr: ' + str(C.LEARNING_RATE) + ' , ' +
              'network: ' + C.model_type + ' dhn')

        for params_group in optimizer.param_groups:
            print(params_group['lr'])

        max_loss=1e8
        # 新增学习速率下降
        last_epoch = 1e8
        start_epoch=0
        # TODO
        # 训练前先存模型
        print('save model before train!')
        torch.save(
            {
                'epoch': -1,
                'loss': -1,
                'optimizer': optimizer.state_dict(),
                'model': resnet_dhn_model.state_dict(),
            },
            C.MODEL_SAVE_NAME + '_-1_-1' + '.pth'
        )

    now=0
    for epoch_i in range(start_epoch,C.MAX_EPOCH):

        # TODO 查看模型的参数是否相同
        # before_resnet=[v.clone().detach() for _,v in resnet_dhn_model.resnet_model.state_dict().items()]
        # before_dhn=[v.clone().detach() for _,v in resnet_dhn_model.dhn_model.state_dict().items()]


        epoch_loss = train(epoch_i)


        # TODO 查看模型参数是否相同
        # after_resnet=[v.clone().detach() for _,v in resnet_dhn_model.resnet_model.state_dict().items()]
        # after_dhn=[v.clone().detach() for _,v in resnet_dhn_model.dhn_model.state_dict().items()]

        # res_num=0
        # for i in range(len(before_resnet)):
        #     if (torch.sum(before_resnet[i]!=after_resnet[i])!=0):
        #         res_num+=1
        # if res_num==0:
        #     print('resnet all same!')
        # else:
        #     print('resnet change!')
        # dhn_num = 0
        # for i in range(len(before_dhn)):
        #     if (torch.sum(before_dhn[i] != after_dhn[i]) != 0):
        #         dhn_num += 1
        # if dhn_num == 0:
        #     print('dhn all same!')
        # else:
        #     print('dhn change!')


        print("epoch loss:"+"%.6f"%epoch_loss)

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
                    'model': resnet_dhn_model.state_dict(),
                },
                C.MODEL_SAVE_NAME+'_'+str(epoch_i)+'_'+str(epoch_loss)+'.pth'
            )




