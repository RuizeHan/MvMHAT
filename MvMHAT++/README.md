## Dataset (MvMHAT)

### Baidu Drive
~~~
Part 1 (from Self-collected)
Link：https://pan.baidu.com/s/1gsYTHffmfRq84Hn-8XtzDQ 
Password：2cfh

Part 2 (from Campus)
Link: https://pan.baidu.com/s/1Ts6xnESH-9UV8goiTrSuwQ 
Password: 8sg9

Part 3 (from EPFL) 
Link: https://pan.baidu.com/s/1G84npt61rYDUEPqnaHJUlg 
Password: jjaw 
~~~

### One Drive
~~~
Complete Dataset
Link: https://tjueducn-my.sharepoint.com/:f:/g/personal/han_ruize_tju_edu_cn/EuYKZsvYBvFBvewQPdjvRIoB20iQfMNr_c7_fMDXFRZ7uw?e=19rwJF
Password: MvMHAT
~~~

## Dataset (MMP-MvMHAT)

### Baidu Drive
~~~
Link: https://pan.baidu.com/s/1D_ex9fXwtIUvLaB3oua6uQ?pwd=mmpt 
Password: mmpt
~~~

### One Drive
~~~
Link: 
Password:
~~~

Set line 2 in config.py 'dataset_root_dir' to your root path of download MvMHAT and MMP-MvMHAT datasets.

## Installation
The code is deployed on Ubuntu 20.04, with Anaconda Python 3.10.4 and PyTorch v1.13.1. NVIDIA GPUs are needed for both training and testing. After install Anaconda:

0. [Optional but recommended] create a new conda environment：
~~~
   conda create -n MVMHAT python=3.10.4
~~~
And activate the environment:
~~~
   conda activate MVMHAT
~~~
1. Install pytorch:
~~~
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
~~~
2. Install the requirements:
~~~
   pip install -r requirements.txt
~~~

## Training
0. [optional] generate pre-train datasets for STAN:
~~~
# Modify lines 148 and 150 in /pretrain_dhn/generate_pretrain_datset.py to your save path and dataset type ('train' or 'test'), respectively
# Also modify line 177 in /pretrain_dhn/generate_pretrain_datset.py to your wanted pre-train dataset error rate (default 0.1)
python ./pretrain_dhn/generate_pretrain_dataset.py
~~~
1. [opitional] pre-train STAN network:
~~~
# Modify to the type of STAN network you want to pre-train in config.py line 6 'stan_type' ('rnn' or 'transformer')
# Also modify lines 179 and 184 in pretrain_main.py to your pre-train dataset error rate and path, respectively
python pretrain_main.py
~~~
2. Train your model on MvMHAT or MMP-MvMHAT.

Choose model type of feature extraction network and STAN, respectively:
~~~
Modify lines 4 and 6 in config.py
~~~
If you want to use the pre-trained STAN:
~~~
Set line 59 in config.py 'use_pretrained_dhn' as True
Then, modify line 44 (RNN-based) or 48 (Transformer-based) in resnet_dhn_model.py to your pre-trained STAN path
~~~
We also provide pre-trained RNN-based and transformer-based STAN, you can download from:
~~~
Link: https://pan.baidu.com/s/1kLrwgdBAW2JUjco5wrT98w?pwd=wcjd
Password: wcjd
~~~
Start training:
~~~
# Modify lines 83 and 84 in config.py to your model save name and path, respectively
# train on MvMHAT
python train.py
# train on MMP-MvMHAT
python MMPTrack_train.py
~~~

## Inference
Choose test model type of feature extraction network and STAN, respectively:
~~~
Modify lines 4 and 6 in config.py
~~~
We also provide trained models on MvMHAT and MMP-MvMHAT, respectively. You can download from:
~~~
Link: https://pan.baidu.com/s/1kLrwgdBAW2JUjco5wrT98w?pwd=wcjd
Password: wcjd
~~~
Start inference:
~~~
# Modify lines 104 and 105 in config.py to your test model name and path, respectively
# test on MvMHAT
python inference.py
# test on MMP-MvMHAT
python MMPTrack_inference.py
~~~
## Evaluation
We add the evaluation code and the raw results of the proposed method in 'Eval_MvMHAT_public.zip'.
Besides, we define a new metric STMA (Spatial-Temporal Matching Accuracy) for the overall evaluation of the MvMHAT task:
~~~
# use you want to test dataset ('mvmhat' or 'MMPTrack') in new_metric.py line 265
# Then, Modify lines 271 and 272 of new_metry.py to your predicted and ground-truth result paths, respectively.
python new_metric.py
~~~
