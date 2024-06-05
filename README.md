# MvMHAT: Multi-view Multi-Human Association and Tracking

**[New 2024]** We have extended this work to form a journal version (submitted to PAMI) from the following aspects.

#### New journal paper (MvMHAT++):

>[**Unveiling the Power of Self-supervision for Multi-view Multi-human Association and Tracking**](https://arxiv.org/abs/2401.17617)

- First, we add a new spatial-temporal assignment matrix learning module, which shares the self-consistency rationale for the appearance feature learning module (in the previous conference paper) to together form a fully self-supervised end-to-end framework. 
- Second, a new pseudo-label generation strategy with dummy nodes used for more general MvMHAT cases is introduced. 
- Third, we include a new dataset MMP-MvMHAT and  significantly extend the experimental comparisons and analyses.

#### Previous conference paper (MvMHAT):

> [**Self-supervised Multi-view Multi-Human Association and Tracking** (ACM MM 2021)](https://dl.acm.org/doi/10.1145/3474085.3475177),            
> Yiyang Gan, Ruize Han<sup>&dagger;</sup>, Liqiang Yin, Wei Feng, Song Wang<sup>&dagger;</sup>

- A self-supervised learning framework for MvMHAT.
- A new benchmark for training and testing MvMHAT.

<div align=center><img src="https://github.com/RuizeHan/MvMHAT/blob/main/readme/mvmhat.png" width="500" height="260" alt="example"/><br/>

<div align= left>
   
## Abstract

<div align= justify>
Multi-view multi-human association and tracking (MvMHAT), is a new but important problem for multi-person scene video surveillance, aiming to track a group of people over time in each view, as well as to identify the same person across different views at the same time, which is different from previous MOT and  multi-camera MOT tasks only considering the over-time human tracking. This is a relatively new problem but is very important for multi-person scene video surveillance. This way, the videos for MvMHAT require more complex annotations while containing more information for self-learning. In this work, we tackle this problem with a self-supervised learning aware end-to-end network. 
Specifically, we propose to take advantage of the spatial-temporal self-consistency rationale by considering three properties of reflexivity, symmetry, and transitivity. Besides the reflexivity property that naturally holds, we design the self-supervised learning losses based on the properties of symmetry and transitivity, for both appearance feature learning and assignment matrix optimization, to associate multiple humans over time and across views. Furthermore, to promote the research on MvMHAT, we build two new large-scale benchmarks for the network training and testing of different algorithms. Extensive experiments on the proposed benchmarks verify the effectiveness of our method. 


<div align=center><img src="https://github.com/RuizeHan/MvMHAT/blob/main/readme/2_00-min.png" width="750" height="275" alt="example"/><br/>

<div align= left>
   
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

https://pan.baidu.com/s/1D_ex9fXwtIUvLaB3oua6uQ?pwd=mmpt 

Password: mmpt 

## Evaluation

We add the evaluation code and the raw results of the proposed method in 'Eval_MvMHAT_public.zip'.

## Install (to be completed)

The code was tested on Ubuntu 16.04, with Anaconda Python 3.6 and PyTorch v1.7.1. NVIDIA GPUs are needed for both training and testing. After install Anaconda:

0. [Optional but recommended] create a new conda environment：
~~~
   conda create -n MVMHAT python=3.6
~~~
And activate the environment:
~~~
   conda activate MVMHAT
~~~
1. Install pytorch:
~~~
   conda install pytorch=1.7.1 torchvision -c pytorch
~~~
2. Clone the repository:
~~~
   MVMHAT_ROOT=/path/to/clone/MVMHAT
   git clone https://github.com/realgump/MvMHAT.git $MVMHAT_ROOT
~~~
3. Install the requirements:
~~~
   pip install -r requirements.txt
~~~
4. Download the pretrained model to promote convergence:
~~~
   cd $MVMHAT_ROOT/models
   wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O pretrained.pth
~~~

**[Notes]** The public code of the conference paper (ACM MM 21) can be found at https://github.com/realgump/MvMHAT.

## Citation
If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{gan2021mvmhat,
      title={Self-supervised Multi-view Multi-Human Association and Tracking},
      author={Yiyang Gan, Ruize Han, Liqiang Yin, Wei Feng, Song Wang},
      booktitle={ACM MM},
      year={2021}
    }

     @inproceedings{MvMHAT++,
      title={Unveiling the Power of Self-supervision for Multi-view Multi-human Association and Tracking},
      author={Wei Feng, Feifan Wang, Ruize Han, Zekun Qian, Song Wang},
      booktitle={arXiv},
      year={2023}
    }

## References
Portions of the code are borrowed from [Deep SORT](https://github.com/nwojke/deep_sort), thanks for their great work.

**More information is coming soon ...**

Contact: [han_ruize@tju.edu.cn](mailto:han_ruize@tju.edu.cn) (Ruize Han), [realgump@tju.edu.cn](mailto:realgump@tju.edu.cn) (Yiyang Gan). Any questions or discussions are welcomed! 
