# MvMHAT: Multi-view Multi-Human Association and Tracking

**[New 2024]** We have extended this work to form a journal version (submitted to PAMI) from the following aspects.

#### New journal paper (MvMHAT++):

>[**Unveiling the Power of Self-supervision for Multi-view Multi-human Association and Tracking**](https://arxiv.org/abs/2401.17617)

- First, we add a new spatial-temporal assignment matrix learning module, which shares the self-consistency rationale for the appearance feature learning module (in the previous conference paper) to together form a fully self-supervised end-to-end framework. 
- Second, a new pseudo-label generation strategy with dummy nodes used for more general MvMHAT cases is introduced. 
- Third, we include a new dataset MMP-MvMHAT and  significantly extend the experimental comparisons and analyses.

code available: [**here**](https://github.com/RuizeHan/MvMHAT/tree/main/MvMHAT%2B%2B)

#### Previous conference paper (MvMHAT):

> [**Self-supervised Multi-view Multi-Human Association and Tracking** (ACM MM 2021)](https://dl.acm.org/doi/10.1145/3474085.3475177),            
> Yiyang Gan, Ruize Han<sup>&dagger;</sup>, Liqiang Yin, Wei Feng, Song Wang<sup>&dagger;</sup>

- A self-supervised learning framework for MvMHAT.
- A new benchmark for training and testing MvMHAT.

code available: [**here**](https://github.com/realgump/MvMHAT)
<div align= left>
   
## Abstract

<div align= justify>
Multi-view multi-human association and tracking (MvMHAT), is an emerging yet important problem for multi-person scene
video surveillance, aiming to track a group of people over time in each view, as well as to identify the same person across different
views at the same time, which is different from previous MOT and multi-camera MOT tasks only considering the over-time human
tracking. This way, the videos for MvMHAT require more complex annotations while containing more information for self-learning. In
this work, we tackle this problem with an end-to-end neural network in a self-supervised learning manner. Specifically, we propose to
take advantage of the spatial-temporal self-consistency rationale by considering three properties of reflexivity, symmetry, and transitivity.
Besides the reflexivity property that naturally holds, we design the self-supervised learning losses based on the properties of symmetry
and transitivity, for both appearance feature learning and assignment matrix optimization, to associate multiple humans over time and
across views. Furthermore, to promote the research on MvMHAT, we build two new large-scale benchmarks for the network training
and testing of different algorithms. Extensive experiments on the proposed benchmarks verify the effectiveness of our method.


<div align= left>

## Install

~~~
See the 'README' file included in the code folder 'MvMHAT++'.

**[Notes]** The public code of the conference paper (ACM MM 21) can be found at https://github.com/realgump/MvMHAT.
~~~
   
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

## Dataset (MMP-MvMHAT): Reconstructed from the MMPTRACK dataset

### Baidu Drive
~~~
Link: https://pan.baidu.com/s/1D_ex9fXwtIUvLaB3oua6uQ?pwd=mmpt 
Password: mmpt
~~~

### One Drive
~~~
Link: https://1drv.ms/u/s!AnmuGPcsxJTJhgx5CGSUlIo0wKGR?e=LQVv2O
Password: mmpmvmhat
~~~

Note that, the videos in MMP-MvMHAT dataset are not collected in this work, it is better to contact the original authors for usage application.
   
## Evaluation

We add the evaluation code and the raw results of the proposed method in 'Eval_MvMHAT_public.zip'. 

Besides, we define a new metric STMA (Spatial-Temporal Matching Accuracy) for the overall evaluation of the MvMHAT task. Detailed information about this metric can be found in the 'README' file within the 'MvMHAT++' code folder.


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

Contact: [wff@tju.edu.cn](mailto:wff@tju.edu.cn) (Feifan Wang), [han_ruize@tju.edu.cn](mailto:han_ruize@tju.edu.cn) (Ruize Han). Any questions or discussions are welcome! 
