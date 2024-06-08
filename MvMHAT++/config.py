import os
dataset_root_dir = '/data1/wff/dataset/'
# TODO feature extractor model_type:'resnet','transformer'
model_type = 'transformer'
# TODO stan_type:'rnn','transformer','fc'
stan_type = 'transformer'

# dataset
TRAIN_DATASET = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
# TRAIN_DATASET = ['12']
TEST_DATASET = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
FRAMES = 2
VIEWS = 4


# softmax setting
use_softmax='temperature_softmax'
delta=0.5
epsilon = 0.1
# 用来控制映射中x是否有对应y的阈值（默认0.5）
have_target_threshold=0.5
# 用来控制对角线元素该是1时的损失margin
MARGIN = 0.5
# 用来控制对角线元素该是0时的损失margin
MARGIN2=0.4

# loss权重设置
pairwise_loss_weight=1.0
triplewise_loss_weight=1.0
cross_entropy=10.0
pseduo_label_weight=5.0
symmetry_loss=0.002
nuclear_norm=0.0001
# nuclear_norm=1.0
sum_loss=0.004


# TODO dhn network setting
element_dim=1
hidden_dim=256
target_size=1
bidrectional=True
batch_size=1
is_cuda=True

# TODO stan_transformer network setting
input_dim=1
stan_hidden_dim=128
output_dim=1
#  TODO 默认为4
n_head=4
# TODO 默认为1，总层数默认为2
num_layers=2


# training
# 很重要！！！该变量决定最后四个loss是只训练dhn还是dhn和feature extractor都训练
only_train_dhn=0
use_pretrained_dhn=1
consider_dialog01=1
# 伪标签的类别：(1)匈牙利算法'Hungarian' (2)根据阈值随机生成'random_label' (3)自己本身 'self_label'
pseduo_label_type='Hungarian'
use_gt_label=0
# patience用来控制调整学习速率，当有patince个epoch的loss都比之前时刻的epoch的loss大时则降低学习速率
patience=3

TRAIN_GPUS = '0'
# LOSS = ['pairwise', 'triplewise', 'pseudo_label', 'symmetry_loss', 'sum_loss']
LOSS = ['pairwise', 'triplewise', 'pseudo_label', 'symmetry_loss', 'nuclear_norm', 'sum_loss']
# 初始学习速率
LEARNING_RATE = 1e-5
# 新增降低后的学习速率
REDUCTION_LEARNING_RATE = 1e-5

# feature_extractor+dhn最大训练周期
MAX_EPOCH = 15
# dhn最大预训练周期
PRETRAIN_MAX_EPOCH = 15

RE_ID = 0
# RE_ID = 'mypretrain_dhn0.3_unsuperviesd_dialog01_Hungarian_new_19_0.10984037089137488'
# EX_ID = 'trainwithout_retail_pretrain_dhn_010_unsuperviesd_dialog01_Hungarian_withoutnucloss'
EX_ID = 'MvMHAT_vit_pretrain_error_ratio_010_transformer_based_stan_unsupervised_dialog01_Hungarian'
save_folder='./models/MvMHAT_vit_pretrain_error_ratio_010_transformer_based_stan/'
if RE_ID:
    TRAIN_RESUME = save_folder + RE_ID + '.pth'
    # EX_ID=RE_ID

if not os.path.exists(save_folder):
    os.mkdir(save_folder)
MODEL_SAVE_NAME = save_folder + EX_ID

# 表示训练集是否打乱顺序
DATASET_SHUFFLE = 0
LOADER_SHUFFLE = 1

# TODO inference
start_epoch=-1
end_epoch=30

# TODO inference mode:"without_association", "without_tracking", "association_and_tracking". default:"association_and_tracking"
inference_mode = "association_and_tracking"

INF_ID = 'MvMHAT_vit_pretrain_error_ratio_010_transformer_based_stan'
test_folder='./models/'
DISPLAY = 0
INFTY_COST = 1e+5
# 默认10
RENEW_TIME = 10

# 推理阶段新增加
inference_time_consider_filter=0
inference_time_simliarity_threshold=-1
# 默认0.5
inference_global_threshold=0.5
# 类别包括'intersection'（交集）,'union_matching_cascade'（并集得到，并以原级联匹配结果为准）,'union_assign_matrix'（并集得到，并以指派矩阵结果为准）,'no_use'（不使用指派矩阵只使用级联匹配结果）,'only_assign_matrix'（只使用指派矩阵）
inference_time_assign_type='no_use'

# inference_output_path="output/SVT_FIND_mypretrain_dhn0.1_unsuperviesd_dialog01_Hungarian_14_0.12270036315416981_union_assign_matrix/"

# 新增SVT参数，只是用来测试找出最好参数
SVT_threshold=0.0
SVT_iter=0
