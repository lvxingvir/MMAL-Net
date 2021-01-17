from utils.indices2coordinates import indices2coordinates
from utils.compute_window_nums import compute_window_nums
import numpy as np

CUDA_VISIBLE_DEVICES = '2'  # The current version only supports one GPU training


set = 'Mura'  # Different dataset with different
model_name = ''

batch_size = 4
vis_num = batch_size  # The number of visualized images in tensorboard
eval_trainset = True  # Whether or not evaluate trainset
save_interval = 10
max_checkpoint_num = 100
end_epoch = 100
init_lr = 0.001
lr_milestones = [10,20,30,40,60,80,100]
lr_decay_rate = 0.1
weight_decay = 1e-4
stride = 32
channels = 2048
input_size = 448  #448 as original

# The pth path of pretrained model
pretrain_path = './models/pretrained/resnet50-19c8e357.pth'

if set == 'CUB':
    model_path = './checkpoint/cub'  # pth save path
    root = './datasets/CUB_200_2011'  # dataset path
    num_classes = 200
    # windows info for CUB
    N_list = [2, 3, 2]
    proposalN = sum(N_list)  # proposal window num
    window_side = [128, 192, 256]
    iou_threshs = [0.25, 0.25, 0.25]
    ratios = [[4, 4], [3, 5], [5, 3],
              [6, 6], [5, 7], [7, 5],
              [8, 8], [6, 10], [10, 6], [7, 9], [9, 7], [7, 10], [10, 7]]
else:
    # windows info for CAR and Aircraft
    # N_list = [3, 2, 1]
    # proposalN = sum(N_list)  # proposal window num

    # window_side = [192, 256, 320]
    # iou_threshs = [0.25, 0.25, 0.25]
    # ratios = [[6, 6], [5, 7], [7, 5],
    #           [8, 8], [6, 10], [10, 6], [7, 9], [9, 7],
    #           [10, 10], [9, 11], [11, 9], [8, 12], [12, 8]]

    # window_side = [128, 192, 256]
    # iou_threshs = [0.25, 0.25, 0.25]
    # ratios = [  [4, 4], [3, 5], [5, 3],
    #             [6, 6], [5, 7], [7, 5],
    #             [8, 8], [6, 10], [10, 6], [7, 9], [9, 7]]

    N_list = [4, 2, 1]
    proposalN = sum(N_list)  # proposal window num

    window_side = [64, 128, 192]
    iou_threshs = [0.25, 0.25, 0.25]
    ratios = [[2, 2],[1,3],[3,1],
              [4, 4], [3, 5], [5, 3],
              [6, 6], [5, 7], [7, 5]]
    if set == 'CAR':
        model_path = './checkpoint/car'      # pth save path
        root = './datasets/Stanford_Cars'  # dataset path
        num_classes = 196
    elif set == 'Aircraft':
        model_path = './checkpoint/aircraft'      # pth save path
        root = './datasets/FGVC-aircraft'  # dataset path
        num_classes = 100
    elif set == 'Mura':
        model_path = './checkpoint/mura_onlyappm_4imgs'      # pth save path
        root = r'E:\Xing\Data\MURA-v1.1'  # dataset path
        num_classes = 1


'''indice2coordinates'''
window_nums = compute_window_nums(ratios, stride, input_size)
indices_ndarrays = [np.arange(0,window_num).reshape(-1,1) for window_num in window_nums]
coordinates = [indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # 每个window在image上的坐标
coordinates_cat = np.concatenate(coordinates, 0)
window_milestones = [sum(window_nums[:i+1]) for i in range(len(window_nums))]
if set == 'CUB':
    window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:6]), sum(window_nums[6:])]
else:
    window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:8]), sum(window_nums[8:])]
