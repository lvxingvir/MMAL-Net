#coding=utf-8
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import time
from config import num_classes, model_name, model_path, lr_milestones, lr_decay_rate, input_size, \
    root, end_epoch, save_interval, init_lr, batch_size, CUDA_VISIBLE_DEVICES, weight_decay, \
    proposalN, set, channels
# from utils.train_model import train
from utils.train_model_2input import train
from utils.read_dataset import read_dataset
from utils.auto_laod_resume import auto_load_resume
from networks.model_onlyappm import MainNet,MainNet_2input

import os

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

def main():

    #加载数据
    trainloader, testloader = read_dataset(input_size, batch_size, root, set)

    #定义模型
    # model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)
    model = MainNet_2input(proposalN=proposalN, num_classes=num_classes, channels=channels)

    #设置训练参数
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()  # for mura
    parameters = model.parameters()

    #加载checkpoint
    save_path = os.path.join(model_path, model_name)
    if os.path.exists(save_path):
        start_epoch, lr = auto_load_resume(model, save_path, status='train')
        assert start_epoch < end_epoch
    else:
        os.makedirs(save_path)
        start_epoch = 0
        lr = init_lr


    # bst_path = r'C:\Users\Xing\Projects\AirGo\MMAL-Net\checkpoint\mura_onlyappm\best_model.pth'
    bst_path = ''
    if os.path.exists(bst_path):
        epoch = auto_load_resume(model, bst_path, status='test')
        # start_epoch = 10 if epoch > 10 else epoch
        lr = 0.0001

    # define optimizers
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)

    model = model.cuda()  # 部署在GPU
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model).cuda()

    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate)

    # 保存config参数信息
    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./config.py', os.path.join(save_path, "{}config.py".format(time_str)))

    # 开始训练
    train(model=model,
          trainloader=trainloader,
          testloader=testloader,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          save_path=save_path,
          start_epoch=start_epoch,
          end_epoch=end_epoch,
          save_interval=save_interval)


if __name__ == '__main__':
    main()