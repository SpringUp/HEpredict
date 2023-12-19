from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import ImageFile

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.common import logger
from utils.custom_dset import CustomDset
# from utils.analytics import draw_roc, draw_roc_for_multiclass

import train_test_splitter 
from train_debug import train_model
from test import test

from Net import Net, Cnn_With_Clinical_Net
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '12355'

plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        #transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class Config(object):
    ngpu = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    local_rank = 0


def generative_model(config, model, k, cnv="", split_folder='halves'):
    image_datasets = {x: CustomDset(os.getcwd()+f'/{split_folder}/{x}_{k}.csv',
                        data_transforms[x]) for x in ['train']}
    sampler=torch.utils.data.distributed.DistributedSampler(
        image_datasets["train"],
        num_replicas=config.ngpu,
        rank=config.local_rank,
        )
    dataloaders = torch.utils.data.DataLoader(
        image_datasets["train"],
        batch_size=256,
        num_workers=8,
        pin_memory=True,
        sampler=sampler,
        drop_last=True, # !!!!!!!!!!!!!!!!!!!!!!!!Modify To True! 
    )
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    class_names = image_datasets['train'].classes

    logger.info(f'model {model} / 第 {k+1} 折')

    available_policies = {"resnet18": models.resnet18, "vgg16": models.vgg16, "vgg19": models.vgg19, 
            "alexnet": models.alexnet, "inception": models.inception_v3}
    
    # model_ft = available_policies[model](pretrained=True)
    model_ft = models.__dict__[model](pretrained=True)
    
    if cnv:
        model_ft = Cnn_With_Clinical_Net(model_ft)
    else:
        model_ft = Net(model_ft)
        
    model_ft = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_ft)
    model_ft = model_ft.to(config.device)

    model_ft = torch.nn.parallel.DistributedDataParallel(
        model_ft,
        device_ids=[config.local_rank],
        output_device=config.local_rank,
        find_unused_parameters=True
    )
    # model_ft = model_ft.to(config.device)
    
    criterion = nn.CrossEntropyLoss()#交叉熵损失函数
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)#优化器 model_ft.parameters()获取网络参数 SGD是随机梯度下降 momentum 动量加速
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)#每过step_size个epoch，做一次更新 gamma更新lr的乘法因子

    model_ft, tb = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, 
        dataset_sizes, config.device, num_epochs=30, cnv=cnv)#调用train_model（train）
    tb.close()
    os.makedirs(os.getcwd()+f'/results/models_resnet50/', exist_ok=True)
    if k < 0:
        save_model = os.getcwd()+f'/results/models_resnet50/{model}'
    else:
        save_model = os.getcwd()+f'/results/models_resnet50/{model}_{k}'
    if cnv:
        save_model = save_model + '_cnv'
    save_model = save_model + '.pkl'
    
    torch.save(model_ft.module.state_dict(), save_model)


def main(config, ocs, classification, K, cnv, isCrossValidation, use_model='resnet18'):
    print(isCrossValidation)
    #train_test_splitter.main("/media/zw/Elements1/xisx/backup/data/HE/tcga/tiles_cn", "/home/xisx/tmbpredictor/labels/uteri_median.csv", "halves", isCrossValidation=isCrossValidation)
    from collections import OrderedDict
    
    if isCrossValidation:
        for k in range(K):
            generative_model(config, use_model, k, cnv=cnv)
            #path = os.getcwd()+f'/results/models/PIK3CA/resnet18_{k}'
            #if cnv:
                #path = path + '_cnv'
            #net = Net(models.__dict__["resnet18"]())
            # model_ft = torch.load(path + '.pkl')
            #test(net, path + '.pkl', "resnet18", k, 5)
    else:
        os.makedirs(os.getcwd()+f'/data/', exist_ok=True)
        data_path = os.getcwd()+f"/data/train.csv"
        generative_model(use_model, data_path, -11, cnv=cnv)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script',
                                     epilog="authorized by geneis ")

    parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument("--nproc_per_node", type=int, default=1)                                 
    parser.add_argument("--classification", type=int, default=2)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--cnv", type=str, default="")
    parser.add_argument("--isCrossValidation", type=bool, default=True)
    parser.add_argument("--model", type=str, default='resnet18')
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    print("local_rank is ", args.local_rank)
    torch.cuda.set_device(args.local_rank)
    # GPU Number
    world_size = args.world_size
    print('world_size', world_size, type(world_size))
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=args.local_rank,
    )

    print("here")
    device = torch.device('cuda:{}'.format(args.local_rank))

    config = Config()
    config.ngpu = world_size
    config.device = device
    config.local_rank = args.local_rank

    print(config)
    origirn_classfication_set = None
    main(config, origirn_classfication_set, args.classification, args.K, args.cnv, args.isCrossValidation, use_model=args.model)
