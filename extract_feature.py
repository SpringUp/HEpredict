# -*- coding: utf-8 -*-
import os
import torch
import pickle
import pandas as pd
import argparse
from collections import OrderedDict
import numpy as np
import collections
import torch.nn as nn
from torchvision import transforms
from utils.custom_dset import CustomDset
from utils.resnet_custom import resnet18_baseline

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def load_data(cnv,file_csv,A,pretrain_weight,model_name):

    print('sss')
    if A=='train':
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(224),  # 224
                # transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    else:
        data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),  # 从中心裁剪
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    sets = CustomDset(file_csv, data_transforms[A])
    train_loader = torch.utils.data.DataLoader(sets, batch_size=4, shuffle=False,
                                               num_workers=4, pin_memory=True)
    bags = collections.defaultdict(list)
    bag_label = []
    SlideNames = []
    FeatList = []
    device = torch.device('cuda')

    print('loading model checkpoint')
    model = resnet18_baseline(cnv=False, pretrained=False, model_name=model_name)
    model1= torch.load(pretrain_weight)
    model_dict = model1.state_dict()
    model.load_state_dict(model_dict,strict=True)
    model = model.to(device)
    #if torch.cuda.device_count() > 1:
        #model = nn.DataParallel(model)

    with torch.no_grad():
        for data1 in train_loader:
            inputs, labels, names_, image_name = data1
            inputs = inputs.to(device)
            features = model(inputs)

            features = features.cpu().numpy()
            for i in range(labels.size(0)):
                people = names_[i]
                if people not in bags.keys():
                    bags[people]=list()
                    SlideNames.append(people)
                    bag_label.append(labels[i].item())
                bags[people].append(features[i])

    WSI_feature= []
    labels=0
    for slideName in bags.keys():
        labels+=1
        patch_data_list = bags[slideName]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch)
            tfeat=tfeat.unsqueeze(0)
            featGroup.append(tfeat)
        featGroup = torch.cat(featGroup, dim=0)  ## numPatch x fs
        feature_wsi = torch.mean(featGroup, dim=0, keepdim=True)
        feature_wsi = feature_wsi.cpu().numpy() 
        if slideName.startswith('TCGA'):
            slideName = slideName.split('.')[0]
        WSI_feature.append([slideName] + feature_wsi.flatten().tolist())
    return WSI_feature

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='abc')
    parser.add_argument('--cnv', default=False, type=bool)
    parser.add_argument('--file_csv', default='./test_csv', type=str)
    parser.add_argument('--pretrain_weight', default='./',
                        type=str)
    parser.add_argument('--pretrain_model', default='resnet18', type=str)
    parser.add_argument('--output_feature_file', default='output_feature', type=str)
    params = parser.parse_args()


    WSI_feature = load_data(params.cnv,file_csv=params.file_csv,A='test', pretrain_weight = params.pretrain_weight,model_name=params.pretrain_model)
    pdf = pd.DataFrame(WSI_feature, columns=['ID', 'feature','label']).sort_values(by=['ID'], ascending=True).reset_index(drop=True)

    pdf.to_csv(os.path.join(os.getcwd(), params.output_feature_file+ f'.csv'), index=None, header=None)