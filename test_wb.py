import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
from sklearn import metrics
import os
from utils.custom_dset import CustomDset
from utils.common import logger
import json
import sys
from sklearn import preprocessing
import pandas as pd
from Net import Cnn_With_Clinical_Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),#从中心裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def test(model, model_path, k=0, K=10, cnv="", split_folder='halves', output_prefix='test'):
    '''
    '''
    model.to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    # model = nn.parallel.DistributedDataParallel(model, device_ids=range(1))
    # model.to(device)

    if cnv:
        cnv_feature=pd.read_csv(cnv)
        peoples=[i for i in cnv_feature.id]
        features=[cnv_feature[i] for i in cnv_feature.columns[1:]]
        min_max_scaler = preprocessing.MinMaxScaler()#归一化
        cnv_features = min_max_scaler.fit_transform(features)#数据标准化
    
    testset = CustomDset(os.getcwd()+f'/{split_folder}/test_{k}.csv', data_transforms['test'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,shuffle=False, num_workers=8)

    #print(len(testloader))
    person_prob_dict = dict()
    with torch.no_grad():#停止自动求导
        for itd, data in enumerate(testloader):
            images, labels, names_, images_names = data
            if cnv:
                X_train_minmax = [cnv_features[:,peoples.index(i)] for i in names_]
                outputs = model(images.to(device), torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(device))
            else:
                outputs = model(images.to(device))
            probability = F.softmax(outputs, dim=1).data.squeeze()#按行做归一化，去掉为1的维度
            probs = probability.cpu().numpy()
            for i in range(labels.size(0)):
                p = names_[i]
                if p not in person_prob_dict.keys():
                    person_prob_dict[p] = {
                        'prob_0': 0, 
                        'prob_1': 0,
                        'label': labels[i].item(),#遍历
                        'img_num': 0}
                if probs.ndim == 2:
                    person_prob_dict[p]['prob_0'] += probs[i, 0]
                    person_prob_dict[p]['prob_1'] += probs[i, 1]
                    person_prob_dict[p]['img_num'] += 1
                else:
                    person_prob_dict[p]['prob_0'] += probs[0]
                    person_prob_dict[p]['prob_1'] += probs[1]
                    person_prob_dict[p]['img_num'] += 1

    y_true = []
    y_pred = []
    score_list = []

    total = len(person_prob_dict)
    correct = 0
    for key in person_prob_dict.keys():
        predict = 0
        if person_prob_dict[key]['prob_0'] < person_prob_dict[key]['prob_1']:
            predict = 1
        if person_prob_dict[key]['label'] == predict:
            correct += 1
        y_true.append(person_prob_dict[key]['label'])
        score_list.append([person_prob_dict[key]['prob_0']/person_prob_dict[key]["img_num"],person_prob_dict[key]['prob_1']/person_prob_dict[key]["img_num"]])
        y_pred.append(predict)

    os.makedirs(output_prefix, exist_ok=True)
    np.save(f'{output_prefix}/y_true_{k}.npy', np.array(y_true))
    np.save(f'{output_prefix}/score_{k}.npy', np.array(score_list))
    np.save(f'{output_prefix}/y_pred_{k}.npy', np.array(y_pred))
    logger.info('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))

if __name__ == '__main__':
    
    test(model=Cnn_With_Clinical_Net(models.__dict__['resnet18']()), model_path='results.bk20220628_162800/models_resnet50/resnet18_0_cnv.pkl', \
        k=0, K=5, cnv="/mnt/efs/fs1/code/tmbclinic/clinical_with_necessary_columns.csv", split_folder='halves', output_prefix='test')
