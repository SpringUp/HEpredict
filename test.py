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
from eval_metrics import calculate_metrics
from plot_confusion_matrix import plot_conf_mat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),#从中心裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def test(model, model_path, model_name, k=0, K=10, types=0, cnv=False, split_folder='halves', specify_tile_path_csv=None, specify_output_folder=None):
    if model_path and os.path.exists(model_path):
        model = torch.load(model_path)
    model.to(device)
    # model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    # model = nn.parallel.DistributedDataParallel(model, device_ids=range(1))
    # model.to(device)

    if cnv:
        cnv_feature=pd.read_csv(cnv)
        peoples=[i for i in cnv_feature.id]
        features=[cnv_feature[i] for i in cnv_feature.columns[1:]]
        min_max_scaler = preprocessing.MinMaxScaler()#归一化
        cnv_features = min_max_scaler.fit_transform(features)#数据标准化
    
    if isinstance(k, int):
        path_csv = os.getcwd()+f'/{split_folder}/test_{k}.csv'
    elif specify_tile_path_csv is not None:
        path_csv = specify_tile_path_csv
    else:
        raise Exception('k must be int or a path, got: {}'.format(k))
    testset = CustomDset(path_csv, data_transforms['test'])
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
        # open(f'{model_name}_confusion_matrix_classification_{types}.txt', 'a+').write(str(person_prob_dict[key]['label'])+"\t"+str(predict)+'\n')
    
    if specify_output_folder:
        output_folder = specify_output_folder
    else:
        output_folder = os.path.join(os.getcwd(), 'results')
    os.makedirs(output_folder, exist_ok=True)
        
    np.save(os.path.join(output_folder, f'y_true_{k}.npy'), np.array(y_true))
    np.save(os.path.join(output_folder, f'score_{k}.npy'), np.array(score_list))
    np.save(os.path.join(output_folder, f'y_pred_{k}.npy'), np.array(y_pred))

    df_probas = pd.DataFrame(person_prob_dict).T
    df_probas.loc[:,'predict'] = df_probas.apply(lambda row:1 if row['prob_0'] < row['prob_1'] else 0, axis=1)
    df_probas.loc[:,'correct'] = df_probas.apply(lambda row:1 if row['predict'] == row['label'] else 0, axis=1)
    df_probas.to_csv(os.path.join(output_folder, 'df_probas.tsv'), sep='\t')
    
    array_y_true = np.array(y_true)
    array_score_list = np.array(score_list)
    array_y_pred = np.array(y_pred)

    dict_metrics = calculate_metrics(array_y_true, array_score_list, array_y_pred)
    series_metrics = pd.Series(dict_metrics)
    series_metrics.to_csv(os.path.join(output_folder, 'metrics.tsv'), sep='\t', header=False)

    plot_conf_mat(array_y_true, array_y_pred, path_output=os.path.join(output_folder, 'confusion_matrix.pdf'), title='confusion matrix')

    logger.info('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
    from Net import Net
    path_model_pkl, path_csv, dir_output = sys.argv[1:4]
    test(Net(models.__dict__['resnet18']()), path_model_pkl, "useless", \
        k=None, K=None, specify_tile_path_csv=path_csv, specify_output_folder=dir_output)
