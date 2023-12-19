import os
import numpy as np
import torch
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn import metrics
from collections import Counter

def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction)
    return c_auc, threshold_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def calculate_metrics(label, score_list, prob):
    auc = roc_auc_score(label, score_list[:,1])
    accuracy = metrics.accuracy_score(label, prob)
    precision = metrics.precision_score(label, prob)  #average='micro'
    recall = metrics.recall_score(label, prob )

    F1 = 2*(precision * recall) / (precision + recall+1e-12)
    dict_to_return = {
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'F1':F1,
        'auc':auc,
    }
    return dict_to_return

if __name__ == '__main__':
    label = np.load(os.getcwd()+f'/save_result/y_true_nm_qingyi_Sanger_revision_clinc.npy')
    score_list = np.load(os.getcwd()+f'/save_result/score_nm_qingyi_Sanger_revision_clinc.npy')
    prob = np.load(os.getcwd()+f'/save_result/y_pred_nm_qingyi_Sanger_revision_clinc.npy')
    print(len(score_list))
    print(Counter(label))
    auc = roc_auc_score(label, score_list[:,1])
    print(auc)
    accuracy = metrics.accuracy_score(label, prob)
    precision = metrics.precision_score(label, prob)  #average='micro'
    recall = metrics.recall_score(label, prob )

    F1 = 2*(precision * recall) / (precision + recall+1e-12)
    print(f'accuracy={accuracy},precision={precision},recall={recall},F1={F1},auc={auc}')

