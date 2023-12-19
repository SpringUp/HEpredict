from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    classes = ['0','1']
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # �ڻ���������ÿ��ĸ���ֵ
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=20, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix',fontsize=23)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label',fontsize=23)
    plt.xlabel('Predict label',fontsize=23)

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='pdf')

def plot_conf_mat(y_true, y_pre, path_output, title='confusion matrix'):
    cm = confusion_matrix(y_true, y_pre)
    plot_confusion_matrix(cm, path_output, title=title)

if __name__ == '__main__':
    y_pre= np.load(os.getcwd()+f'/save_model/y_pred_nm_qingyi_TCGA_Sanger_revision_test.npy')
    y_true = np.load(os.getcwd()+f'/save_model/y_true_nm_qingyi_TCGA_Sanger_revision_test.npy')
    print(y_pre)
    print('.........')
    print(y_true)
    cm = confusion_matrix(y_true, y_pre)
    plot_confusion_matrix(cm, 'confusion_matrix_y_true_nm_qingyi_TCGA_Sanger_revision_test.pdf', title='confusion matrix')