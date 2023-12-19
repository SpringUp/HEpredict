# -*- coding: UTF-8 -*-

import os
import random
from glob import glob
import pandas as pd
import numpy as np
from pathlib import Path
import fire

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

def useCrossValidation(X, y, save_dir, K, tile_suffix='.png', random_seed=1):
    skf = StratifiedKFold(n_splits=K, shuffle=True)

    save_dir_for_slide_id = '{}_slide_id'.format(save_dir.rstrip(os.path.sep))
    os.makedirs(save_dir_for_slide_id, exist_ok=True)

    for fold, (train, test) in enumerate(skf.split(X, y)):
        train_data = []
        test_data = []

        df_slide_train = pd.concat([pd.Series(X).iloc[train], pd.Series(y).iloc[train]], axis=1)
        df_slide_train.columns = ['path', 'label']
        df_slide_test = pd.concat([pd.Series(X).iloc[test], pd.Series(y).iloc[test]], axis=1)
        df_slide_test.columns = ['path', 'label']

        df_slide_train.loc[:,'id'] = df_slide_train['path'].map(lambda x:Path(x).name)
        df_slide_test.loc[:,'id'] = df_slide_test['path'].map(lambda x:Path(x).name)

        df_slide_train.to_csv(os.path.join(save_dir_for_slide_id, f'train_{fold}.csv'), index=False)
        df_slide_test.to_csv(os.path.join(save_dir_for_slide_id, f'test_{fold}.csv'), index=False)

        train_set, train_label = df_slide_train['path'], df_slide_train['label']
        test_set, test_label = df_slide_test['path'], df_slide_test['label']
        
        for (data, label) in zip(train_set, train_label):
            for img in glob(os.path.join(data, '*{}'.format(tile_suffix))):
                train_data.append((img, label)) 
        for (data, label) in zip(test_set, test_label):
            for img in glob(os.path.join(data, '*{}'.format(tile_suffix))):
                test_data.append((img, label))

        df_data_train = pd.DataFrame(train_data, columns=['img', 'label']).sort_values(by=['label'], ascending=True).reset_index(drop = True)
        
        # Get the smallest number of image in each category
        min_num = min(df_data_train['label'].value_counts())
        
        downsampled_dfs_for_each_label = []

        for label, df_this_label in df_data_train.groupby('label'):
            df_this_label_downsampled = df_this_label.sample(n=min_num, replace=False, random_state=random_seed)
            downsampled_dfs_for_each_label.append(df_this_label_downsampled)
        df_data_train = pd.concat(downsampled_dfs_for_each_label)
        df_data_train.reset_index(inplace=True, drop=True)
        
        # Shuffle
        df_data_train = df_data_train.reindex(np.random.permutation(df_data_train.index)).reset_index(drop = True)
        df_data_train.to_csv(os.path.join(save_dir, f"train_{fold}.csv"), index=None, header=None)

        df_data_test = pd.DataFrame(test_data)
        df_data_test.to_csv(os.path.join(save_dir, f"test_{fold}.csv"), index=None, header=None)


def split_according_to_csv(path_df_features_and_labels, save_dir, K=5, col_tiles_dir='tiles_dir', col_label='TMB', tile_suffix='.png', random_seed=1):
    '''
    path_df_features_and_labels: df_features_and_labels的存储路径
    df_features_and_labels应包含至少以下变量所指向的列：
        col_tiles_dir: 该样本的tiles所在文件夹
        col_label: 样本训练标签
    df_features_and_labels只允许存储本次实验感兴趣的样本，本文件中所有样本都会参与分割
    save_dir: 分割产生csv文件的存储文件夹
    K: 按照K折交叉验证方式进行分割
    tile_suffix: tiles文件的后缀
    '''
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.makedirs(save_dir, exist_ok=True)

    assert Path(path_df_features_and_labels).suffix in ['.csv','.tsv'], "Error: 标签文件需要是csv文件"
    
    df_features_and_labels = pd.read_csv(path_df_features_and_labels, sep=',' if path_df_features_and_labels.endswith(',') else '\t')
    
    assert len(df_features_and_labels[col_label].unique()) > 1, 'col_label中类别数目应大于1'
    label_encoder = LabelEncoder()
    df_features_and_labels.loc[:,'encoded_labels'] = label_encoder.fit_transform(df_features_and_labels[col_label])
    
    print('label_encoder.classes_: {}'.format(label_encoder.classes_))
    df_encoding = pd.DataFrame(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    df_encoding.columns = ['class', 'encoded']
    df_encoding.to_csv(os.path.join(save_dir, 'encode_labels.csv'), index=False)

    useCrossValidation(df_features_and_labels[col_tiles_dir].tolist(), 
        df_features_and_labels['encoded_labels'].tolist(), 
        save_dir, K, tile_suffix=tile_suffix, random_seed=random_seed)

if __name__ == '__main__':
    fire.Fire(split_according_to_csv)
