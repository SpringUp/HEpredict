import sys
import os
from pathlib import Path
from glob import glob
import fire
import pandas as pd
from extract_feature import load_data


def globToCsvAndExtractFeatures(dir_pngs, dir_output_parent, pretrain_weight, pretrain_model):
    list_pngs = glob(os.path.join(dir_pngs, '*png'))
    df_pngs = pd.DataFrame({'png':list_pngs, 'label': -1})
    dir_output = os.path.join(dir_output_parent, Path(dir_pngs).name)
    os.makedirs(dir_output, exist_ok=True)
    file_csv = os.path.join(dir_output, 'pngs.csv')
    df_pngs.to_csv(file_csv, index=False, header=False)

    WSI_feature = load_data(False, file_csv=file_csv, A='test', pretrain_weight = pretrain_weight,model_name=pretrain_model)
    pdf = pd.DataFrame(WSI_feature).sort_values(by=[0], ascending=True).reset_index(drop=True)

    path_feature = os.path.join(dir_output, 'feature.csv')
    pdf.to_csv(path_feature, index=None, header=None)

if __name__ == '__main__':
    fire.Fire(globToCsvAndExtractFeatures)