# -*- coding:utf-8 -*-
import pandas as pd

import os
import argparse

def feature_extract_fusion(setting):
	dfs=[]
	for root, dirs, files in os.walk(setting.data_root_dir):
		for dir in dirs:
			df = pd.read_csv(os.path.join(root,f'{dir}/feature.csv'), index_col=0, header=None)
			dfs.append(df)
	pd.concat(dfs).to_csv(setting.save_file, index=True, header=False)



if __name__=="__main__":
	parser = argparse.ArgumentParser(description="feature extraction from patch.")
	parser.add_argument("--data_root_dir", default='/data/data/FromHospital/QingYi-HE-PCR-MSI-QianZhan-Colon/Features_Of_Tile_From_Tiatoolbox040/', type=str)
	parser.add_argument("--save_file", default='resnet18_qingyiqianzhan_feature.csv', type=str)
	setting = parser.parse_args()
	feature_extract_fusion(setting)