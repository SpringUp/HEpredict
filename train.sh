source /home/ubuntu/anaconda3/etc/profile.d/conda.sh

source activate pytorch_p39
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboard

python main_sync_wb.py --local_rank 0 --classification 2 --K 5 --cnv /mnt/efs/fs1/code/tmbclinic/clinical_with_necessary_columns.csv --model resnet18

