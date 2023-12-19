source /home/ubuntu/anaconda3/etc/profile.d/conda.sh

source activate pytorch_p39
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboard

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 main_sync_wb.py  --classification 2 --K 5 --cnv /mnt/efs/fs1/code/tmbclinic/clinical_with_necessary_columns.csv --model resnet18 --world_size 4  > `hostname`_`date +%Y%m%d_%H%M%S`.log 2>&1
#python main_sync_wb.py --local_rank 0 --classification 2 --K 5 --cnv /mnt/efs/fs1/code/tmbclinic/clinical_with_necessary_columns.csv --model resnet18 --world_size 4

