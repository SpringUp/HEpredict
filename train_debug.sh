source activate pytorch_p39

export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.launch --nproc_per_node=3 --master_port 29501 main_sync_wb_debug.py  --classification 2 --K 5 --cnv /mnt/efs/fs1/code/tmbclinic/clinical_with_necessary_columns.csv --model resnet18 --world_size 3  
#python main_sync_wb.py --local_rank 0 --classification 2 --K 5 --cnv /mnt/efs/fs1/code/tmbclinic/clinical_with_necessary_columns.csv --model resnet18 --world_size 4

