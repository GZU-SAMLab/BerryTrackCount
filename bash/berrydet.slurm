#!/bin/bash

#SBATCH -J 7-berrydet_train
#SBATCH -o logs/%j.out.txt
#SBATCH -e logs/%j.err.txt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --gres=gpu:1

source ~/.bashrc
source activate yolo
# 进入工作目录
cd /home/wh1234_/code/Counting
echo "7-berrydet_train, replace the C3k2 module in the YOLO11 backbone with the FCM module (optimized with residual connection)"
python detector/BerryDet.py  
echo "Job finished at: $(date)"
