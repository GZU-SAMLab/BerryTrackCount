#!/bin/bash

#SBATCH -J 1-opt-fcm
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
echo ""1-opt-fcm, replace C2f-C3k2 in the backbone with FCM""
python detector/FCM.py  
echo "Job finished at: $(date)"
