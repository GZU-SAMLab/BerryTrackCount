#!/bin/bash

#SBATCH -J 2-msdaf
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
echo "2-msdaf, replace C2f-C3k2 in the backbone with MSDAF"
python detector/MSDAF.py  
echo "Job finished at: $(date)"
