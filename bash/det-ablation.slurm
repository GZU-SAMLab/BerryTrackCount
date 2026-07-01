#!/bin/bash

#SBATCH -J 2-msdaf
#SBATCH -o logs/9-det-ablation2.out.txt
#SBATCH -e logs/9-det-ablation2.err.txt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --gres=gpu:1

source ~/.bashrc
source activate yolo
# 进入工作目录
cd /home/wh1234_/code/Counting
echo "9-det-ablation, improved YOLO11 detector ablation experiments. The original YOLO11 is trained on the yolo_82_640_20250731 dataset with n, s, m, l variants."
python detector/det_ablation.py  
echo "Job finished at: $(date)"
