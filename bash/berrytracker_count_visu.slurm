#!/bin/bash

#SBATCH -J 23-berrytracker_count_visu
#SBATCH -o logs/23-berrytracker_count_visu.out.txt
#SBATCH -e logs/23-berrytracker_count_visu.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/.bashrc
source activate yolo
# 进入工作目录
cd /home/wh1234_/code/Counting

echo "23-berrytracker_count_visu: Visualization of tracking performance for different trackers on a specific video. Start time: $(date)"
python visualize/berrytracker_count_visu.py \
    --sequence-dir /home/wh1234_/data/blueberry_mot_stitched_walk/test/Blueberry-Test-10 \
    --reid-weights weights/osnet_ain_x1_0_blueberry.pt \
    --tracker-config-dir configs/tracker \
    --output-dir output/visualize/tracker/BerryTrack_count/10

echo "Job finished at: $(date)"
