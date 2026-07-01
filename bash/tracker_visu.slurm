#!/bin/bash

#SBATCH -J 22-tracker_visu
#SBATCH -o logs/22-tracker_visu.out.txt
#SBATCH -e logs/22-tracker_visu.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/.bashrc
source activate yolo310
# 进入工作目录
cd /home/wh1234_/code/Counting

echo "22-tracker_visu: Visualization of tracking results from different trackers on a single video. Start time: $(date)"
# python visualize/tracker_visu.py \
#     --video /home/wh1234_/data/video/10s/20250425block4.mp4 \
#     --reid-weights weights/osnet_ain_x1_0_blueberry.pt \
#     --tracker-config-dir configs/trackers \
#     --output-dir output/visualize/tracker

python visualize/tracker_visu.py \
    --sequence-dir /home/wh1234_/data/blueberry_mot_stitched_walk/test/Blueberry-Test-10 \
    --reid-weights weights/osnet_ain_x1_0_blueberry.pt \
    --tracker-config-dir configs/tracker \
    --output-dir output/visualize/tracker/seq-10

echo "Job finished at: $(date)"
