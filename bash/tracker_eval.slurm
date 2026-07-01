#!/bin/bash

#SBATCH -J tracker_eval
#SBATCH -o logs/9-tracker_eval.out.txt
#SBATCH -e logs/9-tracker_eval.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/.bashrc
source activate yolo310
# 进入工作目录
cd /home/wh1234_/code/Counting
echo "9-tracker_eval: Evaluate the tracker on the blueberry_mot_stitched_walk dataset using the berrydet_s.pt weights without an appearance model, with min_hits=2. Output results to output/eval/0_stwalk_boxmot_minhits2. Start time: $(date)"
python evaluation/0_tracker_eval.py \
    --data_root /home/wh1234_/data/blueberry_mot_stitched_walk \
    --yolo_weights weights/berrydet_s.pt \
    --tracker_config_dir configs/trackers \
    --output_dir output/eval/0_stwalk_boxmot_minhits2
echo "Job finished at: $(date)"
