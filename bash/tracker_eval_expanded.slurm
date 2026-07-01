#!/bin/bash

#SBATCH -J tracker_eval_with_appearance
#SBATCH -o logs/9-tracker_eval_with_appearance.out.txt
#SBATCH -e logs/9-tracker_eval_with_appearance.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/.bashrc
source activate yolo310
# 进入工作目录
cd /home/wh1234_/code/Counting

echo "9-tracker_eval,tracker_eval: Evaluate the tracker on the blueberry_mot_stitched_walk dataset using the berrydet_s.pt weights and an appearance model. The configuration file is configs/trackers, and the results are output to output/eval/1_stwalk_boxmot3. Start time: $(date)"
python evaluation/1_tracker_eval.py \
    --data_root /home/wh1234_/data/blueberry_mot_stitched_walk \
    --yolo_weights weights/berrydet_s.pt \
    --reid_path weights/osnet_ain_x1_0_blueberry.pt \
    --tracker_config_dir configs/trackers \
    --output_dir output/eval/1_stwalk_boxmot3
echo "Job finished at: $(date)"
