#!/bin/bash

# Purpose: run center-area counting ablations on the stitched-walk blueberry MOT dataset.

#SBATCH -J 20-area_count_ablation
#SBATCH -o logs/20-area_count_ablation-stitched_walk.out.txt
#SBATCH -e logs/20-area_count_ablation-stitched_walk.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/.bashrc
source activate yolo

cd /home/wh1234_/code/Counting

mkdir -p logs

export PYTHONPATH="/home/wh1234_/code/Counting/trackers:${PYTHONPATH}"

DATA_ROOT="/home/wh1234_/data/blueberry_mot_stitched_walk"
OUTPUT_DIR="/home/wh1234_/code/Counting/output/count/stitched_walk_boxmot3/area_ablation"
REID_PATH="/home/wh1234_/code/Counting/weights/osnet_ain_x1_0_blueberry.pt"
TRACKER_CONFIG_DIR="/home/wh1234_/code/Counting/configs/trackers"
SCRIPT_PATH="count/blueberry_area_ablation.py"

echo "20-area_count_ablation, run center-area counting ablations on stitched_walk with boxmot3"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "REID_PATH=${REID_PATH}"
echo "TRACKER_CONFIG_DIR=${TRACKER_CONFIG_DIR}"

python "${SCRIPT_PATH}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --reid-path "${REID_PATH}" \
  --tracker-config-dir "${TRACKER_CONFIG_DIR}" \
  --device cuda

echo "Job finished at: $(date)"
