#!/bin/bash

# Purpose: run blueberry counting on the stitched-walk dataset with multiple trackers and counter strategies.

#SBATCH -J 19-count
#SBATCH -o logs/19-count-stitched_walk.out.txt
#SBATCH -e logs/19-count-stitched_walk.err.txt
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
OUTPUT_DIR="/home/wh1234_/code/Counting/output/count/stitched_walk_boxmot3"
REID_PATH="/home/wh1234_/code/Counting/weights/osnet_ain_x1_0_blueberry.pt"
TRACKER_CONFIG_DIR="/home/wh1234_/code/Counting/configs/trackers"
SCRIPT_PATH="count/blueberry_count.py"

echo "19-count, run blueberry counting on stitched_walk with multiple trackers"
echo "Start time: $(date)"

python "${SCRIPT_PATH}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --reid-path "${REID_PATH}" \
  --tracker-config-dir "${TRACKER_CONFIG_DIR}" \
  --device cuda

echo "Job finished at: $(date)"
