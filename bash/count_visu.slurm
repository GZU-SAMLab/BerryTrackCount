#!/bin/bash

# Purpose: run BerryTracker counting visualization on selected videos and MOT sequence.

#SBATCH -J 24-count_visu
#SBATCH -o logs/24-count_visu.out.txt
#SBATCH -e logs/24-count_visu.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/.bashrc
source activate yolo

cd /home/wh1234_/code/Counting

mkdir -p logs output/count_visu

export PYTHONPATH="/home/wh1234_/code/Counting/trackers:${PYTHONPATH}"

SCRIPT_PATH="tools/count_visu.py"

INPUTS=(
  "/home/wh1234_/data/video/10s/20250427block8.mp4"
  "/home/wh1234_/data/video/count_apply/block3.mp4"
  "/home/wh1234_/data/video/count_apply/block4.mp4"
  "/home/wh1234_/data/blueberry_mot_stitched_walk/train/Blueberry-Train-10"
)

echo "24-count_visu, run BerryTracker count visualization"
echo "Start time: $(date)"

python tools/count_visu.py \
  --input "${INPUTS[@]}" \
  --output-dir output/count_visu \
  --yolo-weights weights/berrydet_s.pt \
  --reid-weights weights/osnet_ain_x1_0_blueberry.pt \
  --tracker-config configs/trackers/mytrack.yaml

echo "Job finished at: $(date)"
