#!/bin/bash

# Purpose: run MyTrack appearance ablations on the stitched-walk blueberry MOT dataset.

#SBATCH -J 18-ablation_appearance
#SBATCH -o logs/18-ablation_appearance-stitched_walk.out.txt
#SBATCH -e logs/18-ablation_appearance-stitched_walk.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/.bashrc
source activate yolo

cd /home/wh1234_/code/Counting

mkdir -p logs
mkdir -p output/eval/ablation_appearance

export PYTHONPATH="/home/wh1234_/code/Counting/trackers:/home/wh1234_/code/Counting/evaluation/TrackEval:${PYTHONPATH}"

DATA_ROOT="/home/wh1234_/data/blueberry_mot_stitched_walk"
YOLO_MODEL="/home/wh1234_/code/Counting/weights/berrydet_s.pt"
REID_MODEL="/home/wh1234_/code/Counting/weights/osnet_ain_x1_0_blueberry.pt"
OUTPUT_DIR="/home/wh1234_/code/Counting/output/eval/ablation_appearance"
SCRIPT_PATH="trackers/boxmot/trackers/mytrack/ablation_appearance.py"

echo "18-ablation_appearance, run MyTrack appearance ablations on blueberry_mot_stitched_walk"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "YOLO_MODEL=${YOLO_MODEL}"
echo "REID_MODEL=${REID_MODEL}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python "${SCRIPT_PATH}" \
  --source "${DATA_ROOT}" \
  --yolo-model "${YOLO_MODEL}" \
  --reid-model "${REID_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --device 0

echo "Job finished at: $(date)"
