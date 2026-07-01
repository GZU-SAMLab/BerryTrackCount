#!/bin/bash

#SBATCH -J 17-train-Reid
#SBATCH -o logs/%j.out.txt
#SBATCH -e logs/%j.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/.bashrc
source activate yolo

cd /home/wh1234_/code/Counting
mkdir -p logs weights/tracker

export PYTHONPATH="/home/wh1234_/code/Counting/trackers:${PYTHONPATH}"

DATA_ROOT="/home/wh1234_/data/blueberry_mot_stitched_walk"
REID_DATA_ROOT="/home/wh1234_/code/Counting/dataset/blueberry_reid_ain"
TRAIN_ENTRY="trackers/boxmot/trackers/mytrack/train_reid_osnet_ain_x1_0.py"
OUTPUT_CKPT="/home/wh1234_/code/Counting/weights/osnet_ain_x1_0_blueberry.pt"
LAST_CKPT="/home/wh1234_/code/Counting/weights/osnet_ain_x1_0_blueberry_last.pt"
INIT_WEIGHTS_PT="/home/wh1234_/code/Counting/weights/osnet_ain_x1_0_imagenet.pth"

echo "17-train-Reid, export blueberry ReID crops and train osnet_ain_x1_0"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MOT root: ${DATA_ROOT}"
echo "ReID root: ${REID_DATA_ROOT}"
echo "Training entry: ${TRAIN_ENTRY}"
echo "Output checkpoint: ${OUTPUT_CKPT}"
echo "Resume checkpoint: ${LAST_CKPT}"
echo "Init weights (.pth): ${INIT_WEIGHTS_PT}"

TRAIN_ARGS=(
  --data-root "${DATA_ROOT}"
  --reid-root "${REID_DATA_ROOT}"
  --output "${OUTPUT_CKPT}"
  --val-source tail
  --val-count 4
  --class-ids 0 1 2 3
  --min-box-size 8
  --min-area 64
  --visibility-thresh 0.0
  --conf-thresh 0.0
  --crop-margin 0.08
  --min-samples-per-id 2
  --image-format jpg
  --jpeg-quality 95
  --epochs 100
  --batch-size 64
  --instances-per-identity 4
  --lr 3e-4
  --weight-decay 5e-4
  --margin 0.3
  --ce-weight 1.0
  --tri-weight 1.0
  --label-smoothing 0.05
  --height 256
  --width 128
  --hflip-prob 0.5
  --color-jitter 0.05
  --random-erasing 0.15
  --num-workers 8
  --seed 42
  --device cuda
  --early-stop-patience 50
  --early-stop-min-delta 5e-4
)

if [ -f "${INIT_WEIGHTS_PT}" ]; then
  echo "Using initialization weights: ${INIT_WEIGHTS_PT}"
  TRAIN_ARGS+=(--init-weights "${INIT_WEIGHTS_PT}")
else
  echo "Initialization weights not found, training from scratch."
fi

if [ -f "${LAST_CKPT}" ]; then
  echo "Resuming from last checkpoint: ${LAST_CKPT}"
  TRAIN_ARGS+=(--resume)
fi

python "${TRAIN_ENTRY}" "${TRAIN_ARGS[@]}"

echo "Job finished at: $(date)"
