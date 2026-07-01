#!/bin/bash

# Purpose: run MyTrack association-function ablations on the stitched-walk blueberry MOT dataset.

#SBATCH -J 16-ablation_assofunc
#SBATCH -o logs/16-ablation_assofunc-stitched_walk.out.txt
#SBATCH -e logs/16-ablation_assofunc-stitched_walk.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/.bashrc
source activate yolo310

cd /home/wh1234_/code/Counting

mkdir -p logs
mkdir -p output/eval/mytrack_ablation_assofunc

export PYTHONPATH="/home/wh1234_/code/Counting/trackers:/home/wh1234_/code/Counting/evaluation/TrackEval:${PYTHONPATH}"

echo "16-ablation_assofunc, run MyTrack association-function ablations on blueberry_mot_stitched_walk"
SCRIPT_PATH="trackers/boxmot/trackers/mytrack/ablation_assofunc.py"

python "${SCRIPT_PATH}" \
  --source /home/wh1234_/data/blueberry_mot_stitched_walk \
  --yolo-model weights/berrydet_s.pt \
  --reid-model weights/osnet_ain_x1_0_blueberry.pt \
  --tracker-config-dir configs/trackers \
  --output-dir output/eval/mytrack_ablation_assofunc \
  --device 0

echo "Job finished at: $(date)"
