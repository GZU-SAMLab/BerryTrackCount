#!/bin/bash

# Purpose: run MyTrack ablations on the stitched-walk blueberry MOT dataset.

#SBATCH -J 14-mytrack_ablate
#SBATCH -o logs/14-mytrack-stitched_walk.out.txt
#SBATCH -e logs/14-mytrack-stitched_walk.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/.bashrc
source activate yolo

cd /home/wh1234_/code/Counting

mkdir -p logs

export PYTHONPATH="/home/wh1234_/code/Counting/trackers:/home/wh1234_/code/Counting/evaluation/TrackEval:${PYTHONPATH}"


SCRIPT_PATH="trackers/boxmot/trackers/mytrack/ablate.py"

echo "14-mytrack_ablate: Ablation study of MyTrack on the blueberry_mot_stitched_walk dataset using berrydet_s.pt weights. The configuration file is configs/trackers, and results are output to output/eval/mytrack_ablation_boxmot3. Start time: $(date)"
python "${SCRIPT_PATH}" \
  --source /home/wh1234_/data/blueberry_mot_stitched_walk \
  --yolo-model weights/berrydet_s.pt \
  --reid-model weights/osnet_ain_x1_0_blueberry.pt \
  --tracker_config_dir configs/trackers \
  --output-dir output/eval/mytrack_ablation_boxmot3 \
  --device 0

echo "Job finished at: $(date)"
