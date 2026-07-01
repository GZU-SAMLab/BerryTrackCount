#!/bin/bash

#SBATCH -J 21-count_apply
#SBATCH -o logs/21-count_apply.out.txt
#SBATCH -e logs/21-count_apply.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/.bashrc
source activate yolo
# 进入工作目录
cd /home/wh1234_/code/Counting

echo "21-count_apply: Counting application of BerryTracker on real-world videos. Start time: $(date)"
python tools/count_apply.py \
  --video-path-json dataset/video_path.json \
  --yolo-weights weights/berrydet_s.pt \
  --reid-weights weights/osnet_ain_x1_0_blueberry.pt \
  --tracker-config configs/trackers/mytrack.yaml \
  --output-dir output/count_apply
echo "Job finished at: $(date)"
