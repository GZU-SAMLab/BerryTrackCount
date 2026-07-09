# BerryTrackCount

BerryTrackCount is a video-based phenotyping framework for detecting, tracking, and counting blueberry flowers and fruits across multiple phenological stages. It integrates scale-sensitive detection, identity-preserving multi-object tracking, and trajectory-gated phenological counting to support high-throughput phenotyping, yield estimation, harvest planning, and precision orchard management.

This repository provides the main code for the BerryTrackCount pipeline, including detection, tracking, counting, evaluation, visualization, and data-processing tools.

## Highlights

- Constructed a phenology-aware blueberry video counting dataset, including 368 detection images, 66,717 annotations, 40 MOT sequences, 9,213 frames, and 14,819 trajectories.
- Developed BerryDet, a scale-sensitive blueberry detector that integrates the Micro-target Spatial--Semantic Coupling Module (MSSC) and Slicing Aided Hyper Inference (SAHI) to improve small-target representation in high-resolution orchard images.
- Designed BerryTracker by introducing Vertical-Consistency Modulated Complete IoU (VCM-CIoU) and Trajectory-Conditioned Appearance Reconstruction (TCAR) to improve identity association in dense and visually homogeneous blueberry clusters.
- Proposed Phenology-specific Trajectory-Gated Counting (PTGC), which converts stabilized trajectories into category-specific counts while reducing repeated counting caused by continuous video sampling.
- Reported paper results include 89.1% mAP@0.5 for BerryDet-s, 62.08% MOTA and 77.35% IDF1 for BerryTracker, and 93.48% counting accuracy with an R^2 of 0.97 for the complete BerryTrackCount framework.

## Framework

BerryTrackCount follows a detection--tracking--counting pipeline:

1. **BerryDet detection**  
   Detects blueberry flowers and fruits in each video frame and outputs bounding boxes, class labels, and confidence scores.

2. **BerryTracker association**  
   Maintains target identities across frames by combining VCM-CIoU geometric similarity, VDC motion consistency, and TCAR-based appearance similarity.

3. **PTGC counting**  
   Counts each trajectory only when it first enters the predefined counting region and maintains separate ID sets for `Flower`, `Green`, `Light Purple`, and `Blue`.

The four counting classes are:

```text
Flower
Green
Light Purple
Blue

## Repository Structure

```text
.
|-- bash/                  # SLURM scripts for training, counting, evaluation, and visualization
|-- configs/               # Dataset, detector, and tracker configuration files
|-- count/                 # Blueberry counting pipeline and reusable counter logic
|-- detector/              # Detector modules, ablation code, and training helpers
|-- evaluation/            # Tracker and counter evaluation scripts, including TrackEval
|-- tools/                 # Data conversion, video detection, visualization, and utility tools
|-- trackers/              # BoxMOT-based tracking code and BerryTracker implementation
|-- ultralytics/           # Local Ultralytics/YOLO code used by the detector
|-- visualize/             # Tracking, counting, heatmap, ablation, and result visualization scripts
`-- requirements.txt
```

## Installation

Create a Python environment with PyTorch support, then install the repository dependencies:

```bash
pip install -r requirements.txt
```

The requirements file installs local editable packages for:

```text
./trackers
./pytorch-grad-cam
./evaluation/TrackEval
```

For GPU inference or training, install a PyTorch build that matches your CUDA runtime before installing the remaining dependencies.

## Required Assets

Download the dataset and model weights from Baidu Netdisk [here](https://pan.baidu.com/s/1TPP9mp5VCl4D0D0pyIdmTA?pwd=1234).

After downloading, place the files under the repository root following the layout below before running the full pipeline. All model weights are stored directly in `weights/`.

```text
weights/
|-- yolo11s.pt
|-- yolo11l.pt
`-- osnet_ain_x1_0_blueberry.pt
```

The `dataset/` directory contains two blueberry datasets:

```text
dataset/
|-- 20251027_yolo_82_640/          # Blueberry detection image dataset in YOLO format
|   |-- images/
|   |   |-- train/
|   |   `-- val/
|   `-- labels/
|       |-- train/
|       `-- val/
`-- blueberry_mot_stitched_walk/   # Blueberry multi-object tracking dataset
    |-- seqmaps/
    |-- train/
    |-- test/
    |-- manifest.json
    |-- train_manifest.json
    `-- test_manifest.json
```

For detector training, update `configs/data/mydata.yaml` so that `path` points to `dataset/20251027_yolo_82_640` or the absolute path where this YOLO-format detection dataset is stored:

```yaml
path: dataset/20251027_yolo_82_640
train: images/train
val: images/val

nc: 4
names: ['Flower', 'Green', 'Light Purple', 'Blue']
```

## Training

Detector  and ReID training:

```text
detector/BerryDet.py
trackers/boxmot/trackers/mytrack/train_reid_osnet_ain_x1_0.py
```

These scripts use the local `ultralytics` package and read dataset settings from `configs/data/mydata.yaml`. Some helpers contain hard-coded experiment paths and config names for the original training environment; check the paths in the target script before launching. SLURM launch examples are provided in `bash/`.

## Evaluation

Evaluate trackers on the blueberry MOT test set:

```bash
python evaluation/1_tracker_eval.py \
  --data_root dataset/blueberry_mot_stitched_walk \
  --yolo_weights weights/berrydet_s.pt \
  --tracker_config_dir configs/trackers \
  --reid_path weights/osnet_ain_x1_0_blueberry.pt \
  --output_dir output/eval/blueberry_mot_stitched_walk
```

Evaluate counting outputs against ground-truth count CSV files:

```bash
python evaluation/1_counter_eval.py \
  --gt dataset/mot_Count_GT/stitched_walk_count_GT.csv \
  --pred-dir output/count/stitched_walk \
  --out output/count_eval/stitched_walk/counter_eval.csv \
  --acc-out output/count_eval/stitched_walk/counter_accuracy_by_sequence.csv
```

The counter evaluation reports Accuracy, GEH, R^2, RMSE, and FPS for each stage and total count.

## Visualization

Visualize tracker outputs on a video or MOT image sequence:

```bash
python visualize/tracker_visu.py \
    --sequence-dir dataset/blueberry_mot_stitched_walk/test/Blueberry-Test-10 \
    --reid-weights weights/osnet_ain_x1_0_blueberry.pt \
    --tracker-config-dir configs/tracker \
    --output-dir output/visualize/tracker/seq-10 \
    --device cuda
```

Visualize BerryTracker counting results:

```bash
python visualize/berrytracker_count_visu.py \
  --sequence-dir dataset/blueberry_mot_stitched_walk/test/Blueberry-Test-15 \
  --output-dir output/visualize/BerryTracker_Count \
  --trackers mytrack strongsort botsort \
  --yolo-weights weights/yolo11l.pt \
  --default-yolo-weights weights/yolo11s.pt \
  --reid-weights weights/osnet_ain_x1_0_blueberry.pt \
  --device cuda
```

## Count Real Videos

Apply BerryTracker counting to real videos listed in `dataset/video_path.json`:

```bash
python tools/count_apply.py \
  --video-path-json dataset/video_path.json \
  --yolo-weights weights/berrydet_s.pt \
  --reid-weights weights/osnet_ain_x1_0_blueberry.pt \
  --tracker-config configs/trackers/mytrack.yaml \
  --output-dir output/count_apply \
  --device cuda
```
or
```bash
INPUTS=(
  "/home/wh1234_/data/video/10s/20250427block8.mp4"
  "/home/wh1234_/data/video/count_apply/block3.mp4"
  "/home/wh1234_/data/video/count_apply/block4.mp4"
  "/home/wh1234_/data/blueberry_mot_stitched_walk/train/Blueberry-Train-10"
)
python tools/count_visu.py \
  --input "${INPUTS[@]}" \
  --output-dir output/count_visu \
  --yolo-weights weights/berrydet_s.pt \
  --reid-weights weights/osnet_ain_x1_0_blueberry.pt \
  --tracker-config configs/trackers/mytrack.yaml
```

## Outputs

Counting scripts write CSV files such as:

```text
output/count_apply/id_count.csv
output/count_apply/line_count.csv
output/count_apply/area_count.csv
```

Tracker evaluation writes summary and per-sequence metrics to:

```text
output/eval/<dataset_name>/tracker_evaluation_results.csv
output/eval/<dataset_name>/tracker_sequence_evaluation_results.csv
```

## Notes

- The blueberry datasets, trained detector weights, and ReID weights are provided through the Baidu Netdisk link above.
- Most scripts assume CUDA when available; pass `--device cpu` for CPU-only inference.
- `mytrack` is the BerryTracker implementation and uses `configs/trackers/mytrack.yaml`.