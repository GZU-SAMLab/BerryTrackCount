#!/usr/bin/env python3
# Purpose: apply BerryTracker counting on real blueberry videos with ID, line, and area strategies.

import argparse
import csv
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(REPO_ROOT / "trackers") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "trackers"))

from loguru import logger as loguru_logger

loguru_logger.disable("boxmot.motion.cmc.ecc")

from boxmot.tracker_zoo import create_tracker
from tools.video_detector import VideoDetector


COUNTER_PATH = REPO_ROOT / "count" / "counter.py"
COUNTER_SPEC = importlib.util.spec_from_file_location("count_counter", COUNTER_PATH)
if COUNTER_SPEC is None or COUNTER_SPEC.loader is None:
    raise ImportError(f"Failed to load counter module: {COUNTER_PATH}")
COUNTER_MODULE = importlib.util.module_from_spec(COUNTER_SPEC)
COUNTER_SPEC.loader.exec_module(COUNTER_MODULE)
ObjectCounter = COUNTER_MODULE.ObjectCounter


LOGGER = logging.getLogger("count_apply")
TRACKER_NAME = "mytrack"
STAGES = ("Flower", "Green", "Light Purple", "Blue")
CSV_HEADER = ("video_name", *STAGES, "total")
STAGE_NAMES = {0: "Flower", 1: "Green", 2: "Light Purple", 3: "Blue"}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def stage_name(class_id: int) -> str:
    return STAGE_NAMES.get(int(class_id), f"Class_{int(class_id)}")


def load_video_paths(path: Path) -> List[Path]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [Path(video_path) for _, paths in sorted(data.items()) for video_path in paths]


def load_tracker_args(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    tracker_args = {key: value["default"] for key, value in config.items()}
    tracker_args["embedding_off"] = False
    tracker_args["aw_off"] = False
    tracker_args["aarm_open"] = True
    return tracker_args


def build_tracker(config_path: Path, reid_weights: Path, device: str, half: bool):
    return create_tracker(
        tracker_type=TRACKER_NAME,
        tracker_config=str(config_path),
        reid_weights=reid_weights,
        device=device,
        half=half,
        per_class=False,
        evolve_param_dict=load_tracker_args(config_path),
    )


def counter_tracks(tracks: np.ndarray) -> np.ndarray:
    if tracks is None:
        return np.empty((0, 6), dtype=np.float32)
    arr = np.asarray(tracks, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    class_ids = arr[:, [6]] if arr.shape[1] > 6 else np.zeros((arr.shape[0], 1), dtype=np.float32)
    return np.concatenate([arr[:, :4], arr[:, [4]], class_ids], axis=1).astype(np.float32)


def count_geometry(frame_width: int, area_ratio: float) -> Tuple[int, int, int]:
    line_x = int(frame_width * 0.5)
    half_width = int(frame_width * area_ratio * 0.5)
    return line_x, max(0, line_x - half_width), min(frame_width - 1, line_x + half_width)


def empty_counts() -> Dict[str, int]:
    return {stage: 0 for stage in STAGES}


def count_video(
    video_path: Path,
    detector: VideoDetector,
    config_path: Path,
    reid_weights: Path,
    tracker_device: str,
    half: bool,
    area_ratio: float,
    log_interval: int,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = build_tracker(config_path, reid_weights, tracker_device, half)
    counter = None
    processed = 0
    start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed += 1

            if counter is None:
                line_x, area_x1, area_x2 = count_geometry(frame.shape[1], area_ratio)
                counter = ObjectCounter(line_x=line_x, area_x1=area_x1, area_x2=area_x2, label_resolver=stage_name)
                LOGGER.info(
                    "Counter geometry | video=%s | line_x=%d | area_x1=%d | area_x2=%d",
                    video_path.name,
                    line_x,
                    area_x1,
                    area_x2,
                )

            detections = np.asarray(detector.detect_frame_with_sahi(frame), dtype=np.float32)
            if detections.size == 0:
                detections = np.empty((0, 6), dtype=np.float32)
            tracks = tracker.update(detections, frame)
            counter.update(counter_tracks(tracks), frame.shape[:2])

            if log_interval > 0 and processed % log_interval == 0:
                LOGGER.info("Progress | video=%s | frame=%d/%d", video_path.name, processed, total_frames)
    finally:
        cap.release()

    if counter is None:
        LOGGER.warning("No frames processed | video=%s", video_path)
        return empty_counts(), empty_counts(), empty_counts()

    LOGGER.info(
        "Video done | video=%s | frames=%d | fps=%.2f",
        video_path.name,
        processed,
        processed / max(time.perf_counter() - start, 1e-9),
    )
    return counter.results(as_counts=True)


def row(video_name: str, counts: Dict[str, int]) -> List[object]:
    values = [int(counts.get(stage, 0)) for stage in STAGES]
    return [video_name, *values, sum(values)]


def write_csv(path: Path, rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def validate_inputs(args: argparse.Namespace) -> None:
    for label, path in (
        ("video path json", args.video_path_json),
        ("detector weights", args.yolo_weights),
        ("tracker config", args.tracker_config),
        ("ReID weights", args.reid_weights),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply BerryTracker counting on real videos.")
    parser.add_argument("--video-path-json", type=Path, default=REPO_ROOT / "dataset" / "video_path.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "count_apply")
    parser.add_argument("--yolo-weights", type=Path, default=REPO_ROOT / "weights" / "berrydet_s.pt")
    parser.add_argument("--tracker-config", type=Path, default=REPO_ROOT / "configs" / "trackers" /"mytrack.yaml")
    parser.add_argument("--reid-weights", type=Path, default=REPO_ROOT / "weights" / "osnet_ain_x1_0_blueberry.pt")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--area-ratio", type=float, default=0.08)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    validate_inputs(args)

    video_paths = load_video_paths(args.video_path_json)
    LOGGER.info("Loaded videos | count=%d | source=%s", len(video_paths), args.video_path_json)
    detector = VideoDetector(
        yolo_weights_path=str(args.yolo_weights),
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
        conf_threshold=args.conf,
        device=args.device,
    )
    tracker_device = "0" if torch.cuda.is_available() and args.device != "cpu" else "cpu"

    id_rows, line_rows, area_rows = [], [], []
    for index, video_path in enumerate(video_paths, start=1):
        LOGGER.info("Start video | index=%d/%d | path=%s", index, len(video_paths), video_path)
        id_count, line_count, area_count = count_video(
            video_path=video_path,
            detector=detector,
            config_path=args.tracker_config,
            reid_weights=args.reid_weights,
            tracker_device=tracker_device,
            half=args.half,
            area_ratio=args.area_ratio,
            log_interval=args.log_interval,
        )
        video_name = video_path.stem
        id_rows.append(row(video_name, id_count))
        line_rows.append(row(video_name, line_count))
        area_rows.append(row(video_name, area_count))

    write_csv(args.output_dir / "id_count.csv", id_rows)
    write_csv(args.output_dir / "line_count.csv", line_rows)
    write_csv(args.output_dir / "area_count.csv", area_rows)
    LOGGER.info("Saved results | output_dir=%s", args.output_dir)


if __name__ == "__main__":
    main()
