#!/usr/bin/env python3
# Purpose: visualize BerryTracker counting results for ID, line, and center-area strategies.

import argparse
import configparser
import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.video_detector import VideoDetector
from visualize.tracker_visu import (
    CONFIG_DIR,
    DEFAULT_WEIGHTS,
    DEFAULT_REID_WEIGHTS,
    DISPLAY_NAMES,
    MYTRACK_WEIGHTS,
    OUTPUT_DIR,
    REID_TRACKERS,
    TRACKERS as TRACKER_CHOICES,
    VIDEO_PATH,
    build_tracker,
    draw_tracks,
    ensure_nonempty,
    normalize_tracks,
    setup_logging,
    stage_name,
    video_info,
    write_frame,
)


COUNTER_PATH = REPO_ROOT / "count" / "counter.py"
COUNTER_SPEC = importlib.util.spec_from_file_location("count_counter", COUNTER_PATH)
if COUNTER_SPEC is None or COUNTER_SPEC.loader is None:
    raise ImportError(f"Failed to load counter module: {COUNTER_PATH}")
COUNTER_MODULE = importlib.util.module_from_spec(COUNTER_SPEC)
COUNTER_SPEC.loader.exec_module(COUNTER_MODULE)
ObjectCounter = COUNTER_MODULE.ObjectCounter


LOGGER = logging.getLogger("berrytracker_count_visu")
# DEFAULT_TRACKERS = ("mytrack", "strongsort", "botsort")
DEFAULT_TRACKERS = ("mytrack", )
STAGES = ("Flower", "Green", "Light Purple", "Blue")


def counter_tracks(tracks: np.ndarray) -> np.ndarray:
    arr = normalize_tracks(tracks)
    if arr.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    return np.concatenate([arr[:, :4], arr[:, [4]], arr[:, [6]]], axis=1).astype(np.float32)


def stage_counts(count_map: Dict[str, int]) -> list[int]:
    return [int(count_map.get(stage, 0)) for stage in STAGES]


def resolve_count_geometry(frame_width: int, area_ratio: float) -> Tuple[int, int, int]:
    line_x = int(frame_width * 0.5)
    half_width = int(frame_width * area_ratio * 0.5)
    return line_x, max(0, line_x - half_width), min(frame_width - 1, line_x + half_width)


def draw_count_geometry(frame: np.ndarray, line_x: int, area_x1: int, area_x2: int) -> np.ndarray:
    overlay = frame.copy()
    area_color = (84, 130, 255)
    line_color = (30, 30, 30)
    cv2.rectangle(overlay, (area_x1, 0), (area_x2, frame.shape[0]), area_color, -1)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), line_color, 3, cv2.LINE_AA)
    cv2.rectangle(frame, (area_x1, 0), (area_x2, frame.shape[0] - 1), area_color, 2, cv2.LINE_AA)
    return frame


def draw_count_text(
    frame: np.ndarray,
    frame_idx: int,
    id_counts: Sequence[int],
    line_counts: Sequence[int],
    area_counts: Sequence[int],
    gt_counts: Optional[Sequence[int]] = None,
) -> np.ndarray:
    rows = []
    if gt_counts is not None:
        rows.append(("GT", gt_counts))
    rows.extend([("ID", id_counts), ("LINE", line_counts), ("AREA", area_counts)])

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.45
    thickness = 4
    margin = 36
    row_gap = 32
    row_height = cv2.getTextSize("LINE", font, scale, thickness)[0][1]
    col_x = [210, 470, 730, 1060]
    width = min(frame.shape[1] - 16, 1360)
    height = margin * 2 + (len(rows) + 2) * row_height + (len(rows) + 1) * row_gap
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)

    x0 = 8 + margin
    y = 8 + margin + row_height
    cv2.putText(frame, f"Frame: {frame_idx}", (x0, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    y += row_height + row_gap
    for label, x in zip(STAGES, col_x):
        cv2.putText(frame, label, (x0 + x, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    y += row_height + row_gap
    for label, counts in rows:
        cv2.putText(frame, label, (x0, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
        for value, x in zip(counts, col_x):
            cv2.putText(frame, str(int(value)), (x0 + x, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
        y += row_height + row_gap
    return frame


def validate_assets(args: argparse.Namespace) -> None:
    if args.sequence_dir is not None:
        ensure_nonempty(args.sequence_dir, "Sequence directory")
        ensure_nonempty(resolve_img_dir(args.sequence_dir), "Sequence image directory")
    else:
        ensure_nonempty(args.video, "Video")
    for tracker_name in args.trackers:
        ensure_nonempty(args.tracker_config_dir / f"{tracker_name}.yaml", "Tracker config")
        ensure_nonempty(detector_weights_for(tracker_name, args), "YOLO weights")
    if any(tracker_name in REID_TRACKERS for tracker_name in args.trackers):
        if args.reid_weights is None:
            raise FileNotFoundError("Selected appearance trackers require ReID weights. Pass --reid-weights.")
        ensure_nonempty(args.reid_weights, "ReID weights")


def detector_weights_for(tracker_name: str, args: argparse.Namespace) -> Path:
    return args.yolo_weights if tracker_name == "mytrack" else args.default_yolo_weights


def tracker_display_name(tracker_name: str) -> str:
    return DISPLAY_NAMES.get(tracker_name, tracker_name)


def resolve_img_dir(sequence_dir: Path) -> Path:
    img_dir = sequence_dir / "img1"
    return img_dir if img_dir.exists() else sequence_dir


def load_frame_paths(sequence_dir: Path) -> list[Path]:
    img_dir = resolve_img_dir(sequence_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frame_paths = sorted(path for path in img_dir.iterdir() if path.suffix.lower() in image_exts)
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in: {img_dir}")
    return frame_paths


def sequence_info(sequence_dir: Path, frame_paths: Sequence[Path]) -> Tuple[int, int, int]:
    seqinfo_path = sequence_dir / "seqinfo.ini"
    if seqinfo_path.exists():
        config = configparser.ConfigParser()
        config.read(seqinfo_path, encoding="utf-8")
        width = config.getint("Sequence", "imWidth", fallback=0)
        height = config.getint("Sequence", "imHeight", fallback=0)
        length = config.getint("Sequence", "seqLength", fallback=len(frame_paths))
        if width > 0 and height > 0:
            return width, height, min(length, len(frame_paths))

    frame = cv2.imread(str(frame_paths[0]))
    if frame is None:
        raise FileNotFoundError(f"Cannot read first frame: {frame_paths[0]}")
    height, width = frame.shape[:2]
    return width, height, len(frame_paths)


def load_gt_counts(sequence_dir: Path) -> list[int]:
    gt_path = sequence_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        LOGGER.warning("GT file not found; GT row will use zeros: %s", gt_path)
        return [0, 0, 0, 0]

    track_to_class: Dict[int, int] = {}
    with gt_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            parts = line.strip().split(",")
            if len(parts) < 8:
                LOGGER.warning("Skip malformed GT row: %s:%d", gt_path, line_no)
                continue
            mark = float(parts[6]) if parts[6] else 1.0
            visibility = float(parts[8]) if len(parts) > 8 and parts[8] else 1.0
            if mark <= 0 or visibility <= 0:
                continue
            track_id = int(float(parts[1]))
            class_id = int(float(parts[7]))
            if class_id < 0 or class_id >= len(STAGES):
                continue
            prev_class = track_to_class.get(track_id)
            if prev_class is not None and prev_class != class_id:
                LOGGER.warning(
                    "Inconsistent GT class assignment: seq=%s track_id=%d %d->%d",
                    sequence_dir.name,
                    track_id,
                    prev_class,
                    class_id,
                )
                continue
            track_to_class[track_id] = class_id

    gt_counts = [0, 0, 0, 0]
    for class_id in track_to_class.values():
        gt_counts[class_id] += 1
    return gt_counts


def iter_frames(args: argparse.Namespace, total: int) -> Iterator[Tuple[int, np.ndarray]]:
    if args.sequence_dir is not None:
        for frame_idx, img_path in enumerate(args.frame_paths[:total], start=1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                LOGGER.warning("Skip unreadable frame: %s", img_path)
                continue
            yield frame_idx, frame
        return

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    try:
        for frame_idx in range(1, total + 1):
            ok, frame = cap.read()
            if not ok:
                break
            yield frame_idx, frame
    finally:
        cap.release()


def run_count_visualization(args: argparse.Namespace, tracker_name: str, info: Tuple[int, int, int]) -> None:
    width, _, total_frames = info
    weights_path = detector_weights_for(tracker_name, args).resolve()
    tracker_device = "0" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    LOGGER.info("Loading detector | tracker=%s | weights=%s", tracker_name, weights_path)
    detector = VideoDetector(
        yolo_weights_path=str(weights_path),
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
        conf_threshold=args.conf,
        device=args.device,
    )
    tracker = build_tracker(tracker_name, args.tracker_config_dir, args.reid_weights, tracker_device, args.half)
    line_x, area_x1, area_x2 = resolve_count_geometry(width, args.area_ratio)
    counter = ObjectCounter(line_x=line_x, area_x1=area_x1, area_x2=area_x2, label_resolver=stage_name)
    tracker_dir = args.output_dir / tracker_display_name(tracker_name)
    total = min(total_frames, args.max_frames) if args.max_frames > 0 else total_frames

    start = time.perf_counter()
    detection_time = tracking_time = counting_time = 0.0
    processed = 0
    display_id_map: Dict[int, int] = {}

    for frame_idx, frame in iter_frames(args, total):
        detect_start = time.perf_counter()
        detections = np.asarray(detector.detect_frame_with_sahi(frame), dtype=np.float32)
        detection_time += time.perf_counter() - detect_start
        if detections.size == 0:
            detections = np.empty((0, 6), dtype=np.float32)

        track_start = time.perf_counter()
        tracks = tracker.update(detections, frame)
        tracking_time += time.perf_counter() - track_start

        count_start = time.perf_counter()
        counter.update(counter_tracks(tracks), frame.shape[:2])
        id_count, line_count, area_count = counter.results(as_counts=True)
        counting_time += time.perf_counter() - count_start

        gt_counts = args.gt_counts if args.gt_counts is not None else None
        count_frame = draw_tracks(frame.copy(), tracks, display_id_map)
        count_frame = draw_count_geometry(count_frame, line_x, area_x1, area_x2)
        count_frame = draw_count_text(
            count_frame,
            frame_idx,
            stage_counts(id_count),
            stage_counts(line_count),
            stage_counts(area_count),
            gt_counts,
        )
        write_frame(tracker_dir / f"frame_{frame_idx:06d}.jpg", count_frame, args.jpg_quality)
        processed += 1

        if frame_idx % args.log_interval == 0:
            LOGGER.info("Progress | tracker=%s | frame=%d/%d | det=%d", tracker_name, frame_idx, total, len(detections))

    LOGGER.info(
        "Count visualization done | tracker=%s | frames=%d | detect_fps=%.2f | track_fps=%.2f | count_fps=%.2f | overall_fps=%.2f | output=%s",
        tracker_name,
        processed,
        processed / max(detection_time, 1e-9),
        processed / max(tracking_time, 1e-9),
        processed / max(counting_time, 1e-9),
        processed / max(time.perf_counter() - start, 1e-9),
        tracker_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize BerryTracker ID, line, and area counting results.")
    parser.add_argument("--video", type=Path, default=VIDEO_PATH, help="Input video. Ignored when --sequence-dir is set.")
    parser.add_argument("--sequence-dir", type=Path, default=None, help="Input MOT image sequence directory, e.g. .../test/Blueberry-Test-15.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "BerryTracker_Count", help="Output frame sequence directory.")
    parser.add_argument("--tracker-config-dir", type=Path, default=CONFIG_DIR, help="BoxMOT tracker config directory.")
    parser.add_argument("--trackers", nargs="+", default=list(DEFAULT_TRACKERS), choices=list(TRACKER_CHOICES), help="Trackers to visualize.")
    parser.add_argument("--yolo-weights", type=Path, default=MYTRACK_WEIGHTS, help="BerryTracker detector weights.")
    parser.add_argument("--default-yolo-weights", type=Path, default=DEFAULT_WEIGHTS, help="Detector weights for StrongSORT and BoTSORT.")
    parser.add_argument("--reid-weights", type=Path, default=DEFAULT_REID_WEIGHTS, help="ReID weights for appearance trackers.")
    parser.add_argument("--conf", type=float, default=0.1, help="SAHI detection confidence threshold.")
    parser.add_argument("--slice-size", type=int, default=640, help="SAHI slice height and width.")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI slice overlap ratio.")
    parser.add_argument("--area-ratio", type=float, default=0.08, help="Center counting area width ratio.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Detector device.")
    parser.add_argument("--half", action="store_true", help="Use half precision for ReID backends.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames for quick checks; 0 means all frames.")
    parser.add_argument("--jpg-quality", type=int, default=95, help="Output JPEG quality.")
    parser.add_argument("--log-interval", type=int, default=50, help="Progress log interval in frames.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    args.sequence_dir = args.sequence_dir.resolve() if args.sequence_dir else None
    args.video = args.video.resolve() if args.sequence_dir is None else args.video
    args.output_dir = args.output_dir.resolve()
    args.tracker_config_dir = args.tracker_config_dir.resolve()
    args.yolo_weights = args.yolo_weights.resolve()
    args.default_yolo_weights = args.default_yolo_weights.resolve()
    args.reid_weights = args.reid_weights.resolve() if args.reid_weights else None
    validate_assets(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sequence_dir is not None:
        args.frame_paths = load_frame_paths(args.sequence_dir)
        args.gt_counts = load_gt_counts(args.sequence_dir)
        info = sequence_info(args.sequence_dir, args.frame_paths)
        LOGGER.info("Sequence | path=%s | size=%dx%d | frames=%d", args.sequence_dir, info[0], info[1], info[2])
    else:
        args.frame_paths = []
        args.gt_counts = None
        info = video_info(args.video)
        LOGGER.info("Video | path=%s | size=%dx%d | frames=%d", args.video, info[0], info[1], info[2])

    LOGGER.info(
        "Counter geometry | line_x=%d | area_x1=%d | area_x2=%d",
        *resolve_count_geometry(info[0], args.area_ratio),
    )
    for tracker_name in args.trackers:
        LOGGER.info("Start count visualization | tracker=%s", tracker_name)
        run_count_visualization(args, tracker_name, info)
    LOGGER.info("All count visualizations saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
