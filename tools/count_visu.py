#!/usr/bin/env python3
# Purpose: visualize BerryTracker AREA counting on videos or MOT image sequences.

import argparse
import configparser
import importlib.util
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

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


COUNTER_PATH = REPO_ROOT / "count" / "00-counter.py"
COUNTER_SPEC = importlib.util.spec_from_file_location("count_00_counter", COUNTER_PATH)
if COUNTER_SPEC is None or COUNTER_SPEC.loader is None:
    raise ImportError(f"Failed to load counter module: {COUNTER_PATH}")
COUNTER_MODULE = importlib.util.module_from_spec(COUNTER_SPEC)
COUNTER_SPEC.loader.exec_module(COUNTER_MODULE)
ObjectCounter = COUNTER_MODULE.ObjectCounter


LOGGER = logging.getLogger("count_visu")
TRACKER_NAME = "mytrack"
DISPLAY_NAME = "BerryTracker"
STAGES = ("Flower", "Green", "Light Purple", "Blue")
STAGE_NAMES = {0: "Flower", 1: "Green", 2: "Light Purple", 3: "Blue"}
STAGE_COLORS = {
    0: (0, 159, 230),
    1: (115, 158, 0),
    2: (167, 121, 204),
    3: (178, 114, 0),
}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_INPUTS = (
    Path("/home/wh1234_/data/video/10s/20250427block8.mp4"),
    Path("/home/wh1234_/data/video/count_apply/block1.mp4"),
    Path("/home/wh1234_/data/video/count_apply/block4.mp4"),
    Path("/home/wh1234_/data/blueberry_mot_stitched_walk/train/Blueberry-Train-10"),
)
DEFAULT_REID_WEIGHTS = REPO_ROOT / "weights" / "osnet_ain_x1_0_blueberry.pt"


@dataclass(frozen=True)
class SourceInfo:
    width: int
    height: int
    total_frames: int
    fps: float


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def stage_name(class_id: int) -> str:
    return STAGE_NAMES.get(int(class_id), f"Class_{int(class_id)}")


def ensure_nonempty(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.is_file() and path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is empty: {path}")


def load_tracker_args(config_path: Path) -> Dict[str, object]:
    ensure_nonempty(config_path, "Tracker config")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    tracker_args = {key: value["default"] for key, value in config.items()}
    tracker_args["embedding_off"] = False
    tracker_args["aw_off"] = False
    tracker_args["aarm_open"] = True
    tracker_args["cmc_off"] = True
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


def normalize_tracks(tracks: np.ndarray) -> np.ndarray:
    if tracks is None:
        return np.empty((0, 7), dtype=np.float32)
    arr = np.asarray(tracks, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    conf = arr[:, [5]] if arr.shape[1] > 5 else np.ones((arr.shape[0], 1), dtype=np.float32)
    cls = arr[:, [6]] if arr.shape[1] > 6 else np.zeros((arr.shape[0], 1), dtype=np.float32)
    return np.concatenate([arr[:, :5], conf, cls], axis=1).astype(np.float32)


def counter_tracks(tracks: np.ndarray) -> np.ndarray:
    arr = normalize_tracks(tracks)
    if arr.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    return np.concatenate([arr[:, :4], arr[:, [4]], arr[:, [6]]], axis=1).astype(np.float32)


def stage_counts(count_map: Dict[str, int]) -> list[int]:
    return [int(count_map.get(stage, 0)) for stage in STAGES]


def text_color(bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (0, 0, 0) if sum(bg_color) > 420 else (255, 255, 255)


def draw_label(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.42, min(0.6, frame.shape[1] / 1800.0))
    thickness = max(1, int(round(scale * 3)))
    x, y = origin
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, th + base + 2)
    cv2.rectangle(frame, (x, y - th - base - 4), (x + tw + 6, y + base + 3), color, -1)
    cv2.putText(frame, text, (x + 3, y), font, scale, text_color(color), thickness, cv2.LINE_AA)


def get_display_id(raw_id: int, display_id_map: Dict[int, int]) -> int:
    return display_id_map.get(raw_id, raw_id)


def draw_tracks(frame: np.ndarray, tracks: np.ndarray, display_id_map: Dict[int, int]) -> np.ndarray:
    normalized = normalize_tracks(tracks)
    new_tracks: Dict[int, Tuple[float, float]] = {}
    for x1, y1, x2, y2, track_id, _, _ in normalized:
        raw_id = int(track_id)
        if raw_id not in display_id_map and raw_id not in new_tracks:
            new_tracks[raw_id] = ((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5)
    for raw_id, _ in sorted(new_tracks.items(), key=lambda item: (item[1][0], item[1][1], item[0])):
        display_id_map[raw_id] = len(display_id_map) + 1

    for x1, y1, x2, y2, track_id, conf, cls_id in normalized:
        raw_id = int(track_id)
        class_id = int(cls_id)
        color = STAGE_COLORS.get(class_id, (150, 150, 150))
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(frame, p1, p2, color, 2, cv2.LINE_AA)
        draw_label(frame, f"{get_display_id(raw_id, display_id_map)} {stage_name(class_id)} {float(conf):.2f}", (p1[0], p1[1] - 6), color)
    return frame


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
    rows.extend((("ID", id_counts), ("LINE", line_counts), ("AREA", area_counts)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.48, min(1.15, frame.shape[1] / 1500.0))
    thickness = max(1, int(round(scale * 2.7)))
    margin = max(12, int(round(24 * scale)))
    row_gap = max(10, int(round(22 * scale)))
    row_height = cv2.getTextSize("Light Purple", font, scale, thickness)[0][1]
    panel_width = min(frame.shape[1] - 16, max(620, int(frame.shape[1] * 0.72)))
    panel_height = margin * 2 + (len(rows) + 2) * row_height + (len(rows) + 1) * row_gap

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)

    x0 = 8 + margin
    y = 8 + margin + row_height
    title = f"{DISPLAY_NAME} AREA  Frame: {frame_idx}"
    cv2.putText(frame, title, (x0, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    y += row_height + row_gap

    label_col = max(92, int(round(128 * scale)))
    usable_width = max(360, panel_width - margin * 2 - label_col)
    col_step = usable_width / len(STAGES)
    col_x = [x0 + label_col + int(round(i * col_step)) for i in range(len(STAGES))]

    for label, x in zip(STAGES, col_x):
        cv2.putText(frame, label, (x, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    y += row_height + row_gap
    for label, counts in rows:
        cv2.putText(frame, label, (x0, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
        for value, x in zip(counts, col_x):
            cv2.putText(frame, str(int(value)), (x, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
        y += row_height + row_gap
    return frame


def resolve_img_dir(sequence_dir: Path) -> Path:
    img_dir = sequence_dir / "img1"
    return img_dir if img_dir.exists() else sequence_dir


def load_frame_paths(sequence_dir: Path) -> list[Path]:
    img_dir = resolve_img_dir(sequence_dir)
    frame_paths = sorted(path for path in img_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in: {img_dir}")
    return frame_paths


def sequence_info(sequence_dir: Path, frame_paths: Sequence[Path]) -> SourceInfo:
    seqinfo_path = sequence_dir / "seqinfo.ini"
    if seqinfo_path.exists():
        config = configparser.ConfigParser()
        config.read(seqinfo_path, encoding="utf-8")
        width = config.getint("Sequence", "imWidth", fallback=0)
        height = config.getint("Sequence", "imHeight", fallback=0)
        length = config.getint("Sequence", "seqLength", fallback=len(frame_paths))
        fps = config.getfloat("Sequence", "frameRate", fallback=25.0)
        if width > 0 and height > 0:
            return SourceInfo(width, height, min(length, len(frame_paths)), fps if fps > 0 else 25.0)

    frame = cv2.imread(str(frame_paths[0]))
    if frame is None:
        raise FileNotFoundError(f"Cannot read first frame: {frame_paths[0]}")
    height, width = frame.shape[:2]
    return SourceInfo(width, height, len(frame_paths), 25.0)


def video_info(video_path: Path) -> SourceInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return SourceInfo(width, height, frames, fps if fps > 0 else 25.0)


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
            previous_class = track_to_class.get(track_id)
            if previous_class is not None and previous_class != class_id:
                LOGGER.warning(
                    "Inconsistent GT class assignment: seq=%s track_id=%d %d->%d",
                    sequence_dir.name,
                    track_id,
                    previous_class,
                    class_id,
                )
                continue
            track_to_class[track_id] = class_id

    gt_counts = [0, 0, 0, 0]
    for class_id in track_to_class.values():
        gt_counts[class_id] += 1
    return gt_counts


def iter_video_frames(video_path: Path, total: int) -> Iterator[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        for frame_idx in range(1, total + 1):
            ok, frame = cap.read()
            if not ok:
                break
            yield frame_idx, frame
    finally:
        cap.release()


def iter_sequence_frames(frame_paths: Sequence[Path], total: int) -> Iterator[Tuple[int, np.ndarray]]:
    for frame_idx, img_path in enumerate(frame_paths[:total], start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            LOGGER.warning("Skip unreadable frame: %s", img_path)
            continue
        yield frame_idx, frame


def input_kind(input_path: Path) -> str:
    if input_path.is_dir():
        return "sequence"
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS:
        return "video"
    if input_path.suffix.lower() in VIDEO_EXTS:
        return "video"
    raise ValueError(f"Input must be a video file or MOT sequence directory: {input_path}")


def safe_stem(input_path: Path) -> str:
    stem = input_path.name if input_path.is_dir() else input_path.stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return stem or "input"


def make_output_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{safe_stem(input_path)}_berrytracker_area.mp4"


def open_video_writer(output_path: Path, info: SourceInfo, codec: str) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, info.fps, (info.width, info.height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")
    return writer


def validate_assets(args: argparse.Namespace) -> None:
    ensure_nonempty(args.yolo_weights, "YOLO weights")
    ensure_nonempty(args.tracker_config, "Tracker config")
    ensure_nonempty(args.reid_weights, "ReID weights")
    for input_path in args.input:
        ensure_nonempty(input_path, "Input")


def process_input(input_path: Path, args: argparse.Namespace, detector: VideoDetector) -> Path:
    kind = input_kind(input_path)
    if kind == "sequence":
        frame_paths = load_frame_paths(input_path)
        info = sequence_info(input_path, frame_paths)
        gt_counts: Optional[list[int]] = load_gt_counts(input_path)
        frame_iter = lambda total: iter_sequence_frames(frame_paths, total)
        LOGGER.info("Sequence | path=%s | size=%dx%d | frames=%d | fps=%.2f", input_path, info.width, info.height, info.total_frames, info.fps)
    else:
        frame_paths = []
        info = video_info(input_path)
        gt_counts = None
        frame_iter = lambda total: iter_video_frames(input_path, total)
        LOGGER.info("Video | path=%s | size=%dx%d | frames=%d | fps=%.2f", input_path, info.width, info.height, info.total_frames, info.fps)

    if not frame_paths and kind == "sequence":
        raise FileNotFoundError(f"No sequence frames found: {input_path}")

    total = min(info.total_frames, args.max_frames) if args.max_frames > 0 else info.total_frames
    line_x, area_x1, area_x2 = resolve_count_geometry(info.width, args.area_ratio)
    counter = ObjectCounter(line_x=line_x, area_x1=area_x1, area_x2=area_x2, label_resolver=stage_name)
    tracker = build_tracker(args.tracker_config, args.reid_weights, args.tracker_device, args.half)
    output_path = make_output_path(input_path, args.output_dir)
    writer = open_video_writer(output_path, info, args.codec)

    LOGGER.info("Counter geometry | line_x=%d | area_x1=%d | area_x2=%d", line_x, area_x1, area_x2)
    start = time.perf_counter()
    detection_time = tracking_time = counting_time = 0.0
    processed = 0
    display_id_map: Dict[int, int] = {}

    try:
        for frame_idx, frame in frame_iter(total):
            if frame.shape[1] != info.width or frame.shape[0] != info.height:
                frame = cv2.resize(frame, (info.width, info.height), interpolation=cv2.INTER_LINEAR)

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

            output_frame = draw_tracks(frame.copy(), tracks, display_id_map)
            output_frame = draw_count_geometry(output_frame, line_x, area_x1, area_x2)
            output_frame = draw_count_text(
                output_frame,
                frame_idx,
                stage_counts(id_count),
                stage_counts(line_count),
                stage_counts(area_count),
                gt_counts,
            )
            writer.write(output_frame)
            processed += 1

            if args.log_interval > 0 and frame_idx % args.log_interval == 0:
                LOGGER.info("Progress | input=%s | frame=%d/%d | det=%d", input_path.name, frame_idx, total, len(detections))
    finally:
        writer.release()

    elapsed = time.perf_counter() - start
    LOGGER.info(
        "Visualization done | input=%s | frames=%d | detect_fps=%.2f | track_fps=%.2f | count_fps=%.2f | overall_fps=%.2f | output=%s",
        input_path,
        processed,
        processed / max(detection_time, 1e-9),
        processed / max(tracking_time, 1e-9),
        processed / max(counting_time, 1e-9),
        processed / max(elapsed, 1e-9),
        output_path,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize BerryTracker AREA counting on videos or MOT image sequences.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[DEFAULT_INPUTS[0]],
        help="Input video file(s) or MOT sequence directory/directories.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "count_visu", help="Output video directory.")
    parser.add_argument("--yolo-weights", type=Path, default=REPO_ROOT / "weights" / "berrydet_s.pt")
    parser.add_argument("--tracker-config", type=Path, default=REPO_ROOT / "configs" / "trackers" / "mytrack.yaml")
    parser.add_argument("--reid-weights", type=Path, default=DEFAULT_REID_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--area-ratio", type=float, default=0.08)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--codec", type=str, default="mp4v", help="OpenCV fourcc codec, e.g. mp4v or avc1.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames for quick checks; 0 means all frames.")
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    args.input = [path.resolve() for path in args.input]
    args.output_dir = args.output_dir.resolve()
    args.yolo_weights = args.yolo_weights.resolve()
    args.tracker_config = args.tracker_config.resolve()
    args.reid_weights = args.reid_weights.resolve()
    args.tracker_device = "0" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    validate_assets(args)

    LOGGER.info("Loading detector | weights=%s", args.yolo_weights)
    detector = VideoDetector(
        yolo_weights_path=str(args.yolo_weights),
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
        conf_threshold=args.conf,
        device=args.device,
    )

    output_paths = []
    for input_path in args.input:
        output_paths.append(process_input(input_path, args, detector))
    LOGGER.info("Saved %d visualization video(s) to %s", len(output_paths), args.output_dir)


if __name__ == "__main__":
    main()
