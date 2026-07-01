#!/usr/bin/env python3
# Purpose: visualize blueberry tracking results from multiple trackers frame by frame.

import argparse
import configparser
import logging
import sys
import time
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


LOGGER = logging.getLogger("tracker_visu")
VIDEO_PATH = REPO_ROOT / "dataset" / "video" / "20250425block4.mp4"
OUTPUT_DIR = REPO_ROOT / "output" / "visualize" / "tracker"
CONFIG_DIR = REPO_ROOT / "configs" / "trackers"
MYTRACK_WEIGHTS = REPO_ROOT / "weights" / "berrydet_s.pt"
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "yolo11s.pt"
DEFAULT_REID_WEIGHTS = REPO_ROOT / "weights" / "osnet_ain_x1_0_blueberry.pt"
# TRACKERS = ("bytetrack", "strongsort", "boosttrack", "hybridsort", "deepocsort", "mytrack")
TRACKERS = ("mytrack", "hybridsort")
REID_TRACKERS = {"strongsort", "boosttrack", "hybridsort", "deepocsort", "mytrack", "botsort"}
DISPLAY_NAMES = {
    "bytetrack": "ByteTrack",
    "strongsort": "StrongSORT",
    "botsort": "BotSort",
    "boosttrack": "BoostTrack",
    "hybridsort": "Hybrid-SORT",
    "deepocsort": "Deep_OC-SORT",
    "mytrack": "BerryTracker",
}
STAGE_NAMES = {0: "Flower", 1: "Green", 2: "Light Purple", 3: "Blue"}
STAGE_COLORS = {
    0: (0, 159, 230),
    1: (115, 158, 0),
    2: (167, 121, 204),
    3: (178, 114, 0),
}


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


def load_tracker_args(config_dir: Path, tracker_name: str) -> Dict[str, object]:
    config_path = config_dir / f"{tracker_name}.yaml"
    ensure_nonempty(config_path, "Tracker config")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return {key: value["default"] for key, value in config.items()}


def adapt_tracker_args(tracker_name: str, tracker_args: Dict[str, object]) -> Dict[str, object]:
    if tracker_name == "deepocsort":
        tracker_args["embedding_off"] = False
        tracker_args["aw_off"] = False
    elif tracker_name == "mytrack":
        tracker_args["embedding_off"] = False
        tracker_args["aw_off"] = False
        tracker_args["aarm_open"] = True
    elif tracker_name in {"boosttrack", "hybridsort"}:
        tracker_args["with_reid"] = True
        if tracker_name == "hybridsort":
            tracker_args["with_longterm_reid"] = True
            tracker_args["with_longterm_reid_correction"] = True
    return tracker_args


def build_tracker(
    tracker_name: str,
    config_dir: Path,
    reid_weights: Optional[Path],
    device: str,
    half: bool,
):
    tracker_args = adapt_tracker_args(tracker_name, load_tracker_args(config_dir, tracker_name))
    if tracker_name in REID_TRACKERS:
        if reid_weights is None:
            raise FileNotFoundError(f"{tracker_name} requires ReID weights. Pass --reid-weights.")
        ensure_nonempty(reid_weights, "ReID weights")

    return create_tracker(
        tracker_type=tracker_name,
        tracker_config=str(config_dir / f"{tracker_name}.yaml"),
        reid_weights=reid_weights if tracker_name in REID_TRACKERS else None,
        device=device,
        half=half,
        per_class=False,
        evolve_param_dict=tracker_args,
    )


def detector_weights_for(tracker_name: str) -> Path:
    return MYTRACK_WEIGHTS if tracker_name == "mytrack" else DEFAULT_WEIGHTS


def validate_assets(args: argparse.Namespace) -> None:
    if args.sequence_dir is not None:
        ensure_nonempty(args.sequence_dir, "Sequence directory")
        ensure_nonempty(resolve_img_dir(args.sequence_dir), "Sequence image directory")
    else:
        ensure_nonempty(args.video, "Video")
    for tracker_name in args.trackers:
        ensure_nonempty(args.tracker_config_dir / f"{tracker_name}.yaml", "Tracker config")
        ensure_nonempty(detector_weights_for(tracker_name), "YOLO weights")
    if any(tracker_name in REID_TRACKERS for tracker_name in args.trackers):
        if args.reid_weights is None:
            raise FileNotFoundError("Selected appearance trackers require ReID weights. Pass --reid-weights.")
        ensure_nonempty(args.reid_weights, "ReID weights")


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


def text_color(bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (0, 0, 0) if sum(bg_color) > 420 else (255, 255, 255)


def draw_label(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    x, y = origin
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, th + base + 2)
    cv2.rectangle(frame, (x, y - th - base - 4), (x + tw + 6, y + base + 3), color, -1)
    cv2.putText(frame, text, (x + 3, y), font, scale, text_color(color), thickness, cv2.LINE_AA)


def get_display_id(raw_id: int, display_id_map: Optional[Dict[int, int]]) -> int:
    if display_id_map is None:
        return raw_id
    return display_id_map.get(raw_id, raw_id)


def draw_tracks(frame: np.ndarray, tracks: np.ndarray, display_id_map: Optional[Dict[int, int]] = None) -> np.ndarray:
    normalized = normalize_tracks(tracks)
    if display_id_map is not None:
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


def write_frame(path: Path, frame: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])


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


def run_tracker_visualization(
    args: argparse.Namespace,
    tracker_name: str,
    video_info: Tuple[int, int, int],
) -> None:
    _, _, total_frames = video_info
    weights_path = detector_weights_for(tracker_name).resolve()
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
    tracker_dir = args.output_dir / DISPLAY_NAMES[tracker_name]
    total = min(total_frames, args.max_frames) if args.max_frames > 0 else total_frames
    start = time.perf_counter()
    detection_time = 0.0
    tracking_time = 0.0
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
        processed += 1

        write_frame(tracker_dir / f"frame_{frame_idx:06d}.jpg", draw_tracks(frame.copy(), tracks, display_id_map), args.jpg_quality)

        if frame_idx % args.log_interval == 0:
            LOGGER.info(
                "Progress | tracker=%s | frame=%d/%d | det=%d",
                tracker_name,
                frame_idx,
                total,
                len(detections),
            )

    LOGGER.info(
        "Tracker done | tracker=%s | frames=%d | detect_fps=%.2f | track_fps=%.2f | overall_fps=%.2f | output=%s",
        tracker_name,
        processed,
        processed / max(detection_time, 1e-9),
        processed / max(tracking_time, 1e-9),
        processed / max(time.perf_counter() - start, 1e-9),
        tracker_dir,
    )


def video_info(video_path: Path) -> Tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize blueberry tracking results from multiple BoxMOT trackers.")
    parser.add_argument("--video", type=Path, default=VIDEO_PATH, help="Input video. Ignored when --sequence-dir is set.")
    parser.add_argument("--sequence-dir", type=Path, default=None, help="Input MOT image sequence directory, e.g. .../test/Blueberry-Test-15.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output frame sequence directory.")
    parser.add_argument("--tracker-config-dir", type=Path, default=CONFIG_DIR, help="BoxMOT tracker config directory.")
    parser.add_argument("--reid-weights", type=Path, default=DEFAULT_REID_WEIGHTS, help="ReID weights for appearance trackers.")
    parser.add_argument("--trackers", nargs="+", default=list(TRACKERS), choices=list(TRACKERS), help="Trackers to visualize.")
    parser.add_argument("--conf", type=float, default=0.1, help="SAHI detection confidence threshold.")
    parser.add_argument("--slice-size", type=int, default=640, help="SAHI slice height and width.")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI slice overlap ratio.")
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
    args.reid_weights = args.reid_weights.resolve() if args.reid_weights else None
    ensure_nonempty(args.tracker_config_dir, "Tracker config directory")
    validate_assets(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sequence_dir is not None:
        args.frame_paths = load_frame_paths(args.sequence_dir)
        info = sequence_info(args.sequence_dir, args.frame_paths)
        LOGGER.info("Sequence | path=%s | size=%dx%d | frames=%d", args.sequence_dir, info[0], info[1], info[2])
    else:
        args.frame_paths = []
        info = video_info(args.video)
        LOGGER.info("Video | path=%s | size=%dx%d | frames=%d", args.video, info[0], info[1], info[2])

    for tracker_name in args.trackers:
        LOGGER.info("Start tracker visualization | tracker=%s", tracker_name)
        run_tracker_visualization(args, tracker_name, info)
    LOGGER.info("All visualizations saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
