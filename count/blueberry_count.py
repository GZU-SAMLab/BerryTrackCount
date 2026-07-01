#!/usr/bin/env python3
"""Count blueberry stages on image sequences with multiple BoxMOT trackers and counter strategies."""

import argparse
import csv
import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


COUNTER_MODULE_PATH = REPO_ROOT / "count" / "counter.py"
COUNTER_SPEC = importlib.util.spec_from_file_location("count_counter", COUNTER_MODULE_PATH)
if COUNTER_SPEC is None or COUNTER_SPEC.loader is None:
    raise ImportError(f"Failed to load counter module: {COUNTER_MODULE_PATH}")
COUNTER_MODULE = importlib.util.module_from_spec(COUNTER_SPEC)
COUNTER_SPEC.loader.exec_module(COUNTER_MODULE)
ObjectCounter = COUNTER_MODULE.ObjectCounter


LOGGER = logging.getLogger("blueberry_count")
TRACKERS = ["bytetrack", "strongsort", "hybridsort", "boosttrack", "ocsort", "deepocsort","mytrack", "botsort"]
REID_TRACKERS = {"strongsort", "hybridsort", "boosttrack", "deepocsort","mytrack", "botsort"}
APPEARANCE_TRACKERS = ["strongsort", "hybridsort", "boosttrack", "deepocsort","mytrack", "botsort"]
STAGES = ["Flower", "Green", "Light Purple", "Blue"]
CSV_HEADER = ["video_name", "tracker", "fps", "overall_fps", *STAGES, "total"]
MYTRACK_YOLO_WEIGHTS = REPO_ROOT / "weights" / "berrydet_s.pt"
DEFAULT_YOLO_WEIGHTS = REPO_ROOT / "weights" / "yolo11s.pt"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_blueberry_stage_name(class_id: int) -> str:
    mapping = {0: "Flower", 1: "Green", 2: "Light Purple", 3: "Blue"}
    return mapping.get(int(class_id), f"Unknown_{class_id}")


def get_tracker_config_path(config_dir: Path, tracker_name: str) -> Path:
    config_name = "mytrack" if tracker_name == "mytrack_botsort" else tracker_name
    return config_dir / f"{config_name}.yaml"


def load_tracker_args(config_dir: Path, tracker_name: str) -> Dict[str, object]:
    config_path = get_tracker_config_path(config_dir, tracker_name)
    if not config_path.exists():
        raise FileNotFoundError(f"Tracker config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return {key: value["default"] for key, value in config.items()}


def resolve_reid_weights(reid_path: Optional[Path]) -> Optional[Path]:
    if reid_path is not None:
        return reid_path if reid_path.exists() else None
    default_path = REPO_ROOT / "weights" / "osnet_ain_x1_0_blueberry.pt"
    return default_path if default_path.exists() else None


def adapt_tracker_args(
    tracker_name: str,
    tracker_args: Dict[str, object],
    reid_weights: Optional[Path],
    use_appearance: bool,
) -> Tuple[Dict[str, object], Optional[Path]]:
    if use_appearance and tracker_name in REID_TRACKERS and reid_weights is None:
        raise FileNotFoundError(f"{tracker_name} appearance mode requires ReID weights, but none were found.")

    if tracker_name == "deepocsort":
        tracker_args["embedding_off"] = True
        tracker_args["aw_off"] = True
    elif tracker_name in {"botsort", "boosttrack", "hybridsort"}:
        tracker_args["with_reid"] = use_appearance
        if tracker_name == "hybridsort":
            tracker_args["with_longterm_reid"] = use_appearance
            tracker_args["with_longterm_reid_correction"] = use_appearance
    elif tracker_name == "mytrack_botsort":
        tracker_args["with_reid"] = use_appearance
        tracker_args["asso_func"] = "hciou"
        tracker_args["aarm_open"] = True
    elif tracker_name == "mytrack":
        tracker_args["asso_func"] = "hciou"
        tracker_args["embedding_off"] = False
        tracker_args["aw_off"] = False
        tracker_args["aarm_open"] = True
    elif tracker_name == "strongsort":
        if use_appearance:
            pass
        elif reid_weights is None:
            raise FileNotFoundError("StrongSort requires ReID weights, but no usable weights were found.")
    return tracker_args, reid_weights


def build_counter_tracks(tracks: np.ndarray) -> np.ndarray:
    if tracks is None:
        return np.empty((0, 6), dtype=np.float32)
    arr = np.asarray(tracks, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    class_ids = arr[:, [6]] if arr.shape[1] > 6 else np.zeros((arr.shape[0], 1), dtype=np.float32)
    return np.concatenate([arr[:, :4], arr[:, [4]], class_ids], axis=1).astype(np.float32)


def build_count_row(video_name: str, tracker: str, count_map: Dict[str, int], fps: float, overall_fps: float) -> List[object]:
    stage_counts = [int(count_map.get(stage, 0)) for stage in STAGES]
    return [video_name, tracker, f"{fps:.2f}", f"{overall_fps:.2f}", *stage_counts, sum(stage_counts)]


def load_existing_keys(path: Path) -> set[Tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[Tuple[str, str]] = set()
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                keys.add((row[0], row[1]))
    return keys


def append_count_row(path: Path, row: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)


class BlueberryCounterRunner:
    def __init__(
        self,
        data_root: Path,
        output_dir: Path,
        yolo_weights: Path,
        reid_path: Optional[Path],
        tracker_config_dir: Path,
        trackers: Sequence[str],
        appearance_trackers: Sequence[str],
        conf: float,
        slice_size: int,
        overlap: float,
        area_ratio: float,
        device: str,
        half: bool,
    ) -> None:
        self.data_root = data_root
        self.output_dir = output_dir
        self.yolo_weights = yolo_weights
        self.reid_path = reid_path
        self.trackers = list(trackers)
        self.appearance_trackers = set(appearance_trackers)
        self.conf = conf
        self.slice_size = slice_size
        self.overlap = overlap
        self.area_ratio = area_ratio
        self.detector_device = device
        self.tracker_device = "0" if torch.cuda.is_available() and device != "cpu" else "cpu"
        self.half = half
        self.test_dir = self.data_root / "test"
        self.config_dir = tracker_config_dir
        self.reid_weights = resolve_reid_weights(self.reid_path)
        self.detectors: Dict[Path, VideoDetector] = {}

        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Tracker config directory not found: {self.config_dir}")
        self.sequences = sorted([seq for seq in self.test_dir.iterdir() if seq.is_dir()])
        LOGGER.info("Appearance-enabled trackers: %s", sorted(self.appearance_trackers) if self.appearance_trackers else [])

    @staticmethod
    def resolve_counter_geometry(frame_width: int, area_ratio: float) -> Tuple[int, int, int]:
        line_x = int(frame_width * 0.5)
        half_width = int(frame_width * area_ratio * 0.5)
        area_x1 = max(0, line_x - half_width)
        area_x2 = min(frame_width, line_x + half_width)
        return line_x, area_x1, area_x2

    def build_tracker(self, tracker_name: str):
        tracker_args = load_tracker_args(self.config_dir, tracker_name)
        tracker_args, reid_weights = adapt_tracker_args(
            tracker_name,
            tracker_args,
            self.reid_weights if tracker_name in REID_TRACKERS else None,
            tracker_name in self.appearance_trackers,
        )
        tracker_config_path = get_tracker_config_path(self.config_dir, tracker_name)
        return create_tracker(
            tracker_type=tracker_name,
            tracker_config=str(tracker_config_path),
            reid_weights=reid_weights,
            device=self.tracker_device,
            half=self.half,
            per_class=False,
            evolve_param_dict=tracker_args,
        )

    @staticmethod
    def resolve_yolo_weights_for_tracker(tracker_name: str) -> Path:
        return MYTRACK_YOLO_WEIGHTS if tracker_name == "mytrack" else DEFAULT_YOLO_WEIGHTS

    def get_detector(self, tracker_name: str) -> VideoDetector:
        weights_path = self.resolve_yolo_weights_for_tracker(tracker_name)
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found for tracker {tracker_name}: {weights_path}")
        detector = self.detectors.get(weights_path)
        if detector is None:
            LOGGER.info("Loading detector | tracker=%s | yolo_weights=%s", tracker_name, weights_path)
            detector = VideoDetector(
                yolo_weights_path=str(weights_path),
                slice_height=self.slice_size,
                slice_width=self.slice_size,
                overlap_height_ratio=self.overlap,
                overlap_width_ratio=self.overlap,
                conf_threshold=self.conf,
                device=self.detector_device,
            )
            self.detectors[weights_path] = detector
        return detector

    def count_sequence(self, tracker_name: str, sequence_dir: Path):
        LOGGER.info("Tracker=%s | Sequence=%s", tracker_name, sequence_dir.name)
        img_dir = sequence_dir / "img1"
        img_files = sorted(img_dir.glob("*.jpg"))
        if not img_files:
            raise FileNotFoundError(f"No images found in {img_dir}")

        tracker = self.build_tracker(tracker_name)
        detector = self.get_detector(tracker_name)
        counter = None
        detection_time = 0.0
        tracking_time = 0.0
        counting_time = 0.0
        processed_frames = 0

        for frame_id, img_path in enumerate(img_files, start=1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                LOGGER.warning("Skip unreadable frame: %s", img_path)
                continue
            if counter is None:
                line_x, area_x1, area_x2 = self.resolve_counter_geometry(frame.shape[1], self.area_ratio)
                counter = ObjectCounter(
                    line_x=line_x,
                    area_x1=area_x1,
                    area_x2=area_x2,
                    label_resolver=get_blueberry_stage_name,
                )
                LOGGER.info(
                    "Counter geometry | tracker=%s | sequence=%s | line_x=%d | area_x1=%d | area_x2=%d",
                    tracker_name,
                    sequence_dir.name,
                    line_x,
                    area_x1,
                    area_x2,
                )

            detect_start = time.perf_counter()
            det_list = detector.detect_frame_with_sahi(frame)
            detection_time += time.perf_counter() - detect_start
            detections = np.asarray(det_list, dtype=np.float32)
            if detections.size == 0:
                detections = np.empty((0, 6), dtype=np.float32)

            track_start = time.perf_counter()
            tracks = tracker.update(detections, frame)
            tracking_time += time.perf_counter() - track_start

            update_start = time.perf_counter()
            counter.update(build_counter_tracks(tracks), frame.shape[:2])
            counting_time += time.perf_counter() - update_start
            processed_frames += 1

            if frame_id % 100 == 0:
                LOGGER.info("Tracker=%s | Sequence=%s | Progress=%d/%d", tracker_name, sequence_dir.name, frame_id, len(img_files))

        if counter is None:
            raise RuntimeError(f"No valid frames found in {sequence_dir / 'img1'}")

        id_count, line_count, area_count = counter.results(as_counts=True)
        counter_fps = counter.fps_results()
        overall_time = detection_time + tracking_time + counting_time
        fps_stats = dict(counter_fps)
        fps_stats["overall_fps"] = processed_frames / overall_time if overall_time > 0 else 0.0
        LOGGER.info(
            "throughput | tracker=%s | sequence=%s | detect_fps=%.2f | track_fps=%.2f | count_fps=%.2f | overall_fps=%.2f",
            tracker_name,
            sequence_dir.name,
            processed_frames / detection_time if detection_time > 0 else 0.0,
            processed_frames / tracking_time if tracking_time > 0 else 0.0,
            processed_frames / counting_time if counting_time > 0 else 0.0,
            fps_stats["overall_fps"],
        )
        counter.log_results()
        return fps_stats, id_count, line_count, area_count

    def run(self) -> None:
        id_path = self.output_dir / "id_count.csv"
        line_path = self.output_dir / "line_count.csv"
        area_path = self.output_dir / "area_count.csv"
        id_keys = load_existing_keys(id_path)
        line_keys = load_existing_keys(line_path)
        area_keys = load_existing_keys(area_path)

        for tracker_name in self.trackers:
            for sequence_dir in self.sequences:
                key = (sequence_dir.name, tracker_name)
                if key in id_keys and key in line_keys and key in area_keys:
                    LOGGER.info("Skip existing result | tracker=%s | sequence=%s", tracker_name, sequence_dir.name)
                    continue

                try:
                    fps_stats, id_count, line_count, area_count = self.count_sequence(tracker_name, sequence_dir)
                except Exception as exc:
                    LOGGER.exception("Failed on tracker=%s sequence=%s: %s", tracker_name, sequence_dir.name, exc)
                    continue

                overall_fps = fps_stats.get("overall_fps", 0.0)
                if key not in id_keys:
                    append_count_row(id_path, build_count_row(sequence_dir.name, tracker_name, id_count, fps_stats.get("id_fps", 0.0), overall_fps))
                    id_keys.add(key)
                if key not in line_keys:
                    append_count_row(line_path, build_count_row(sequence_dir.name, tracker_name, line_count, fps_stats.get("line_fps", 0.0), overall_fps))
                    line_keys.add(key)
                if key not in area_keys:
                    append_count_row(area_path, build_count_row(sequence_dir.name, tracker_name, area_count, fps_stats.get("area_fps", 0.0), overall_fps))
                    area_keys.add(key)

        LOGGER.info("Saved counting results to %s", self.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blueberry stage counting with multiple BoxMOT trackers.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/wh1234_/data/blueberry_mot_stitched_walk"), help="Dataset root directory.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "count" / "stitched_walk", help="Output directory.")
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS, help="Unused legacy option. Detector weights are selected automatically per tracker.")
    parser.add_argument("--reid-path", type=Path, default=REPO_ROOT / "weights" / "osnet_ain_x1_0_blueberry.pt", help="ReID weights path.")
    parser.add_argument("--tracker-config-dir", type=Path, default=REPO_ROOT / "configs" / "trackers", help="Tracker config directory.")
    parser.add_argument("--trackers", nargs="+", default=TRACKERS, choices=TRACKERS, help="Tracker names to run.")
    parser.add_argument("--appearance-trackers", nargs="*", default=APPEARANCE_TRACKERS, choices=APPEARANCE_TRACKERS, help="Trackers forced to enable appearance models.")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold.")
    parser.add_argument("--slice-size", type=int, default=640, help="SAHI slice size.")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio.")
    parser.add_argument("--area-ratio", type=float, default=0.1, help="Counting area ratio. Default is 0.1.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Detector device.")
    parser.add_argument("--half", action="store_true", help="Use half precision for tracker reid backends.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    runner = BlueberryCounterRunner(
        data_root=args.data_root,
        output_dir=args.output_dir,
        yolo_weights=args.yolo_weights,
        reid_path=args.reid_path,
        tracker_config_dir=args.tracker_config_dir,
        trackers=args.trackers,
        appearance_trackers=args.appearance_trackers,
        conf=args.conf,
        slice_size=args.slice_size,
        overlap=args.overlap,
        area_ratio=args.area_ratio,
        device=args.device,
        half=args.half,
    )
    runner.run()


if __name__ == "__main__":
    main()
