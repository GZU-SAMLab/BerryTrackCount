#!/usr/bin/env python3
# Purpose: run center-area counting ablations on blueberry MOT sequences with multiple BoxMOT trackers.

"""Area counting ablation for center regions with widths 2%, 4%, 6%, and 8%."""

import argparse
import csv
import importlib.util
import logging
from pathlib import Path
from typing import Sequence, Tuple

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_MODULE_PATH = REPO_ROOT / "count" / "blueberry_count.py"
BASE_SPEC = importlib.util.spec_from_file_location("count_blueberry_count", BASE_MODULE_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise ImportError(f"Failed to load base count module: {BASE_MODULE_PATH}")
BASE_MODULE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE_MODULE)


LOGGER = logging.getLogger("blueberry_area_ablation")
AREA_RATIOS = (0.02, 0.04, 0.06, 0.08)
CSV_HEADER = ["video_name", "tracker", "fps", "overall_fps", *BASE_MODULE.STAGES, "total"]

ObjectCounter = BASE_MODULE.ObjectCounter
TRACKERS = BASE_MODULE.TRACKERS
APPEARANCE_TRACKERS = BASE_MODULE.APPEARANCE_TRACKERS
build_counter_tracks = BASE_MODULE.build_counter_tracks
build_count_row = BASE_MODULE.build_count_row
load_existing_keys = BASE_MODULE.load_existing_keys


def append_area_row(path: Path, row: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)


class BlueberryAreaAblationRunner(BASE_MODULE.BlueberryCounterRunner):
    @staticmethod
    def resolve_area_geometry(frame_width: int, area_ratio: float) -> Tuple[int, int, int]:
        line_x = int(frame_width * 0.5)
        half_width = int(frame_width * area_ratio * 0.5)
        area_x1 = max(0, line_x - half_width)
        area_x2 = min(frame_width, line_x + half_width)
        return line_x, area_x1, area_x2

    def count_sequence(self, tracker_name: str, sequence_dir: Path, area_ratio: float):
        LOGGER.info(
            "Tracker=%s | Sequence=%s | AreaWidth=%.0f%%",
            tracker_name,
            sequence_dir.name,
            area_ratio * 100.0,
        )
        img_dir = sequence_dir / "img1"
        img_files = sorted(img_dir.glob("*.jpg"))
        if not img_files:
            raise FileNotFoundError(f"No images found in {img_dir}")

        tracker = self.build_tracker(tracker_name)
        detector = self.get_detector(tracker_name)
        counter = None
        detection_time = tracking_time = counting_time = 0.0
        processed_frames = 0

        for frame_id, img_path in enumerate(img_files, start=1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                LOGGER.warning("Skip unreadable frame: %s", img_path)
                continue
            if counter is None:
                line_x, area_x1, area_x2 = self.resolve_area_geometry(frame.shape[1], area_ratio)
                counter = ObjectCounter(
                    line_x=line_x,
                    area_x1=area_x1,
                    area_x2=area_x2,
                    label_resolver=BASE_MODULE.get_blueberry_stage_name,
                )
                LOGGER.info(
                    "Area geometry | tracker=%s | sequence=%s | width=%.0f%% | area_x1=%d | area_x2=%d",
                    tracker_name,
                    sequence_dir.name,
                    area_ratio * 100.0,
                    area_x1,
                    area_x2,
                )

            start = BASE_MODULE.time.perf_counter()
            detections = BASE_MODULE.np.asarray(detector.detect_frame_with_sahi(frame), dtype=BASE_MODULE.np.float32)
            detection_time += BASE_MODULE.time.perf_counter() - start
            if detections.size == 0:
                detections = BASE_MODULE.np.empty((0, 6), dtype=BASE_MODULE.np.float32)

            start = BASE_MODULE.time.perf_counter()
            tracks = tracker.update(detections, frame)
            tracking_time += BASE_MODULE.time.perf_counter() - start

            start = BASE_MODULE.time.perf_counter()
            counter.update(build_counter_tracks(tracks), frame.shape[:2])
            counting_time += BASE_MODULE.time.perf_counter() - start
            processed_frames += 1

            if frame_id % 100 == 0:
                LOGGER.info(
                    "Tracker=%s | Sequence=%s | AreaWidth=%.0f%% | Progress=%d/%d",
                    tracker_name,
                    sequence_dir.name,
                    area_ratio * 100.0,
                    frame_id,
                    len(img_files),
                )

        if counter is None:
            raise RuntimeError(f"No valid frames found in {img_dir}")

        _, _, area_count = counter.results(as_counts=True)
        fps_stats = counter.fps_results()
        overall_time = detection_time + tracking_time + counting_time
        overall_fps = processed_frames / overall_time if overall_time > 0 else 0.0
        LOGGER.info(
            "Throughput | tracker=%s | sequence=%s | area_width=%.0f%% | detect_fps=%.2f | track_fps=%.2f | area_fps=%.2f | overall_fps=%.2f",
            tracker_name,
            sequence_dir.name,
            area_ratio * 100.0,
            processed_frames / detection_time if detection_time > 0 else 0.0,
            processed_frames / tracking_time if tracking_time > 0 else 0.0,
            fps_stats.get("area_fps", 0.0),
            overall_fps,
        )
        return area_count, fps_stats.get("area_fps", 0.0), overall_fps

    def run(self) -> None:
        for tracker_name in self.trackers:
            for area_ratio in AREA_RATIOS:
                area_width_percent = f"{int(area_ratio * 100)}"
                output_path = self.output_dir / f"area_count_ablation_{area_width_percent}pct.csv"
                done = load_existing_keys(output_path)
                for sequence_dir in self.sequences:
                    key = (sequence_dir.name, tracker_name)
                    if key in done:
                        LOGGER.info(
                            "Skip existing result | tracker=%s | sequence=%s | area_width=%s%%",
                            tracker_name,
                            sequence_dir.name,
                            area_width_percent,
                        )
                        continue
                    try:
                        area_count, area_fps, overall_fps = self.count_sequence(tracker_name, sequence_dir, area_ratio)
                    except Exception as exc:
                        LOGGER.exception(
                            "Failed | tracker=%s | sequence=%s | area_width=%s%% | error=%s",
                            tracker_name,
                            sequence_dir.name,
                            area_width_percent,
                            exc,
                        )
                        continue
                    row = build_count_row(sequence_dir.name, tracker_name, area_count, area_fps, overall_fps)
                    append_area_row(output_path, row)
                    done.add(key)
                LOGGER.info("Saved area ablation results to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blueberry center-area counting ablation with multiple BoxMOT trackers.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/wh1234_/data/blueberry_mot_stitched_walk"), help="Dataset root directory.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "count" / "stitched_walk", help="Output directory.")
    parser.add_argument("--reid-path", type=Path, default=REPO_ROOT / "weights" / "osnet_ain_x1_0_blueberry.pt", help="ReID weights path.")
    parser.add_argument("--tracker-config-dir", type=Path, default=REPO_ROOT / "configs" / "trackers", help="Tracker config directory.")
    parser.add_argument("--trackers", nargs="+", default=TRACKERS, choices=TRACKERS, help="Tracker names to run.")
    parser.add_argument("--appearance-trackers", nargs="*", default=APPEARANCE_TRACKERS, choices=APPEARANCE_TRACKERS, help="Trackers forced to enable appearance models.")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold.")
    parser.add_argument("--slice-size", type=int, default=640, help="SAHI slice size.")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio.")
    parser.add_argument("--device", type=str, default="cuda" if BASE_MODULE.torch.cuda.is_available() else "cpu", help="Detector device.")
    parser.add_argument("--half", action="store_true", help="Use half precision for tracker reid backends.")
    return parser.parse_args()


def main() -> None:
    BASE_MODULE.setup_logging()
    args = parse_args()
    runner = BlueberryAreaAblationRunner(
        data_root=args.data_root,
        output_dir=args.output_dir,
        yolo_weights=BASE_MODULE.DEFAULT_YOLO_WEIGHTS,
        reid_path=args.reid_path,
        tracker_config_dir=args.tracker_config_dir,
        trackers=args.trackers,
        appearance_trackers=args.appearance_trackers,
        conf=args.conf,
        slice_size=args.slice_size,
        overlap=args.overlap,
        line_x=0.5,
        area_x1=0.49,
        area_x2=0.51,
        device=args.device,
        half=args.half,
    )
    runner.run()


if __name__ == "__main__":
    main()
