#!/usr/bin/env python3
"""
Multi-tracker evaluation on the stitched-walk blueberry MOT dataset.
This variant disables appearance matching where supported.
"""

import sys
import csv
import time
import logging
from pathlib import Path
from collections import defaultdict
import argparse

import cv2
import numpy as np
import torch
import yaml

# Add paths for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "trackers"))
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "TrackEval"))

from boxmot.tracker_zoo import create_tracker
import trackeval
from tools.video_detector import VideoDetector


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TrackerEvaluator:
    """Multi-tracker evaluator for the stitched-walk blueberry MOT dataset."""

    TRACKERS = [
        "bytetrack",
        "ocsort",
        "deepocsort",
        "mytrack",
        "mytrack_botsort",
        "botsort",
        "boosttrack",
        "hybridsort",
        "topictrack",
    ]
    TRACKER_FIELDS = ["Tracker", "HOTA", "DetA", "AssA", "MOTA", "IDSW", "IDF1", "IDs", "GT_IDs", "FPS"]
    SEQUENCE_FIELDS = ["Tracker", "Sequence", "HOTA", "DetA", "AssA", "MOTA", "IDSW", "IDF1", "IDs", "GT_IDs", "FPS"]

    def __init__(self, data_root: str, yolo_weights: str, tracker_config_dir: str, output_dir: str, reid_path: str):
        self.data_root = Path(data_root)
        self.yolo_weights = Path(yolo_weights)
        self.output_dir = Path(output_dir)
        self.tracker_config_dir = Path(tracker_config_dir)
        self.reid_weights = Path(reid_path) if reid_path else None
        self.detector_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tracker_device = "0" if torch.cuda.is_available() else "cpu"
        self.det_conf_threshold = 0.1
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_root}")
        if not self.yolo_weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.yolo_weights}")
        if not self.tracker_config_dir.exists():
            raise FileNotFoundError(f"Tracker config directory not found: {self.tracker_config_dir}")
        if self.reid_weights is not None and not self.reid_weights.exists():
            raise FileNotFoundError(f"ReID weights not found: {self.reid_weights}")

        logger.info(f"Loading VideoDetector: {self.yolo_weights}")
        self.detector = VideoDetector(
            yolo_weights_path=str(self.yolo_weights),
            conf_threshold=self.det_conf_threshold,
            device=self.detector_device,
        )

        self.test_dir = self.data_root / "test"
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")

        self.sequences = sorted([d.name for d in self.test_dir.iterdir() if d.is_dir()])
        self.results_csv = self.output_dir / "tracker_evaluation_results.csv"
        self.sequence_results_csv = self.output_dir / "tracker_sequence_evaluation_results.csv"
        logger.info(f"Found {len(self.sequences)} test sequences")

    @staticmethod
    def _normalize_metric_value(value):
        if isinstance(value, (list, np.ndarray)):
            return float(value[0]) if len(value) > 0 else 0.0
        if isinstance(value, np.generic):
            return float(value)
        return value

    def _extract_metrics(self, tracker_results: dict) -> dict:
        metrics = {
            "HOTA": tracker_results.get("HOTA", {}).get("HOTA", [0]),
            "DetA": tracker_results.get("HOTA", {}).get("DetA", [0]),
            "AssA": tracker_results.get("HOTA", {}).get("AssA", [0]),
            "MOTA": tracker_results.get("CLEAR", {}).get("MOTA", 0),
            "IDSW": tracker_results.get("CLEAR", {}).get("IDSW", 0),
            "IDF1": tracker_results.get("Identity", {}).get("IDF1", 0),
            "IDs": tracker_results.get("Count", {}).get("IDs", 0),
             "GT_IDs": tracker_results.get("Count", {}).get("GT_IDs", 0),
        }
        return {key: self._normalize_metric_value(value) for key, value in metrics.items()}

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "HOTA": 0.0,
            "DetA": 0.0,
            "AssA": 0.0,
            "MOTA": 0.0,
            "IDSW": 0,
            "IDF1": 0.0,
            "IDs": 0,
            "GT_IDs": 0,
        }

    def _tracker_data_dir(self, tracker_name: str) -> Path:
        return self.output_dir / "trackers" / tracker_name / "data"

    def _tracker_eval_dir(self, tracker_name: str) -> Path:
        return self.output_dir / "eval_results" / tracker_name

    def _tracker_data_complete(self, tracker_name: str) -> bool:
        data_dir = self._tracker_data_dir(tracker_name)
        return data_dir.exists() and all((data_dir / f"{sequence}.txt").exists() for sequence in self.sequences)

    def _tracker_eval_complete(self, tracker_name: str) -> bool:
        return (self._tracker_eval_dir(tracker_name) / "pedestrian_summary.txt").exists()

    def _load_existing_tracker_rows(self) -> tuple[dict, dict]:
        tracker_rows = {}
        if self.results_csv.exists():
            with open(self.results_csv, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    tracker_rows[row["Tracker"]] = row

        sequence_rows = {}
        if self.sequence_results_csv.exists():
            with open(self.sequence_results_csv, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sequence_rows[(row["Tracker"], row["Sequence"])] = row

        return tracker_rows, sequence_rows

    def _append_tracker_results(self, tracker_rows: list[dict], sequence_rows: list[dict]):
        tracker_file_exists = self.results_csv.exists()
        with open(self.results_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.TRACKER_FIELDS)
            if not tracker_file_exists or self.results_csv.stat().st_size == 0:
                writer.writeheader()
            writer.writerows(tracker_rows)

        sequence_file_exists = self.sequence_results_csv.exists()
        with open(self.sequence_results_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SEQUENCE_FIELDS)
            if not sequence_file_exists or self.sequence_results_csv.stat().st_size == 0:
                writer.writeheader()
            writer.writerows(sequence_rows)

    @staticmethod
    def _row_to_result(row: dict) -> dict:
        return {
            "Tracker": row["Tracker"],
            "HOTA": float(row["HOTA"]),
            "DetA": float(row["DetA"]),
            "AssA": float(row["AssA"]),
            "MOTA": float(row["MOTA"]),
            "IDSW": int(float(row["IDSW"])),
            "IDF1": float(row["IDF1"]),
            "IDs": int(float(row["IDs"])),
            "GT_IDs": int(float(row["GT_IDs"])),
            "FPS": float(row["FPS"]),
        }

    @staticmethod
    def _row_to_sequence_result(row: dict) -> dict:
        return {
            "Tracker": row["Tracker"],
            "Sequence": row["Sequence"],
            "HOTA": float(row["HOTA"]),
            "DetA": float(row["DetA"]),
            "AssA": float(row["AssA"]),
            "MOTA": float(row["MOTA"]),
            "IDSW": int(float(row["IDSW"])),
            "IDF1": float(row["IDF1"]),
            "IDs": int(float(row["IDs"])),
            "GT_IDs": int(float(row["GT_IDs"])),
            "FPS": float(row["FPS"]),
        }

    def _build_tracker(self, tracker_name: str):
        config_file = self.tracker_config_dir / f"{tracker_name}.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Tracker config not found: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            tracker_config = yaml.load(f, Loader=yaml.FullLoader)
            tracker_args = {param: details["default"] for param, details in tracker_config.items()}

        tracker_args["max_age"] = 25
        reid_weights = None

        if tracker_name == "deepocsort":
            tracker_args["embedding_off"] = True
            tracker_args["aw_off"] = True
        elif tracker_name == "mytrack":
            tracker_args["embedding_off"] = True
            tracker_args["aw_off"] = True
            tracker_args["aarm_open"] = False
        elif tracker_name == "mytrack_botsort":
            tracker_args["with_reid"] = False
            tracker_args["aarm_open"] = False
        elif tracker_name in {"botsort", "boosttrack"}:
            tracker_args["with_reid"] = False
        elif tracker_name == "hybridsort":
            tracker_args["with_reid"] = False
            tracker_args["with_longterm_reid"] = False
            tracker_args["with_longterm_reid_correction"] = False
        elif tracker_name == "topictrack":
            if self.reid_weights is None:
                raise FileNotFoundError("TOPICTrack requires --reid_path because its current implementation always builds a ReID backend.")
            reid_weights = self.reid_weights
            tracker_args["metric"] = "cosine"
            tracker_args["two_round_off"] = False

        tracker = create_tracker(
            tracker_type=tracker_name,
            tracker_config=str(config_file),
            reid_weights=reid_weights,
            device=self.tracker_device,
            half=False,
            per_class=False,
            evolve_param_dict=tracker_args,
        )
        return tracker

    def detect_and_track(self, tracker_name: str, sequence: str) -> tuple:
        logger.info(f"  Running {tracker_name} on {sequence}")

        seq_dir = self.test_dir / sequence
        img_dir = seq_dir / "img1"
        img_files = sorted(img_dir.glob("*.jpg"))
        if not img_files:
            logger.warning(f"No images found in {img_dir}")
            return {}, 0.0

        tracker = self._build_tracker(tracker_name)
        tracking_results = defaultdict(list)
        total_time = 0.0
        frame_count = 0

        for frame_idx, img_file in enumerate(img_files, start=1):
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning(f"Failed to read image: {img_file}")
                continue

            det_list = self.detector.detect_frame_with_sahi(img)
            detections = np.asarray(det_list, dtype=np.float32)
            if detections.size == 0:
                detections = np.empty((0, 6), dtype=np.float32)

            start_time = time.time()
            tracks = tracker.update(detections, img)
            total_time += time.time() - start_time
            frame_count += 1

            if len(tracks) > 0:
                for track in tracks:
                    x1, y1, x2, y2 = track[:4]
                    track_id = int(track[4])
                    conf = track[5] if len(track) > 5 else 1.0
                    cls = int(track[6]) if len(track) > 6 else 0
                    w = x2 - x1
                    h = y2 - y1
                    tracking_results[frame_idx].append((track_id, x1, y1, w, h, conf, cls))

        fps = frame_count / total_time if total_time > 0 else 0.0
        logger.info(f"    FPS: {fps:.2f}")
        return tracking_results, fps

    def save_tracking_results(self, tracker_name: str, sequence: str, tracking_results: dict):
        tracker_output_dir = self._tracker_data_dir(tracker_name)
        seq_output_file = tracker_output_dir / f"{sequence}.txt"
        seq_output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(seq_output_file, "w", encoding="utf-8") as f:
            for frame_id in sorted(tracking_results.keys()):
                for track_id, x, y, w, h, conf, cls in tracking_results[frame_id]:
                    f.write(f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n")

    def run_trackeval(self, tracker_name: str) -> dict:
        eval_config = {
            "USE_PARALLEL": False,
            "NUM_PARALLEL_CORES": 1,
            "BREAK_ON_ERROR": True,
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": True,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
            "DISPLAY_LESS_PROGRESS": True,
        }

        dataset_config = {
            "GT_FOLDER": str(self.data_root / "test"),
            "TRACKERS_FOLDER": str(self.output_dir / "trackers"),
            "OUTPUT_FOLDER": str(self.output_dir / "eval_results"),
            "TRACKERS_TO_EVAL": [tracker_name],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "sub_gmot",
            "SPLIT_TO_EVAL": "test",
            "INPUT_AS_ZIP": False,
            "PRINT_CONFIG": False,
            "DO_PREPROC": False,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": {seq: None for seq in self.sequences},
        }

        metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}

        try:
            evaluator = trackeval.Evaluator(eval_config)
            dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
            metrics_list = [
                trackeval.metrics.HOTA(metrics_config),
                trackeval.metrics.CLEAR(metrics_config),
                trackeval.metrics.Identity(metrics_config),
            ]
            output, _ = evaluator.evaluate([dataset], metrics_list)
            tracker_output = output["MotChallenge2DBox"][tracker_name]
            combined_metrics = self._extract_metrics(tracker_output["COMBINED_SEQ"]["pedestrian"])
            per_sequence_metrics = {}
            for sequence in self.sequences:
                if sequence in tracker_output and "pedestrian" in tracker_output[sequence]:
                    per_sequence_metrics[sequence] = self._extract_metrics(tracker_output[sequence]["pedestrian"])

            return {"combined": combined_metrics, "per_sequence": per_sequence_metrics}
        except Exception as e:
            logger.error(f"TrackEval failed for {tracker_name}: {e}")
            empty_metrics = self._empty_metrics()
            return {
                "combined": empty_metrics.copy(),
                "per_sequence": {sequence: empty_metrics.copy() for sequence in self.sequences},
            }

    def evaluate_tracker(self, tracker_name: str, cached_tracker_rows: dict, cached_sequence_rows: dict) -> dict:
        if self._tracker_eval_complete(tracker_name):
            logger.info(f"Skipping detection/tracking for {tracker_name}: existing eval_results found.")
            metrics = self.run_trackeval(tracker_name)
            metrics["combined"]["FPS"] = float(cached_tracker_rows.get(tracker_name, {}).get("FPS", 0.0))
            for sequence in self.sequences:
                metrics["per_sequence"].setdefault(sequence, self._empty_metrics().copy())
                metrics["per_sequence"][sequence]["FPS"] = float(
                    cached_sequence_rows.get((tracker_name, sequence), {}).get("FPS", 0.0)
                )
            return metrics

        if self._tracker_data_complete(tracker_name):
            logger.info(f"Skipping tracking for {tracker_name}: complete cached MOT data found.")
            metrics = self.run_trackeval(tracker_name)
            metrics["combined"]["FPS"] = float(cached_tracker_rows.get(tracker_name, {}).get("FPS", 0.0))
            for sequence in self.sequences:
                metrics["per_sequence"].setdefault(sequence, self._empty_metrics().copy())
                metrics["per_sequence"][sequence]["FPS"] = float(
                    cached_sequence_rows.get((tracker_name, sequence), {}).get("FPS", 0.0)
                )
            return metrics

        logger.info(f"Evaluating tracker: {tracker_name}")
        fps_list = []
        seq_fps = {}
        for sequence in self.sequences:
            tracking_results, fps = self.detect_and_track(tracker_name, sequence)
            fps_list.append(fps)
            seq_fps[sequence] = fps
            self.save_tracking_results(tracker_name, sequence, tracking_results)

        avg_fps = np.mean(fps_list) if fps_list else 0.0
        logger.info(f"Running TrackEval for {tracker_name}")
        metrics = self.run_trackeval(tracker_name)
        metrics["combined"]["FPS"] = avg_fps
        for sequence in self.sequences:
            metrics["per_sequence"].setdefault(sequence, self._empty_metrics().copy())
            metrics["per_sequence"][sequence]["FPS"] = seq_fps.get(sequence, 0.0)
        return metrics

    def evaluate_all(self):
        logger.info("Starting multi-tracker evaluation")
        logger.info(f"Trackers: {', '.join(self.TRACKERS)}")
        logger.info(f"Sequences: {len(self.sequences)}")

        cached_tracker_rows, cached_sequence_rows = self._load_existing_tracker_rows()
        results = [self._row_to_result(cached_tracker_rows[tracker_name]) for tracker_name in self.TRACKERS if tracker_name in cached_tracker_rows]
        sequence_results = [
            self._row_to_sequence_result(cached_sequence_rows[(tracker_name, sequence)])
            for tracker_name in self.TRACKERS
            for sequence in self.sequences
            if (tracker_name, sequence) in cached_sequence_rows
        ]

        for tracker_name in self.TRACKERS:
            if tracker_name in cached_tracker_rows:
                logger.info(f"Skipping CSV append for {tracker_name}: existing CSV rows found.")
                continue
            try:
                metrics = self.evaluate_tracker(tracker_name, cached_tracker_rows, cached_sequence_rows)
                combined_metrics = metrics.get("combined", {})
                result = {
                    "Tracker": tracker_name,
                    "HOTA": combined_metrics.get("HOTA", 0.0),
                    "DetA": combined_metrics.get("DetA", 0.0),
                    "AssA": combined_metrics.get("AssA", 0.0),
                    "MOTA": combined_metrics.get("MOTA", 0.0),
                    "IDSW": combined_metrics.get("IDSW", 0),
                    "IDF1": combined_metrics.get("IDF1", 0.0),
                    "IDs": combined_metrics.get("IDs", 0),
                    "GT_IDs": combined_metrics.get("GT_IDs", 0),
                    "FPS": combined_metrics.get("FPS", 0.0),
                }
                results.append(result)
                cached_tracker_rows[tracker_name] = {key: str(value) for key, value in result.items()}

                tracker_sequence_rows = []
                for sequence in self.sequences:
                    seq_metrics = metrics.get("per_sequence", {}).get(sequence, {})
                    seq_result = {
                        "Tracker": tracker_name,
                        "Sequence": sequence,
                        "HOTA": seq_metrics.get("HOTA", 0.0),
                        "DetA": seq_metrics.get("DetA", 0.0),
                        "AssA": seq_metrics.get("AssA", 0.0),
                        "MOTA": seq_metrics.get("MOTA", 0.0),
                        "IDSW": seq_metrics.get("IDSW", 0),
                        "IDF1": seq_metrics.get("IDF1", 0.0),
                        "IDs": seq_metrics.get("IDs", 0),
                        "GT_IDs": seq_metrics.get("GT_IDs", 0),
                        "FPS": seq_metrics.get("FPS", 0.0),
                    }
                    sequence_results.append(seq_result)
                    tracker_sequence_rows.append(seq_result)
                    cached_sequence_rows[(tracker_name, sequence)] = {key: str(value) for key, value in seq_result.items()}

                self._append_tracker_results([result], tracker_sequence_rows)

                logger.info(f"Results for {tracker_name}:")
                logger.info(f"  HOTA: {result['HOTA']:.4f}")
                logger.info(f"  DetA: {result['DetA']:.4f}")
                logger.info(f"  AssA: {result['AssA']:.4f}")
                logger.info(f"  MOTA: {result['MOTA']:.4f}")
                logger.info(f"  IDSW: {result['IDSW']}")
                logger.info(f"  IDF1: {result['IDF1']:.4f}")
                logger.info(f"  IDs: {result['IDs']}")
                logger.info(f"  GT_IDs: {result['GT_IDs']}")
                logger.info(f"  FPS: {result['FPS']:.2f}")
            except Exception as e:
                logger.error(f"Failed to evaluate {tracker_name}: {e}")
                result = {
                    "Tracker": tracker_name,
                    "HOTA": 0.0,
                    "DetA": 0.0,
                    "AssA": 0.0,
                    "MOTA": 0.0,
                    "IDSW": 0,
                    "IDF1": 0.0,
                    "IDs": 0,
                    "GT_IDs": 0,
                    "FPS": 0.0,
                }
                results.append(result)
                cached_tracker_rows[tracker_name] = {key: str(value) for key, value in result.items()}
                tracker_sequence_rows = []
                for sequence in self.sequences:
                    seq_result = {
                        "Tracker": tracker_name,
                        "Sequence": sequence,
                        "HOTA": 0.0,
                        "DetA": 0.0,
                        "AssA": 0.0,
                        "MOTA": 0.0,
                        "IDSW": 0,
                        "IDF1": 0.0,
                        "IDs": 0,
                        "GT_IDs": 0,
                        "FPS": 0.0,
                    }
                    sequence_results.append(seq_result)
                    tracker_sequence_rows.append(seq_result)
                    cached_sequence_rows[(tracker_name, sequence)] = {key: str(value) for key, value in seq_result.items()}
                self._append_tracker_results([result], tracker_sequence_rows)

        logger.info(f"Results saved to: {self.results_csv}")
        logger.info(f"Per-sequence results saved to: {self.sequence_results_csv}")

        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 70)
        logger.info(
            f"{'Tracker':<15} {'HOTA':<10} {'DetA':<10} {'AssA':<10} {'MOTA':<10} {'IDSW':<8} {'IDF1':<10} {'IDs':<8} {'GT_IDs':<8} {'FPS':<10}"
        )
        logger.info("-" * 128)
        for result in results:
            logger.info(
                f"{result['Tracker']:<15} {result['HOTA']:<10.4f} {result['DetA']:<10.4f} {result['AssA']:<10.4f} "
                f"{result['MOTA']:<10.4f} {result['IDSW']:<8} {result['IDF1']:<10.4f} {result['IDs']:<8} "
                f"{result['GT_IDs']:<8} {result['FPS']:<10.2f}"
            )
        logger.info("=" * 128)


def main():
    parser = argparse.ArgumentParser(description="Multi-Tracker Evaluation on stitched-walk blueberry MOT Dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(REPO_ROOT / "dataset" / "blueberry_mot_stitched_walk"),
        help="Stitched-walk blueberry MOT dataset root path",
    )
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default="weights/berrydet_s.pt",
        help="YOLO model weights path",
    )
    parser.add_argument("--tracker_config_dir", type=str, default="configs/trackers")
    parser.add_argument(
        "--reid_path",
        type=str,
        default="weights/osnet_ain_x1_0_blueberry.pt",
        help="Optional ReID weights path. Required by the current TOPICTrack implementation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "output" / "eval" / "blueberry_mot_stitched_walk"),
        help="Output directory for evaluation results",
    )

    args = parser.parse_args()

    try:
        evaluator = TrackerEvaluator(
            args.data_root,
            args.yolo_weights,
            args.tracker_config_dir,
            args.output_dir,
            args.reid_path,
        )
        evaluator.evaluate_all()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
