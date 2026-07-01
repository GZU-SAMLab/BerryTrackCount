"""Run BoxMOT tracker ablations with detector-plus-TrackEval evaluation on the blueberry MOT dataset."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "trackers"))
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "TrackEval"))

from boxmot.tracker_zoo import create_tracker
import trackeval
from tools.video_detector import VideoDetector


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("mytrack_ablate")

YOLO11S_WEIGHTS = REPO_ROOT / "weights" / "yolo11s.pt"
BERRYDET_WEIGHTS = REPO_ROOT / "weights" / "berrydet_s.pt"


def _mytrack_botsort_preset(name: str, asso_func: str, with_reid: bool, aarm_open: bool, detector_weights: Path):
    """Build one MyTrack-BoTSORT ablation preset."""
    return (
        name,
        "mytrack_botsort",
        {
            "asso_func": asso_func,
            "with_reid": with_reid,
            "aarm_open": aarm_open,
        },
        detector_weights,
    )

ABLATIONS = [
    # ("deepocsort_baseline", "deepocsort", None, None),
    # (
    #     "deepocsort_yolo11s",
    #     "deepocsort",
    #     None,
    #     YOLO11S_WEIGHTS,
    # ),
    (
        "mytrack_iou_noappr_yolo11s",
        "mytrack",
        {
            "asso_func": "iou",
            "embedding_off": True,
            "aarm_open": False,
            "aw_off": True,
        },
        YOLO11S_WEIGHTS,
    ),
    (
        "mytrack_iou_yolo11s", # "mytrack_iou_noappr_yolo11l",
        "mytrack",
        {
            "asso_func": "iou",
            "embedding_off": False, # True
            "aarm_open": False, 
            "aw_off": False, # True
        },
        YOLO11S_WEIGHTS, # BERRYDET_WEIGHTS
    ),
    (
        "mytrack_iou_yolo11l", # "mytrack_iou"
        "mytrack",
        {
            "asso_func": "iou",
            "embedding_off": False,
            "aarm_open": False,
            "aw_off": False,
        },
        BERRYDET_WEIGHTS,
    ),
    (
        "mytrack_hciou",
        "mytrack",
        {
            "asso_func": "hciou",
            "embedding_off": False,
            "aarm_open": False,
            "aw_off": False,
        },
        BERRYDET_WEIGHTS,
    ),
    (
        "mytrack_aarm",
        "mytrack",
        {
            "asso_func": "iou",
            "embedding_off": False,
            "aarm_open": True,
            "aw_off": False,
        },
        BERRYDET_WEIGHTS,
    ),
    (
        "mytrack_hciou_aarm",
        "mytrack",
        {
            "asso_func": "hciou",
            "embedding_off": False,
            "aarm_open": True,
            "aw_off": False,
        },
        BERRYDET_WEIGHTS,
    ),
    # _mytrack_botsort_preset(
    #     "mytrackbotsort_iou_noappr_yolo11s",
    #     "iou",
    #     False,
    #     False,
    #     YOLO11S_WEIGHTS,
    # ),
    # _mytrack_botsort_preset(
    #     "mytrackbotsort_iou_noappr_yolo11l",
    #     "iou",
    #     False,
    #     False,
    #     BERRYDET_WEIGHTS,
    # ),
    # _mytrack_botsort_preset(
    #     "mytrackbotsort_iou",
    #     "iou",
    #     True,
    #     False,
    #     BERRYDET_WEIGHTS,
    # ),
    # _mytrack_botsort_preset(
    #     "mytrackbotsort_hciou",
    #     "hciou",
    #     True,
    #     False,
    #     BERRYDET_WEIGHTS,
    # ),
    # _mytrack_botsort_preset(
    #     "mytrackbotsort_aarm",
    #     "iou",
    #     True,
    #     True,
    #     BERRYDET_WEIGHTS,
    # ),
    # _mytrack_botsort_preset(
    #     "mytrackbotsort_hciou_aarm",
    #     "hciou",
    #     True,
    #     True,
    #     BERRYDET_WEIGHTS,
    # ),
]
REID_TRACKERS = {"deepocsort", "mytrack", "mytrack_botsort"}


class MyTrackAblationEvaluator:
    """Evaluate MyTrack ablations with the stitched-walk blueberry MOT protocol."""

    def __init__(
        self,
        data_root: str | Path,
        yolo_weights: str | Path,
        reid_weights: str | Path,
        tracker_config_dir: str | Path,
        output_dir: str | Path,
        motion_model_weights: str | Path,
        device: str = "",
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.yolo_weights = Path(yolo_weights).resolve()
        self.reid_weights = Path(reid_weights).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.motion_model_weights = str(Path(motion_model_weights))
        self.tracker_config_dir = Path(tracker_config_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.detector_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tracker_device = self._resolve_tracker_device(device)
        self.det_conf_threshold = 0.1
        self.detectors: dict[Path, VideoDetector] = {}

        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_root}")
        if not self.yolo_weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.yolo_weights}")
        if not self.reid_weights.exists():
            raise FileNotFoundError(f"ReID weights not found: {self.reid_weights}")
        if not self.tracker_config_dir.exists():
            raise FileNotFoundError(f"Tracker config directory not found: {self.tracker_config_dir}")

        self.test_dir = self._resolve_test_dir(self.data_root)
        self.sequences = sorted([d.name for d in self.test_dir.iterdir() if d.is_dir()])
        if not self.sequences:
            raise FileNotFoundError(f"No MOT sequences found in: {self.test_dir}")

        self._get_detector(self.yolo_weights)
        LOGGER.info("Found %d test sequences in %s", len(self.sequences), self.test_dir)

    def _get_detector(self, yolo_weights: str | Path) -> VideoDetector:
        """Load one detector per weight file and reuse it across presets."""
        weights_path = Path(yolo_weights).resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
        detector = self.detectors.get(weights_path)
        if detector is None:
            LOGGER.info("Loading VideoDetector from %s", weights_path)
            detector = VideoDetector(
                yolo_weights_path=str(weights_path),
                conf_threshold=self.det_conf_threshold,
                device=self.detector_device,
            )
            self.detectors[weights_path] = detector
        return detector

    @staticmethod
    def _resolve_tracker_device(device: str) -> str:
        """Resolve the BoxMOT tracker device string."""
        normalized = (device or "").strip().lower()
        if normalized in {"", "0", "cuda", "cuda:0"}:
            return "0" if torch.cuda.is_available() else "cpu"
        if normalized == "cpu":
            return "cpu"
        return normalized

    @staticmethod
    def _resolve_test_dir(data_root: Path) -> Path:
        """Accept either a dataset root containing test/ or the test split itself."""
        if (data_root / "test").exists():
            return data_root / "test"
        if any((child / "img1").exists() for child in data_root.iterdir() if child.is_dir()):
            return data_root
        raise FileNotFoundError(f"Could not locate MOT test split under: {data_root}")

    @staticmethod
    def _normalize_metric_value(value):
        """Convert TrackEval scalar-or-array fields into plain numbers."""
        if isinstance(value, (list, np.ndarray)):
            return float(value[0]) if len(value) > 0 else 0.0
        if isinstance(value, np.generic):
            return float(value)
        return value

    def _extract_metrics(self, tracker_results: dict) -> dict:
        """Extract the metrics required by the ablation table."""
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

    def _load_tracker_args(self, tracker_name: str, override: dict | None) -> dict:
        """Load the local tracker config and apply only the required ablation overrides."""
        import yaml

        config_file = self.tracker_config_dir / f"{tracker_name}.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Tracker config not found: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            tracker_config = yaml.load(f, Loader=yaml.FullLoader)
            tracker_args = {param: details["default"] for param, details in tracker_config.items()}

        # Keep the evaluation-time max_age override used by the existing scripts.
        tracker_args["max_age"] = 25

        # Keep appearance enabled for DeepOcSort/MyTrack ablations so all four settings use ReID.
        if tracker_name in REID_TRACKERS:
            tracker_args["embedding_off"] = False
            tracker_args["aw_off"] = False

        # Keep the motion checkpoint configurable from CLI without rewriting other YAML defaults.
        if tracker_name == "mytrack":
            tracker_args["motion_model_weights"] = self.motion_model_weights

        # Only apply the parameters that define the current ablation preset.
        if override:
            tracker_args.update(override)
        return tracker_args

    def _build_tracker(self, tracker_name: str, override: dict | None):
        """Instantiate a tracker for one ablation preset."""
        tracker_args = self._load_tracker_args(tracker_name, override)
        reid_weights = self.reid_weights if tracker_name in REID_TRACKERS else None
        return create_tracker(
            tracker_type=tracker_name,
            tracker_config=str(self.tracker_config_dir / f"{tracker_name}.yaml"),
            reid_weights=reid_weights,
            device=self.tracker_device,
            half=False,
            per_class=False,
            evolve_param_dict=tracker_args,
        )

    def _collect_detections(self, img: np.ndarray, yolo_weights: str | Path | None = None) -> np.ndarray:
        """Run SAHI-based detection with the same postprocessing as evaluation/1_tracker_eval.py."""
        detector = self._get_detector(yolo_weights or self.yolo_weights)
        det_list = detector.detect_frame_with_sahi(img)
        detections = np.asarray(det_list, dtype=np.float32)
        if detections.size == 0:
            return np.empty((0, 6), dtype=np.float32)
        return detections

    def detect_and_track(
        self,
        preset_name: str,
        tracker_name: str,
        override: dict | None,
        sequence: str,
        yolo_weights: str | Path | None = None,
    ) -> tuple[dict, float]:
        """Run one preset on one sequence."""
        LOGGER.info("Running %s on %s", preset_name, sequence)
        seq_dir = self.test_dir / sequence
        img_dir = seq_dir / "img1"
        img_files = sorted(img_dir.glob("*.jpg"))
        if not img_files:
            img_files = sorted(img_dir.glob("*.png"))
        if not img_files:
            LOGGER.warning("No images found in %s", img_dir)
            return {}, 0.0

        tracker = self._build_tracker(tracker_name, override)
        tracking_results = defaultdict(list)
        total_time = 0.0
        frame_count = 0

        for frame_idx, img_file in enumerate(img_files, start=1):
            img = cv2.imread(str(img_file))
            if img is None:
                LOGGER.warning("Failed to read image: %s", img_file)
                continue

            detections = self._collect_detections(img, yolo_weights=yolo_weights)
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
                    tracking_results[frame_idx].append(
                        (track_id, x1, y1, x2 - x1, y2 - y1, conf, cls)
                    )

        fps = frame_count / total_time if total_time > 0 else 0.0
        LOGGER.info("%s | %s FPS: %.2f", preset_name, sequence, fps)
        return tracking_results, fps

    def save_tracking_results(self, preset_name: str, sequence: str, tracking_results: dict) -> None:
        """Save one preset's tracking results in MOT text format."""
        tracker_output_dir = self.output_dir / "trackers" / preset_name / "data"
        seq_output_file = tracker_output_dir / f"{sequence}.txt"
        seq_output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(seq_output_file, "w", encoding="utf-8") as f:
            for frame_id in sorted(tracking_results.keys()):
                for track_id, x, y, w, h, conf, cls in tracking_results[frame_id]:
                    f.write(f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n")

    def run_trackeval(self, preset_name: str) -> dict:
        """Evaluate one preset with TrackEval using the same protocol as evaluation/1_tracker_eval.py."""
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
            "GT_FOLDER": str(self.test_dir),
            "TRACKERS_FOLDER": str(self.output_dir / "trackers"),
            "OUTPUT_FOLDER": str(self.output_dir / "eval_results"),
            "TRACKERS_TO_EVAL": [preset_name],
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
            tracker_output = output["MotChallenge2DBox"][preset_name]
            combined_metrics = self._extract_metrics(tracker_output["COMBINED_SEQ"]["pedestrian"])
            per_sequence_metrics = {}
            for sequence in self.sequences:
                if sequence in tracker_output and "pedestrian" in tracker_output[sequence]:
                    per_sequence_metrics[sequence] = self._extract_metrics(
                        tracker_output[sequence]["pedestrian"]
                    )
            return {"combined": combined_metrics, "per_sequence": per_sequence_metrics}
        except Exception as exc:
            LOGGER.error("TrackEval failed for %s: %s", preset_name, exc)
            empty_metrics = {
                "HOTA": 0.0,
                "DetA": 0.0,
                "AssA": 0.0,
                "MOTA": 0.0,
                "IDSW": 0,
                "IDF1": 0.0,
                "IDs": 0,
                "GT_IDs": 0,
            }
            return {
                "combined": empty_metrics.copy(),
                "per_sequence": {sequence: empty_metrics.copy() for sequence in self.sequences},
            }

    def evaluate_preset(
        self,
        preset_name: str,
        tracker_name: str,
        override: dict | None,
        yolo_weights: str | Path | None = None,
    ) -> dict:
        """Run one preset on all sequences and aggregate the metrics."""
        fps_list = []
        per_sequence = {}
        for sequence in self.sequences:
            tracking_results, fps = self.detect_and_track(
                preset_name,
                tracker_name,
                override,
                sequence,
                yolo_weights=yolo_weights,
            )
            fps_list.append(fps)
            self.save_tracking_results(preset_name, sequence, tracking_results)
            per_sequence[sequence] = {"FPS": fps}

        metrics = self.run_trackeval(preset_name)
        metrics["combined"]["FPS"] = float(np.mean(fps_list)) if fps_list else 0.0
        for sequence in self.sequences:
            metrics["per_sequence"].setdefault(sequence, {})
            metrics["per_sequence"][sequence]["FPS"] = per_sequence[sequence]["FPS"]
        return metrics


def build_parser() -> argparse.ArgumentParser:
    """Build the ablation CLI."""
    parser = argparse.ArgumentParser(description="Run MyTrack ablations with evaluation/1_tracker_eval.py style evaluation.")
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "dataset" / "blueberry_mot_stitched_walk",
        help="Dataset root or test split path.",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=REPO_ROOT / "weights" / "berrydet_s.pt",
        help="YOLO detector weights.",
    )
    parser.add_argument(
        "--reid-model",
        type=Path,
        default=REPO_ROOT / "weights" / "tracker" / "osnet_x0_25_blueberry.pt",
        help="ReID weights for appearance-based presets.",
    )
    parser.add_argument(
        '--tracker_config_dir',
        type=str,
        default=REPO_ROOT / 'configs/trackers')
    parser.add_argument(
        "--motion-model-weights",
        type=Path,
        default=REPO_ROOT / "weights" / "tracker" / "berrytracker_motion.pt",
        help="Motion model checkpoint used by MyTrack motion ablations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "eval" / "mytrack_ablation",
        help="Directory to store tracking outputs and ablation summaries.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Tracker/detector device, e.g. 0 or cpu.",
    )
    return parser


def main() -> None:
    """Run all four ablations and save combined plus per-sequence summaries."""
    args = build_parser().parse_args()
    evaluator = MyTrackAblationEvaluator(
        data_root=args.source,
        yolo_weights=args.yolo_model,
        reid_weights=args.reid_model,
        tracker_config_dir=args.tracker_config_dir,
        output_dir=args.output_dir,
        motion_model_weights=args.motion_model_weights,
        device=args.device,
    )

    summary_rows = []
    sequence_rows = []
    for preset_name, tracker_name, override, detector_weights in ABLATIONS:
        LOGGER.info("Evaluating preset: %s", preset_name)
        metrics = evaluator.evaluate_preset(
            preset_name,
            tracker_name,
            override,
            yolo_weights=detector_weights,
        )
        combined_metrics = metrics["combined"]
        row = {
            "preset": preset_name,
            "tracking_method": tracker_name,
            "detector_weights": str(Path(detector_weights).resolve()) if detector_weights else str(args.yolo_model.resolve()),
            "assoc_func": None if override is None else override.get("asso_func"),
            "embedding_off": None if override is None else override.get("embedding_off"),
            "aarm_open": None if override is None else override.get("aarm_open"),
            "use_hmiou": None if override is None else override.get("use_hmiou"),
            "use_motion_model": None if override is None else override.get("use_motion_model"),
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
        summary_rows.append(row)
        LOGGER.info("Preset result: %s", json.dumps(row, ensure_ascii=False))

        for sequence, seq_metrics in metrics["per_sequence"].items():
            sequence_rows.append(
                {
                    "preset": preset_name,
                    "tracking_method": tracker_name,
                    "detector_weights": str(Path(detector_weights).resolve()) if detector_weights else str(args.yolo_model.resolve()),
                    "assoc_func": None if override is None else override.get("asso_func"),
                    "embedding_off": None if override is None else override.get("embedding_off"),
                    "aarm_open": None if override is None else override.get("aarm_open"),
                    "sequence": sequence,
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
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "summary.csv"
    seq_csv_path = args.output_dir / "sequence_summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "preset",
                "tracking_method",
                "detector_weights",
                "assoc_func",
                "embedding_off",
                "aarm_open",
                "use_hmiou",
                "use_motion_model",
                "HOTA",
                "DetA",
                "AssA",
                "MOTA",
                "IDSW",
                "IDF1",
                "IDs",
                "GT_IDs",
                "FPS",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(seq_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "preset",
                "tracking_method",
                "detector_weights",
                "assoc_func",
                "embedding_off",
                "aarm_open",
                "sequence",
                "HOTA",
                "DetA",
                "AssA",
                "MOTA",
                "IDSW",
                "IDF1",
                "IDs",
                "GT_IDs",
                "FPS",
            ],
        )
        writer.writeheader()
        writer.writerows(sequence_rows)

    LOGGER.info("Saved summary table to %s", csv_path)
    LOGGER.info("Saved per-sequence table to %s", seq_csv_path)


if __name__ == "__main__":
    main()
