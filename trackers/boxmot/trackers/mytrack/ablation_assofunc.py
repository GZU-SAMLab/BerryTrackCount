"""Evaluate MyTrack association-function ablations without the motion model."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "trackers"))
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "TrackEval"))

from boxmot.trackers.mytrack.ablate import MyTrackAblationEvaluator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("ablation_assofunc")

ASSO_FUNCS = (
    "iou",
    "hiou",
    "hmiou",
    "siou",
    "ciou",
    "diou",
    "eiou",
    "hsiou",
    "hciou",
    "hdiou",
    "heioud",
)

TRACKER_NAMES = ("mytrack",)


def build_override(tracker_name: str, assoc_name: str) -> dict:
    """Build the minimal override set for one tracker/association pair."""
    return {
        "asso_func": assoc_name,
        "aarm_open": False,
        "embedding_off": False,
        "aw_off": False,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for association-function ablations."""
    parser = argparse.ArgumentParser(
        description="Run MyTrack association-function ablations without the motion model."
    )
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
        help="ReID weights used by MyTrack.",
    )
    parser.add_argument(
        "--tracker-config-dir",
        type=Path,
        default=REPO_ROOT / "configs" / "trackers",
        help="Directory containing tracker YAML configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "eval" / "ablation_assofunc",
        help="Directory for tracking outputs and the summary CSV.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Tracker device, e.g. 0 or cpu.",
    )
    return parser


def main() -> None:
    """Evaluate the requested association functions and save one CSV summary."""
    args = build_parser().parse_args()
    evaluator = MyTrackAblationEvaluator(
        data_root=args.source,
        yolo_weights=args.yolo_model,
        reid_weights=args.reid_model,
        tracker_config_dir=args.tracker_config_dir,
        output_dir=args.output_dir,
        motion_model_weights=REPO_ROOT / "weights" / "tracker" / "berrytracker_motion.pt",
        device=args.device,
    )

    rows = []
    for tracker_name in TRACKER_NAMES:
        for assoc_name in ASSO_FUNCS:
            preset_name = f"{tracker_name}_{assoc_name}"
            override = build_override(tracker_name, assoc_name)
            LOGGER.info(
                "Evaluating preset=%s | tracker=%s | assoc_func=%s",
                preset_name,
                tracker_name,
                assoc_name,
            )
            metrics = evaluator.evaluate_preset(preset_name, tracker_name, override)["combined"]
            row = {
                "preset": preset_name,
                "tracking_method": tracker_name,
                "assoc_func": assoc_name,
                "use_motion_model": False,
                "aarm_open": override.get("aarm_open"),
                "HOTA": metrics.get("HOTA", 0.0),
                "DetA": metrics.get("DetA", 0.0),
                "AssA": metrics.get("AssA", 0.0),
                "MOTA": metrics.get("MOTA", 0.0),
                "IDSW": metrics.get("IDSW", 0),
                "IDF1": metrics.get("IDF1", 0.0),
                "IDs": metrics.get("IDs", 0),
                "GT_IDs": metrics.get("GT_IDs", 0),
                "FPS": metrics.get("FPS", 0.0),
            }
            rows.append(row)
            LOGGER.info(
                "Result | tracker=%s | assoc=%s | HOTA=%.4f | DetA=%.4f | AssA=%.4f | MOTA=%.4f | IDSW=%s | IDF1=%.4f | IDs=%s | GT_IDs=%s | FPS=%.2f",
                tracker_name,
                assoc_name,
                row["HOTA"],
                row["DetA"],
                row["AssA"],
                row["MOTA"],
                row["IDSW"],
                row["IDF1"],
                row["IDs"],
                row["GT_IDs"],
                row["FPS"],
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ablation_assofunc.csv"
    fieldnames = [
        "preset",
        "tracking_method",
        "assoc_func",
        "use_motion_model",
        "aarm_open",
        "HOTA",
        "DetA",
        "AssA",
        "MOTA",
        "IDSW",
        "IDF1",
        "IDs",
        "GT_IDs",
        "FPS",
    ]
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Saved association-function ablation summary to %s", csv_path)


if __name__ == "__main__":
    main()
