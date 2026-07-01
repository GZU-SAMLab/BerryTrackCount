"""Evaluate MyTrack appearance ablations with the same summary format as ablation_assofunc."""

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
LOGGER = logging.getLogger("ablation_appearance")

FIELDNAMES = [
    "preset",
    "appearance_mode",
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
]

APPEARANCE_PRESETS = [
    (
        "mytrack_no_appearance",
        "disabled",
        {
            "embedding_off": True,
            "aw_off": True,
            "aarm_open": False,
        },
    ),
    (
        "mytrack_appearance_raw",
        "raw",
        {
            "embedding_off": False,
            "aw_off": False,
            "aarm_open": False,
        },
    ),
    (
        "mytrack_appearance_aarm",
        "aarm",
        {
            "embedding_off": False,
            "aw_off": False,
            "aarm_open": True,
        },
    ),
]


def load_existing_rows(csv_path: Path) -> dict[str, dict]:
    """Load completed presets from an existing summary CSV."""
    if not csv_path.exists():
        return {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return {row["preset"]: row for row in csv.DictReader(f) if row.get("preset")}


def save_rows(csv_path: Path, rows_by_preset: dict[str, dict]) -> None:
    """Persist the current summary in preset order."""
    ordered_rows = [
        rows_by_preset[preset_name]
        for preset_name, _, _ in APPEARANCE_PRESETS
        if preset_name in rows_by_preset
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered_rows)


def has_complete_tracking_outputs(output_dir: Path, preset_name: str, sequences: list[str]) -> bool:
    """Check whether all MOT result files already exist for one preset."""
    tracker_output_dir = output_dir / "trackers" / preset_name / "data"
    return bool(sequences) and all((tracker_output_dir / f"{sequence}.txt").exists() for sequence in sequences)


def build_result_row(preset_name: str, appearance_mode: str, metrics: dict, fps: float | None = None) -> dict:
    """Convert evaluator metrics into one CSV row."""
    return {
        "preset": preset_name,
        "appearance_mode": appearance_mode,
        "use_motion_model": False,
        "HOTA": metrics.get("HOTA", 0.0),
        "DetA": metrics.get("DetA", 0.0),
        "AssA": metrics.get("AssA", 0.0),
        "MOTA": metrics.get("MOTA", 0.0),
        "IDSW": metrics.get("IDSW", 0),
        "IDF1": metrics.get("IDF1", 0.0),
        "IDs": metrics.get("IDs", 0),
        "GT_IDs": metrics.get("GT_IDs", 0),
        "FPS": metrics.get("FPS", 0.0) if fps is None else fps,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for appearance ablations."""
    parser = argparse.ArgumentParser(
        description="Run MyTrack appearance ablations without the motion model."
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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "eval" / "ablation_appearance",
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
    """Evaluate the requested appearance modes and save one CSV summary."""
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ablation_appearance.csv"

    evaluator = MyTrackAblationEvaluator(
        data_root=args.source,
        yolo_weights=args.yolo_model,
        reid_weights=args.reid_model,
        output_dir=args.output_dir,
        motion_model_weights=REPO_ROOT / "weights" / "tracker" / "berrytracker_motion.pt",
        device=args.device,
    )

    rows_by_preset = load_existing_rows(csv_path)
    for preset_name, appearance_mode, override in APPEARANCE_PRESETS:
        if preset_name in rows_by_preset:
            LOGGER.info("Skipping preset=%s | appearance=%s | status=already_in_csv", preset_name, appearance_mode)
            continue

        if has_complete_tracking_outputs(args.output_dir, preset_name, evaluator.sequences):
            LOGGER.info(
                "Reusing tracking outputs for preset=%s | appearance=%s | action=trackeval_only",
                preset_name,
                appearance_mode,
            )
            metrics = evaluator.run_trackeval(preset_name)["combined"]
            row = build_result_row(preset_name, appearance_mode, metrics, fps=0.0)
            LOGGER.warning(
                "Preset=%s reused existing tracking outputs but had no cached FPS. Writing FPS=0.0.",
                preset_name,
            )
        else:
            LOGGER.info("Evaluating preset=%s | appearance=%s", preset_name, appearance_mode)
            metrics = evaluator.evaluate_preset(preset_name, "mytrack", override)["combined"]
            row = build_result_row(preset_name, appearance_mode, metrics)

        rows_by_preset[preset_name] = row
        save_rows(csv_path, rows_by_preset)
        LOGGER.info(
            "Result | appearance=%s | HOTA=%.4f | DetA=%.4f | AssA=%.4f | MOTA=%.4f | IDSW=%s | IDF1=%.4f | IDs=%s | GT_IDs=%s | FPS=%.2f",
            appearance_mode,
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

    save_rows(csv_path, rows_by_preset)
    LOGGER.info("Saved appearance ablation summary to %s", csv_path)


if __name__ == "__main__":
    main()
