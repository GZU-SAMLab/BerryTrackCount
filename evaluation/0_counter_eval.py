#!/usr/bin/env python3
"""Evaluate tracker counting results on blueberry videos with id/line/area counters."""

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
GT_DEFAULT = REPO_ROOT / "dataset" / "mot_Count_GT" / "stitched_walk_count_GT.csv"
PRED_DIR_DEFAULT = REPO_ROOT / "output" / "count" / "stitched_walk_boxmot3"
OUT_DEFAULT = REPO_ROOT / "output" / "count_eval" / "stitched_walk_boxmot3" / "counter_eval.csv"
ACC_OUT_DEFAULT = REPO_ROOT / "output" / "count_eval" / "stitched_walk_boxmot3" / "counter_accuracy_by_sequence.csv"
COUNTER_FILES = {
    "id": "id_count.csv",
    "line": "line_count.csv",
    "area": "area_count.csv",
}
STAGES = ["Flower", "Green", "Light Purple", "Blue", "total"]
ACC_FIELDS = [f"{stage}_acc" for stage in STAGES]

LOGGER = logging.getLogger("counter_eval")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_gt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["video_name"] = df["video_name"].astype(str)
    return df[["video_name", *STAGES]]


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["video_name"] = df["video_name"].astype(str)
    if "fps" not in df.columns:
        df["fps"] = 0.0
    return df[["video_name", "tracker", "fps", *STAGES]]


def calc_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, 1.0, y_true)
    scores = np.where(y_true == 0, (y_pred == 0).astype(float), 1.0 - np.abs(y_pred - y_true) / denom)
    return float(np.mean(np.clip(scores, 0.0, 1.0)) * 100.0)


def calc_geh(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = y_true + y_pred
    geh = np.where(denom == 0, 0.0, np.sqrt(2.0 * (y_pred - y_true) ** 2 / denom))
    return float(np.mean(geh))


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    if math.isclose(ss_tot, 0.0, abs_tol=1e-12):
        return 1.0 if math.isclose(ss_res, 0.0, abs_tol=1e-12) else 0.0
    return 1.0 - ss_res / ss_tot


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, fps: float) -> str:
    metrics = {
        "Accuracy": round(calc_accuracy(y_true, y_pred), 4),
        "GEH": round(calc_geh(y_true, y_pred), 4),
        "R^2": round(calc_r2(y_true, y_pred), 4),
        "RMSE": round(calc_rmse(y_true, y_pred), 4),
        "FPS": round(float(fps), 2),
    }
    return json.dumps(metrics, ensure_ascii=False)


def single_accuracy(gt: float, pred: float) -> float:
    if math.isclose(gt, 0.0, abs_tol=1e-12):
        return 100.0 if math.isclose(pred, 0.0, abs_tol=1e-12) else 0.0
    return max(0.0, 1.0 - abs(pred - gt) / abs(gt)) * 100.0


def evaluate_counter(gt_df: pd.DataFrame, pred_df: pd.DataFrame, counter: str) -> List[Dict[str, str]]:
    merged = pred_df.merge(gt_df, on="video_name", how="inner", suffixes=("_pred", "_gt"))
    rows: List[Dict[str, str]] = []

    for tracker, tracker_df in merged.groupby("tracker", sort=True):
        row = {"tracker": str(tracker), "counter": counter}
        fps = tracker_df["fps"].astype(float).mean()
        for stage in STAGES:
            y_true = tracker_df[f"{stage}_gt"].to_numpy(dtype=float)
            y_pred = tracker_df[f"{stage}_pred"].to_numpy(dtype=float)
            row[stage] = metric_dict(y_true, y_pred, fps)
        rows.append(row)

    return rows


def build_sequence_accuracy_rows(gt_df: pd.DataFrame, pred_df: pd.DataFrame, counter: str) -> List[Dict[str, object]]:
    merged = pred_df.merge(gt_df, on="video_name", how="inner", suffixes=("_pred", "_gt"))
    rows: List[Dict[str, object]] = []

    for _, data in merged.sort_values(["video_name", "tracker"]).iterrows():
        row: Dict[str, object] = {
            "video_name": data["video_name"],
            "tracker": str(data["tracker"]),
            "counter": counter,
        }
        for stage in STAGES:
            pred = float(data[f"{stage}_pred"])
            gt = float(data[f"{stage}_gt"])
            row[stage] = int(pred) if pred.is_integer() else round(pred, 4)
            row[f"{stage}_acc"] = round(single_accuracy(gt, pred), 4)
        rows.append(row)

    return rows


def evaluate(gt_path: Path, pred_dir: Path, out_path: Path, acc_out_path: Path) -> None:
    LOGGER.info("Loading ground truth: %s", gt_path)
    gt_df = load_gt(gt_path)
    results: List[Dict[str, str]] = []
    acc_rows: List[Dict[str, object]] = []

    for counter, filename in COUNTER_FILES.items():
        pred_path = pred_dir / filename
        if not pred_path.exists():
            LOGGER.warning("Prediction file not found: %s", pred_path)
            continue
        LOGGER.info("Evaluating %s counter: %s", counter, pred_path)
        pred_df = load_pred(pred_path)
        results.extend(evaluate_counter(gt_df, pred_df, counter))
        acc_rows.extend(build_sequence_accuracy_rows(gt_df, pred_df, counter))

    results.sort(key=lambda x: (x["tracker"], x["counter"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["tracker", "counter", *STAGES])
        writer.writeheader()
        writer.writerows(results)
    LOGGER.info("Saved %d rows to %s", len(results), out_path)

    acc_rows.sort(key=lambda x: (x["video_name"], x["tracker"], x["counter"]))
    acc_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(acc_out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["video_name", "tracker", "counter", *STAGES, *ACC_FIELDS])
        writer.writeheader()
        writer.writerows(acc_rows)
    LOGGER.info("Saved %d sequence-accuracy rows to %s", len(acc_rows), acc_out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate blueberry counter results.")
    parser.add_argument("--gt", type=Path, default=GT_DEFAULT, help="Ground-truth csv path.")
    parser.add_argument("--pred-dir", type=Path, default=PRED_DIR_DEFAULT, help="Prediction directory.")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT, help="Output csv path.")
    parser.add_argument("--acc-out", type=Path, default=ACC_OUT_DEFAULT, help="Sequence accuracy csv path.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    evaluate(args.gt, args.pred_dir, args.out, args.acc_out)


if __name__ == "__main__":
    main()
