#!/usr/bin/env python3
"""Generate concise statistics and visualizations for blueberry detection and MOT datasets."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DET_ROOT = Path("/home/wh1234_/data/Blueberry_coco_data")
DEFAULT_MOT_ROOT = Path("/home/wh1234_/data/blueberry_mot_stitched_walk")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "visualize" / "data"
OUTPUT_DPI = 600
FIG_SIZE = (7.2, 2.85)
STAGES = ["Flower", "Green", "Light Purple", "Blue"]
STAGE_COLORS = {"Flower": "#D3A253", "Green": "#72A47D", "Light Purple": "#9682BC", "Blue": "#6088B5"}
PLOT_BG = "#FBFCFE"
GRID_COLOR = "#D7E3F4"
SPINE_COLOR = "#3A4A5A"

LOGGER = logging.getLogger("data_visu")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize blueberry detection data and summarize blueberry MOT data.")
    parser.add_argument("--det-root", type=Path, default=DEFAULT_DET_ROOT, help="Blueberry detection dataset root.")
    parser.add_argument("--mot-root", type=Path, default=DEFAULT_MOT_ROOT, help="Blueberry MOT dataset root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--bins", type=int, default=120, help="2D histogram bins for density estimation.")
    return parser.parse_args()


def resolve_coco_annotation(det_root: Path) -> Path:
    candidates = [
        det_root / "annotations.json",
        det_root / "annotations" / "instances_train.json",
        det_root / "annotations" / "instances_val.json",
        det_root / "annotations" / "instances_test.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in sorted(det_root.rglob("*.json")):
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict) and {"images", "annotations"} <= payload.keys():
                return candidate
        except Exception:
            continue
    raise FileNotFoundError(f"No COCO annotation file found under: {det_root}")


def load_detection_stats(det_root: Path) -> tuple[np.ndarray, np.ndarray, list[int], Path]:
    ann_path = resolve_coco_annotation(det_root)
    with ann_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    image_sizes = {int(item["id"]): (float(item["width"]), float(item["height"])) for item in coco.get("images", [])}
    category_to_name = {int(item["id"]): str(item["name"]) for item in coco.get("categories", [])}
    name_to_index = {name.lower(): idx for idx, name in enumerate(STAGES)}

    widths: list[float] = []
    heights: list[float] = []
    stage_counts = [0] * len(STAGES)

    for ann in coco.get("annotations", []):
        image_size = image_sizes.get(int(ann["image_id"]))
        if image_size is None:
            continue
        img_w, img_h = image_size
        bbox = ann.get("bbox", [])
        if len(bbox) < 4 or img_w <= 0 or img_h <= 0:
            continue

        box_w = float(bbox[2])
        box_h = float(bbox[3])
        if box_w <= 0 or box_h <= 0:
            continue

        widths.append(box_w / img_w)
        heights.append(box_h / img_h)

        category_name = category_to_name.get(int(ann["category_id"]), "").lower()
        if category_name in name_to_index:
            stage_counts[name_to_index[category_name]] += 1

    if not widths:
        raise ValueError(f"No valid detection annotations found in: {ann_path}")

    LOGGER.info("Loaded detection annotations from %s", ann_path)
    LOGGER.info("Detection boxes: %d", len(widths))
    return np.asarray(widths), np.asarray(heights), stage_counts, ann_path


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PLOT_BG)
    # ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35, color=GRID_COLOR)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(1.1)


def compute_density(x: np.ndarray, y: np.ndarray, bins: int) -> np.ndarray:
    x_edges = np.linspace(0.0, max(1e-6, float(np.max(x)) * 1.02), bins + 1)
    y_edges = np.linspace(0.0, max(1e-6, float(np.max(y)) * 1.02), bins + 1)
    hist, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, hist.shape[0] - 1)
    y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, hist.shape[1] - 1)
    return hist[x_idx, y_idx]


def plot_detection_stats(
    widths: np.ndarray,
    heights: np.ndarray,
    stage_counts: list[int],
    output_path: Path,
    bins: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "normal",
            "font.size": 8,
            "axes.labelsize": 10,
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, gridspec_kw={"width_ratios": [1.15, 0.85]})
    ax0, ax1 = axes

    style_axis(ax0)
    density = compute_density(widths, heights, bins=bins)
    order = np.argsort(density)
    widths, heights, density = widths[order], heights[order], density[order]
    norm = Normalize(vmin=float(np.min(density)), vmax=float(np.max(density)))
    scatter = ax0.scatter(
        widths,
        heights,
        c=density,
        cmap="turbo",
        norm=norm,
        s=17,
        alpha=0.9,
        linewidths=0.0,
    )
    ax0.set_xlabel("Normalized width")
    ax0.set_ylabel("Normalized height")
    ax0.set_xlim(0.0, float(np.quantile(widths, 0.995)) * 1.08)
    ax0.set_ylim(0.0, float(np.quantile(heights, 0.995)) * 1.08)
    ax0.tick_params(axis="both", labelsize=8)
    ax0.text(0.02, 0.98, "a", transform=ax0.transAxes, va="top", ha="left", fontsize=16, fontweight="normal")
    cbar = fig.colorbar(scatter, ax=ax0, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    # cbar.set_label("density")

    style_axis(ax1)
    x = np.arange(len(STAGES))
    bars = ax1.bar(
        x,
        stage_counts,
        color=[STAGE_COLORS[stage] for stage in STAGES],
        width=0.72,
        edgecolor="white",
        linewidth=1.2,
    )
    ax1.set_xticks(x, STAGES)
    ax1.set_xlabel("Classes")
    ax1.set_ylabel("Number of instances")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax1.tick_params(axis="x", labelsize=9)
    ax1.tick_params(axis="y", labelsize=8)
    max_count = max(stage_counts)
    ax1.set_ylim(0, max_count * 1.16)
    ax1.text(0.02, 0.98, "b", transform=ax1.transAxes, va="top", ha="left", fontsize=16, fontweight="normal")
    for bar, value in zip(bars, stage_counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max_count * 0.018,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="normal",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved detection visualization to %s", output_path)
    LOGGER.info("Saved detection visualization to %s", pdf_path)


def discover_sequences(split_root: Path) -> list[Path]:
    if not split_root.exists():
        return []
    return sorted(path for path in split_root.iterdir() if path.is_dir() and (path / "gt" / "gt.txt").exists())


def read_seq_length(seq_dir: Path) -> int:
    seqinfo_path = seq_dir / "seqinfo.ini"
    if seqinfo_path.exists():
        config = configparser.ConfigParser()
        config.read(seqinfo_path, encoding="utf-8")
        seq_length = config.getint("Sequence", "seqLength", fallback=0)
        if seq_length > 0:
            return seq_length

    gt_path = seq_dir / "gt" / "gt.txt"
    max_frame = 0
    with gt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if parts and parts[0]:
                max_frame = max(max_frame, int(float(parts[0])))
    return max_frame


def iter_gt_rows(gt_path: Path) -> Iterable[tuple[int, int, float, float, int]]:
    with gt_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            parts = line.strip().split(",")
            if len(parts) < 8:
                LOGGER.warning("Skip malformed row: %s:%d", gt_path, line_no)
                continue
            mark = float(parts[6]) if len(parts) > 6 and parts[6] else 1.0
            if mark <= 0:
                continue
            yield (
                int(float(parts[1])),
                int(float(parts[7])),
                float(parts[4]),
                float(parts[5]),
                int(float(parts[0])),
            )


def empty_split_stats() -> dict:
    return {
        "sequences": 0,
        "frames": 0,
        "instances": 0,
        "trajectories": set(),
        "class_instances": defaultdict(int),
        "class_trajectories": defaultdict(set),
        "class_width_sum": defaultdict(float),
        "class_height_sum": defaultdict(float),
    }


def collect_mot_stats(mot_root: Path) -> dict[str, dict]:
    split_names = ["train", "test"]
    split_dirs = {split: discover_sequences(mot_root / split) for split in split_names}
    if not any(split_dirs.values()):
        raise FileNotFoundError(f"No MOT sequences found under train/test in: {mot_root}")

    stats = {split: empty_split_stats() for split in split_names}
    track_class_map: dict[tuple[str, str, int], str] = {}

    for split, seq_dirs in split_dirs.items():
        split_stat = stats[split]
        split_stat["sequences"] = len(seq_dirs)
        for seq_dir in seq_dirs:
            split_stat["frames"] += read_seq_length(seq_dir)
            gt_path = seq_dir / "gt" / "gt.txt"

            for track_id, class_id, width, height, _ in iter_gt_rows(gt_path):
                if class_id < 0 or class_id >= len(STAGES):
                    continue
                stage = STAGES[class_id]
                track_key = (split, seq_dir.name, track_id)
                prev_stage = track_class_map.get(track_key)
                if prev_stage is not None and prev_stage != stage:
                    LOGGER.warning(
                        "Inconsistent class assignment: split=%s seq=%s track=%d %s->%s",
                        split,
                        seq_dir.name,
                        track_id,
                        prev_stage,
                        stage,
                    )
                track_class_map[track_key] = stage

                split_stat["instances"] += 1
                split_stat["trajectories"].add(track_key)
                split_stat["class_instances"][stage] += 1
                split_stat["class_trajectories"][stage].add(track_key)
                split_stat["class_width_sum"][stage] += width
                split_stat["class_height_sum"][stage] += height

    return stats


def merge_totals(split_stats: dict[str, dict]) -> dict:
    total = empty_split_stats()
    for stats in split_stats.values():
        total["sequences"] += stats["sequences"]
        total["frames"] += stats["frames"]
        total["instances"] += stats["instances"]
        total["trajectories"].update(stats["trajectories"])
        for stage in STAGES:
            total["class_instances"][stage] += stats["class_instances"][stage]
            total["class_trajectories"][stage].update(stats["class_trajectories"][stage])
            total["class_width_sum"][stage] += stats["class_width_sum"][stage]
            total["class_height_sum"][stage] += stats["class_height_sum"][stage]
    return total


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def format_num(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def format_int_approx(value: float) -> str:
    return str(int(round(value)))


def build_mot_csv_rows(split_stats: dict[str, dict]) -> list[list[str]]:
    total_stats = merge_totals(split_stats)
    rows = [[
        "Split / Class",
        "Sequences",
        "Frames",
        "Instances",
        "Trajectories",
        "Avg. objects/frame",
        "Avg. track length",
        "Avg. width",
        "Avg. height",
    ]]
    rows.append(["Dataset-level", "", "", "", "", "", "", "", ""])

    for label, stats in [("Train", split_stats["train"]), ("Test", split_stats["test"]), ("Total", total_stats)]:
        trajectories = len(stats["trajectories"])
        rows.append(
            [
                label,
                format_num(stats["sequences"]),
                format_num(stats["frames"]),
                format_num(stats["instances"]),
                format_num(trajectories),
                format_int_approx(safe_div(stats["instances"], stats["frames"])),
                format_int_approx(safe_div(stats["instances"], trajectories)),
                "-",
                "-",
            ]
        )

    rows.append([])
    rows.append(["Class-level", "", "", "", "", "", "", "", ""])
    for stage in STAGES:
        instances = total_stats["class_instances"][stage]
        trajectories = len(total_stats["class_trajectories"][stage])
        rows.append(
            [
                stage,
                "-",
                "-",
                format_num(instances),
                format_num(trajectories),
                "-",
                format_int_approx(safe_div(instances, trajectories)),
                format_num(safe_div(total_stats["class_width_sum"][stage], instances)),
                format_num(safe_div(total_stats["class_height_sum"][stage], instances)),
            ]
        )
    return rows


def write_csv(rows: list[list[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    LOGGER.info("Saved MOT statistics to %s", output_path)


def main() -> None:
    setup_logging()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output directory: %s", args.output_dir)

    widths, heights, stage_counts, ann_path = load_detection_stats(args.det_root)
    plot_detection_stats(
        widths=widths,
        heights=heights,
        stage_counts=stage_counts,
        output_path=args.output_dir / "blueberry_detection_statistics.png",
        bins=args.bins,
    )
    LOGGER.info("Detection source: %s", ann_path)

    split_stats = collect_mot_stats(args.mot_root)
    csv_rows = build_mot_csv_rows(split_stats)
    write_csv(csv_rows, args.output_dir / "blueberry_mot_statistics.csv")
    LOGGER.info("Done")


if __name__ == "__main__":
    main()
