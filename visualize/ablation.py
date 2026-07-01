#!/usr/bin/env python3
"""Visualize ablation figures for counting and tracking experiments."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AREA_RATIO_INPUT = (
    REPO_ROOT / "output" / "count_eval" / "stitched_walk_boxmot3" / "area-ratio_abl_res.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "visualize" / "ablation"
OUTPUT_DPI = 300
COMBINED_FIGSIZE = (6.2, 2.55)

AREA_RATIO_METRICS = ["Accuracy", "GEH", "R^2", "RMSE", "FPS"]
HIGHER_IS_BETTER = {"Accuracy", "R^2", "FPS"}

GEO_ASSOC_METHODS = ["HIoU", "IoU", "EIoU", "CIoU", "HM-IoU", "HM-EIoU", "HM-CIoU"]
GEO_ASSOC_RESULTS = {
    "HOTA": [61.89, 74.84, 74.78, 74.85, 73.84, 74.97, 75.34],
    "MOTA": [59.19, 61.83, 61.69, 61.64, 61.61, 61.98, 62.07],
    "IDF1": [59.87, 76.45, 76.13, 76.13, 75.97, 77.08, 77.25],
    "IDSW": [16755, 3324, 4366, 4260, 3814, 3055, 2888],
}

COLORS = {
    "blue": "#4C72B0",
    "green": "#55A868",
    "red": "#C44E52",
    "purple": "#8172B3",
    "gold": "#CCB974",
    "cyan": "#64B5CD",
}
RADAR_COLORS = [
    COLORS["blue"],
    COLORS["green"],
    COLORS["red"],
    COLORS["purple"],
    COLORS["gold"],
    COLORS["cyan"],
]
GEO_ASSOC_COLORS = {
    "HOTA": COLORS["blue"],
    "MOTA": COLORS["green"],
    "IDF1": COLORS["purple"],
    "IDSW": COLORS["red"],
}
GEO_ASSOC_MARKERS = {"HOTA": "o", "MOTA": "s", "IDF1": "^", "IDSW": "D"}
GEO_ASSOC_X_STEP = 0.78
SPINE_COLOR = "#485464"
GRID_COLOR = "#D9DEE8"

LOGGER = logging.getLogger("ablation")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ablation figures.")
    parser.add_argument("--area-ratio-input", type=Path, default=DEFAULT_AREA_RATIO_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "normal",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.labelweight": "normal",
            "axes.titlesize": 8.5,
            "axes.titleweight": "normal",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.9,
            "figure.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_area_ratio_results(path: Path) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    values: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ratio = float(row["Area Ratio"])
            labels.append(f"{ratio:.0%}")
            values.append([float(row[metric].strip()) for metric in AREA_RATIO_METRICS])
    if not values:
        raise ValueError(f"No ablation rows found in {path}")
    LOGGER.info("Loaded %d area-ratio rows from %s", len(values), path)
    return labels, np.asarray(values, dtype=float)


def normalize_area_ratio(values: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(values, dtype=float)
    for idx, metric in enumerate(AREA_RATIO_METRICS):
        column = values[:, idx]
        best = float(np.max(column)) if metric in HIGHER_IS_BETTER else float(np.min(column))
        if np.allclose(column, best):
            normalized[:, idx] = 1.0
        elif metric in HIGHER_IS_BETTER:
            normalized[:, idx] = column / best if not np.isclose(best, 0.0) else 0.0
        else:
            normalized[:, idx] = np.divide(
                best,
                column,
                out=np.zeros_like(column),
                where=~np.isclose(column, 0.0),
            )
    return np.round(np.clip(normalized, 0.0, 1.0), 3)


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", colors="#2F3640", width=0.8, length=3.0)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.9)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for text in fig.findobj(match=Text):
        text.set_fontweight("normal")
    for fmt in formats:
        output_path = output_dir / f"{stem}.{fmt.lstrip('.')}"
        fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
        LOGGER.info("Saved figure to %s", output_path)
    plt.close(fig)


def draw_area_ratio_ablation_radar(
    ax: plt.Axes,
    labels: list[str],
    scores: np.ndarray,
    *,
    legend_loc: str = "center left",
    legend_bbox: tuple[float, float] | None = (1.02, 0.5),
    legend_ncol: int = 1,
) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, len(AREA_RATIO_METRICS), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]

    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles, AREA_RATIO_METRICS)
    ax.set_ylim(0.0, 1.0)
    radial_ticks = np.arange(0.2, 1.01, 0.2)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in radial_ticks], fontsize=6.8, color="#626B78")
    ax.grid(True, color=GRID_COLOR, linewidth=0.65)
    ax.spines["polar"].set_color(SPINE_COLOR)
    ax.spines["polar"].set_linewidth(0.9)

    for idx, (ratio_label, row) in enumerate(zip(labels, scores)):
        values = np.r_[row, row[0]]
        color = RADAR_COLORS[idx % len(RADAR_COLORS)]
        ax.plot(
            closed_angles,
            values,
            color=color,
            linewidth=1.35,
            marker="o",
            markersize=3.2,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            label=f"Area Ratio {ratio_label}",
        )
        ax.fill(closed_angles, values, color=color, alpha=0.10)

    ax.legend(
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        ncol=legend_ncol,
        frameon=False,
        handlelength=1.55,
        columnspacing=0.85,
        borderaxespad=0.0,
    )


def plot_area_ratio_ablation_radar(labels: list[str], scores: np.ndarray, output_dir: Path, formats: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 3.05), subplot_kw={"projection": "polar"})
    draw_area_ratio_ablation_radar(labels=labels, scores=scores, ax=ax)
    fig.tight_layout(pad=0.6)
    save_figure(fig, output_dir, "area_ratio_ablation_radar", formats)


def draw_geo_assoc_metric_ablation(
    ax_left: plt.Axes,
    *,
    legend_loc: str = "upper center",
    legend_bbox: tuple[float, float] | None = None,
    legend_borderaxespad: float = 0.0,
    highlight_alpha: float = 0.12,
) -> plt.Axes:
    x = np.arange(len(GEO_ASSOC_METHODS)) * GEO_ASSOC_X_STEP
    ax_right = ax_left.twinx()

    for metric in ("HOTA", "MOTA", "IDF1"):
        ax_left.plot(
            x,
            GEO_ASSOC_RESULTS[metric],
            color=GEO_ASSOC_COLORS[metric],
            marker=GEO_ASSOC_MARKERS[metric],
            linewidth=1.45,
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.55,
            label=metric,
        )

    ax_right.plot(
        x,
        GEO_ASSOC_RESULTS["IDSW"],
        color=GEO_ASSOC_COLORS["IDSW"],
        marker=GEO_ASSOC_MARKERS["IDSW"],
        linewidth=1.45,
        markersize=3.6,
        markeredgecolor="white",
        markeredgewidth=0.55,
        linestyle="--",
        label="IDSW",
    )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(GEO_ASSOC_METHODS, rotation=18, ha="right", rotation_mode="anchor")
    ax_left.set_xlabel("Geometric Association Metric")
    ax_left.set_ylabel("HOTA / MOTA / IDF1 (%)")
    ax_right.set_ylabel("IDSW (lower is better)")
    ax_left.set_ylim(57.0, 79.0)
    ax_right.set_ylim(2000, 18000)
    ax_left.set_xlim(x[0] - 0.38, x[-1] + 0.38)
    ax_left.margins(x=0.03)
    ax_left.grid(True, axis="y", color=GRID_COLOR, linewidth=0.65, alpha=0.9)
    ax_left.set_axisbelow(True)
    style_axis(ax_left)
    style_axis(ax_right)
    ax_right.spines["left"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    best_idx = GEO_ASSOC_METHODS.index("HM-CIoU")
    best_x = x[best_idx]
    ax_left.axvspan(
        best_x - GEO_ASSOC_X_STEP * 0.32,
        best_x + GEO_ASSOC_X_STEP * 0.32,
        color="#F2C94C",
        alpha=highlight_alpha,
        linewidth=0,
    )

    handles, legend_labels = ax_left.get_legend_handles_labels()
    right_handles, right_labels = ax_right.get_legend_handles_labels()
    ax_left.legend(
        handles + right_handles,
        legend_labels + right_labels,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        ncol=4,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.1,
        borderaxespad=legend_borderaxespad,
    )
    return ax_right


def plot_geo_assoc_metric_ablation(output_dir: Path, formats: list[str]) -> None:
    fig, ax_left = plt.subplots(figsize=(4.75, 2.75))
    draw_geo_assoc_metric_ablation(ax_left)
    fig.tight_layout(pad=0.45)
    save_figure(fig, output_dir, "geo_assoc_metric_ablation", formats)


def plot_ablation_combined(labels: list[str], scores: np.ndarray, output_dir: Path, formats: list[str]) -> None:
    fig = plt.figure(figsize=COMBINED_FIGSIZE)
    gridspec = fig.add_gridspec(1, 2, width_ratios=[1.58, 1.0], wspace=0.24)
    ax_geo = fig.add_subplot(gridspec[0, 0])
    ax_radar = fig.add_subplot(gridspec[0, 1], projection="polar")

    draw_geo_assoc_metric_ablation(
        ax_geo,
        legend_loc="lower center",
        legend_bbox=(0.5, 1.02),
        legend_borderaxespad=0.0,
        highlight_alpha=0.18,
    )
    draw_area_ratio_ablation_radar(
        ax_radar,
        labels,
        scores,
        legend_loc="upper center",
        legend_bbox=(0.5, -0.10),
        legend_ncol=2,
    )

    ax_geo.text(-0.14, 1.04, "(a)", transform=ax_geo.transAxes, ha="left", va="bottom", fontsize=8.5)
    ax_radar.text(-0.12, 1.04, "(b)", transform=ax_radar.transAxes, ha="left", va="bottom", fontsize=8.5)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.82, bottom=0.25)
    save_figure(fig, output_dir, "ablation_combined", formats)


def main() -> None:
    setup_logging()
    apply_style()
    args = parse_args()

    area_ratio_labels, area_ratio_values = load_area_ratio_results(args.area_ratio_input)
    area_ratio_scores = normalize_area_ratio(area_ratio_values)
    plot_area_ratio_ablation_radar(area_ratio_labels, area_ratio_scores, args.output_dir, args.formats)
    plot_geo_assoc_metric_ablation(args.output_dir, args.formats)
    plot_ablation_combined(area_ratio_labels, area_ratio_scores, args.output_dir, args.formats)
    LOGGER.info("Done | output=%s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
