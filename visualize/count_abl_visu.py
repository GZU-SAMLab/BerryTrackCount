#!/usr/bin/env python3
"""Visualize area-ratio ablation results for blueberry area counting."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "output" / "count_eval" / "stitched_walk_boxmot3" / "area-ratio_abl_res.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "visualize" / "count"
OUTPUT_DPI = 300
METRICS = ["Accuracy", "GEH", "R^2", "RMSE", "FPS"]
HIGHER_IS_BETTER = {"Accuracy", "R^2", "FPS"}
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
SPINE_COLOR = "#485464"
GRID_COLOR = "#D9DEE8"
HEATMAP_CMAP = LinearSegmentedColormap.from_list("cvpr_heat", ["#F7FBFF", "#C9DDF0", "#6F95BC", "#2F5E8E"])

LOGGER = logging.getLogger("count_abl_visu")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot normalized radar and heatmap charts for area-ratio ablations.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Area-ratio ablation CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_results(path: Path) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    values: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ratio = float(row["Area Ratio"])
            labels.append(f"{ratio:.0%}")
            values.append([float(row[metric].strip()) for metric in METRICS])
    if not values:
        raise ValueError(f"No ablation rows found in {path}")
    LOGGER.info("Loaded %d area-ratio rows from %s", len(values), path)
    return labels, np.asarray(values, dtype=float)


def normalize(values: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(values, dtype=float)
    for idx, metric in enumerate(METRICS):
        column = values[:, idx]
        best = float(np.max(column)) if metric in HIGHER_IS_BETTER else float(np.min(column))
        if np.allclose(column, best):
            normalized[:, idx] = 1.0
        elif metric in HIGHER_IS_BETTER:
            normalized[:, idx] = column / best if not np.isclose(best, 0.0) else 0.0
        else:
            normalized[:, idx] = np.divide(best, column, out=np.zeros_like(column), where=~np.isclose(column, 0.0))
    return np.round(np.clip(normalized, 0.0, 1.0), 3)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        output_path = output_dir / f"{stem}{suffix}"
        fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
        LOGGER.info("Saved figure to %s", output_path)
    plt.close(fig)


def plot_radar(labels: list[str], scores: np.ndarray, output_dir: Path) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, len(METRICS), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]

    fig, ax = plt.subplots(figsize=(4.8, 4.35), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles, METRICS, fontweight="semibold")
    ax.set_ylim(0.0, 1.0)
    radial_ticks = np.arange(0.1, 1.01, 0.1)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in radial_ticks], fontsize=7, color="#626B78")
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.spines["polar"].set_color(SPINE_COLOR)
    ax.spines["polar"].set_linewidth(1.2)

    for idx, (ratio_label, row) in enumerate(zip(labels, scores)):
        values = np.r_[row, row[0]]
        color = COLORS[idx % len(COLORS)]
        ax.plot(
            closed_angles,
            values,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4.0,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=f"Area Ratio {ratio_label}",
        )
        ax.fill(closed_angles, values, color=color, alpha=0.12)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "area_ratio_ablation_radar")


def format_raw_value(value: float, metric: str) -> str:
    return f"{value:.2f}" if metric != "R^2" else f"{value:.3f}".rstrip("0").rstrip(".")


def plot_heatmap(labels: list[str], values: np.ndarray, scores: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.35))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    padding = max((max_score - min_score) * 0.04, 0.005)
    vmin = max(0.0, min_score - padding)
    vmax = min(1.0, max_score + padding)
    x_edges = np.arange(scores.shape[1] + 1) - 0.5
    y_edges = np.arange(scores.shape[0] + 1) - 0.5
    mesh = ax.pcolormesh(x_edges, y_edges, scores, cmap=HEATMAP_CMAP, vmin=vmin, vmax=vmax, shading="flat")
    ax.set_xlim(-0.5, scores.shape[1] - 0.5)
    ax.set_ylim(scores.shape[0] - 0.5, -0.5)
    ax.set_xticks(np.arange(len(METRICS)), METRICS, fontweight="semibold")
    ax.set_yticks(np.arange(len(labels)), [f"{label}" for label in labels])
    ax.set_xlabel("Metrics", fontweight="semibold")
    ax.set_ylabel("Area Ratio", fontweight="semibold")
    ax.tick_params(length=0)

    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            threshold = vmin + (vmax - vmin) * 0.62
            text_color = "white" if scores[i, j] >= threshold else "#202020"
            label = f"{format_raw_value(values[i, j], METRICS[j])}\n({scores[i, j]:.3f})"
            ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=8)

    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(1.0)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Normalized Score", fontweight="semibold")
    cbar.outline.set_edgecolor(SPINE_COLOR)
    fig.tight_layout()
    save_figure(fig, output_dir, "area_ratio_ablation_heatmap")


def main() -> None:
    setup_logging()
    apply_style()
    args = parse_args()
    labels, values = load_results(args.input)
    scores = normalize(values)
    plot_radar(labels, scores, args.output_dir)
    plot_heatmap(labels, values, scores, args.output_dir)


if __name__ == "__main__":
    main()
