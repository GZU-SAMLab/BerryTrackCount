#!/usr/bin/env python3
# Purpose: visualize real-scene BerryTracker AREA counting results with grouped stacked bars and pie charts.

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "output" / "count_apply" / "area_count.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "visualize" / "count_apply_visu"
OUTPUT_DPI = 300
STAGES = ["Flower", "Green", "Light Purple", "Blue"]
STAGE_COLORS = {
    "Flower": "#D7A64B",
    "Green": "#5E9F6E",
    "Light Purple": "#9A85C6",
    "Blue": "#4E79A7",
}
SPINE_COLOR = "#4A5568"

LOGGER = logging.getLogger("count_apply")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize BerryTracker AREA real-scene counting results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="AREA counting CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output figures.")
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.labelweight": "normal",
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.framealpha": 0.95,
            "figure.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(1.0)


def parse_date(video_name: str) -> str:
    raw_date = video_name[:8]
    return datetime.strptime(raw_date, "%Y%m%d").strftime("%Y-%m-%d")


def parse_video_label(video_name: str) -> str:
    suffix = video_name[8:].strip()
    if not suffix:
        return video_name
    normalized = suffix[:1].upper() + suffix[1:]
    return normalized.replace("Block", "B")


def load_counts(path: Path) -> dict[str, list[dict[str, object]]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    counts: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(["video_name", *STAGES]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            date = parse_date(row["video_name"])
            counts[date].append(
                {
                    "video": parse_video_label(row["video_name"]),
                    "counts": {stage: int(row[stage]) for stage in STAGES},
                }
            )

    if len(counts) != 4:
        LOGGER.warning("Expected 4 dates, loaded %d dates from %s", len(counts), path)
    LOGGER.info("Loaded AREA counts from %s", path)
    return dict(sorted(counts.items()))


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        output_path = output_dir / f"{stem}{suffix}"
        fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
        LOGGER.info("Saved figure to %s", output_path)
    plt.close(fig)


def plot_count_distribution(counts: dict[str, list[dict[str, object]]], output_dir: Path) -> None:
    dates = list(counts)
    date_labels = [datetime.strptime(date, "%Y-%m-%d").strftime("%m-%d") for date in dates]
    video_rows = [(date, row) for date in dates for row in counts[date]]
    bar_values = np.asarray(
        [[row["counts"][stage] for stage in STAGES] for _, row in video_rows],
        dtype=float,
    )
    pie_values = np.asarray(
        [
            [sum(row["counts"][stage] for row in counts[date]) for stage in STAGES]
            for date in dates
        ],
        dtype=float,
    )
    bar_totals = bar_values.sum(axis=1)
    colors = [STAGE_COLORS[stage] for stage in STAGES]

    fig = plt.figure(figsize=(6.6, 3.45))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.34, 1.0], wspace=-0.025)
    ax_bar = fig.add_subplot(grid[:, 0])
    pie_grid = grid[:, 1].subgridspec(2, 2, wspace=-0.2, hspace=0.34)
    pie_axes = [fig.add_subplot(pie_grid[idx // 2, idx % 2]) for idx in range(len(dates))]

    style_axis(ax_bar)
    group_step = 0.26
    bar_gap = 0.1
    bar_width = 0.08
    centers = np.arange(len(dates), dtype=float) * group_step
    x = np.concatenate([center + np.array([-bar_gap / 2.0, bar_gap / 2.0]) for center in centers])
    bottom = np.zeros(len(video_rows))
    for idx, stage in enumerate(STAGES):
        ax_bar.bar(
            x,
            bar_values[:, idx],
            bottom=bottom,
            width=bar_width,
            label=stage,
            color=colors[idx],
            edgecolor="white",
            linewidth=0.55,
        )
        bottom += bar_values[:, idx]

    for xpos, total in zip(x, bar_totals):
        ax_bar.text(xpos, total + bar_totals.max() * 0.012, f"{int(total):,}", ha="center", va="bottom", fontsize=6.5)
    for xpos, (_, row) in zip(x, video_rows):
        ax_bar.text(
            xpos,
            -0.035,
            row["video"],
            transform=ax_bar.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.5,
            clip_on=False,
        )

    ax_bar.set_xticks(centers, date_labels, rotation=0, ha="center")
    ax_bar.tick_params(axis="x", labelsize=8, pad=13)
    ax_bar.tick_params(axis="y", labelsize=8)
    ax_bar.set_ylabel("Counts")
    ax_bar.set_xlim(x.min() - bar_width * 1.2, x.max() + bar_width * 1.2)
    ax_bar.set_ylim(0, bar_totals.max() * 1.12)
    ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    handles, labels = ax_bar.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
        fontsize=8,
        columnspacing=1.2,
        handlelength=1.2,
        handletextpad=0.4,
    )

    for idx, (date_label, ax_pie) in enumerate(zip(date_labels, pie_axes)):
        _, _, autotexts = ax_pie.pie(
            pie_values[idx],
            colors=colors,
            startangle=95,
            counterclock=False,
            autopct=lambda pct: f"{pct:.1f}%",
            pctdistance=1.17,
            radius=1.14,
            wedgeprops={"edgecolor": "white", "linewidth": 0.9},
            textprops={"fontsize": 6.5, "color": "#1F2937"},
        )
        ax_pie.set_aspect("equal")
        ax_pie.text(0.5, -0.16, date_label, transform=ax_pie.transAxes, ha="center", va="top", fontsize=7.5, color="#2D3748")

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.2, top=0.86)
    save_figure(fig, output_dir, "berrytracker_area_count_apply")


def main() -> None:
    setup_logging()
    args = parse_args()
    apply_style()
    counts = load_counts(args.input)
    plot_count_distribution(counts, args.output_dir)


if __name__ == "__main__":
    main()
