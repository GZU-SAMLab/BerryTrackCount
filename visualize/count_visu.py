#!/usr/bin/env python3
"""Visualize blueberry counting evaluation results as publication-ready figures."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "output" / "count_eval" / "stitched_walk_boxmot3"
DEFAULT_SUMMARY = EVAL_DIR / "count_eval_res.csv"
DEFAULT_BERRYTRACKER = EVAL_DIR / "berrytracker_res.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "visualize" / "count"
OUTPUT_DPI = 300

MODES = ["id", "line", "area"]
STAGES = ["Flower", "Green", "Light Purple", "Blue"]
MODE_LABELS = {"id": "ID", "line": "LINE", "area": "AREA"}
TRACKER_LABELS = {
    "bytetrack": "ByteTrack",
    "strongsort": "StrongSORT",
    "boosttrack": "BoostTrack",
    "hybrid-sort": "Hybrid-SORT",
    "hybridsort": "Hybrid-SORT",
    "deep oc-sort": "Deep OC-SORT",
    "deepocsort": "Deep OC-SORT",
    "berrytracker": "BerryTracker(our)",
}

MODE_COLORS = {"GT": "#7B879D", "id": "#D69A78", "line": "#88B29D", "area": "#6F95BC"}
STAGE_COLORS = {"Flower": "#D3A253", "Green": "#72A47D", "Light Purple": "#9682BC", "Blue": "#6088B5"}
REFERENCE_BAR_COLOR = "#BBD7F0"
REFERENCE_HIGHLIGHT_COLOR = "#35C3E6"
REGRESSION_OTHER_COLOR = "#6F95BC"
REGRESSION_BERRY_COLOR = "#35C3E6"
BAR_ALPHA = 0.78
SPINE_COLOR = "#4A5568"
GRID_COLOR = "#E2E8F0"
BORDER_COLOR = "#D6DCE4"

LOGGER = logging.getLogger("count_visu")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize blueberry counting evaluation results.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Tracker summary result CSV.")
    parser.add_argument("--berrytracker", type=Path, default=DEFAULT_BERRYTRACKER, help="BerryTracker class result CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output figures.")
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.framealpha": 0.95,
            "figure.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, grid: bool = False, grid_axis: str = "y") -> None:
    if grid:
        ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.65, alpha=0.85)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.9)


def tracker_label(name: str) -> str:
    return TRACKER_LABELS.get(name.lower(), name)


def decode_json_columns(payload: str, names: list[str], path: Path) -> dict[str, dict[str, Any]]:
    decoder = json.JSONDecoder()
    values: dict[str, dict[str, Any]] = {}
    position = 0
    for name in names:
        while position < len(payload) and payload[position] in " ,\r\n":
            position += 1
        try:
            values[name], position = decoder.raw_decode(payload, position)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot decode field '{name}' in {path}") from exc
    if payload[position:].strip(" ,\r\n"):
        raise ValueError(f"Unexpected trailing content in {path}")
    return values


def load_total_metrics(path: Path) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    trackers: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        next(handle, None)
        for line in handle:
            if not line.strip():
                continue
            tracker, mode, payload = line.rstrip("\r\n").split(",", maxsplit=2)
            mode = mode.lower()
            if mode not in MODES:
                continue
            if tracker not in metrics:
                trackers.append(tracker)
                metrics[tracker] = {}
            metrics[tracker][mode] = decode_json_columns(payload, ["total"], path)["total"]

    missing = [(tracker, mode) for tracker in trackers for mode in MODES if mode not in metrics[tracker]]
    if not trackers or missing:
        raise ValueError(f"Incomplete tracker-mode results in {path}: {missing}")
    LOGGER.info("Loaded %d trackers from %s", len(trackers), path)
    return trackers, metrics


def load_berrytracker_metrics(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        columns = handle.readline().rstrip("\r\n").split(",")[2:]
        for line in handle:
            if not line.strip():
                continue
            tracker, mode, payload = line.rstrip("\r\n").split(",", maxsplit=2)
            if tracker.lower() != "berrytracker" or mode.lower() not in MODES:
                continue
            values = decode_json_columns(payload, columns, path)
            metrics[mode.lower()] = {stage: values[stage] for stage in STAGES}

    missing = [(mode, stage) for mode in MODES for stage in STAGES if mode not in metrics or stage not in metrics[mode]]
    if missing:
        raise ValueError(f"Incomplete BerryTracker class results in {path}: {missing}")
    LOGGER.info("Loaded BerryTracker class metrics from %s", path)
    return metrics


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        output_path = output_dir / f"{stem}{suffix}"
        fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
        LOGGER.info("Saved figure to %s", output_path)
    plt.close(fig)


def fit_line(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    if true.size < 2 or np.allclose(true, true[0]):
        return 1.0, 0.0
    slope, intercept = np.polyfit(true, pred, 1)
    return float(slope), float(intercept)


def plot_regression_panel(
    ax: plt.Axes,
    values: dict[str, Any],
    color: str,
    title: str | None = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    stat_loc: tuple[float, float] = (0.05, 0.95),
    fit_color: str | None = None,
    scatter_color: str | None = None,
    scatter_size: float = 22,
    show_grid: bool = True,
) -> None:
    style_axis(ax, grid=show_grid, grid_axis="both")
    true = np.asarray(values["GT_values"], dtype=float)
    pred = np.asarray(values["Pred_values"], dtype=float)
    slope, intercept = fit_line(true, pred)
    limit = max(float(true.max()), float(pred.max())) * 1.08
    line_x = np.array([0.0, limit])
    fit_color = fit_color or color
    scatter_color = scatter_color or color

    ax.plot(line_x, slope * line_x + intercept, linewidth=1.35, color=fit_color, label="Linear Regression Fit")
    ax.plot(line_x, line_x, linestyle="--", linewidth=0.9, color="#9AA4B2", label="Ideal Agreement Line(y=x)")
    ax.scatter(true, pred, s=scatter_size, color=scatter_color, edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)
    ax.text(
        stat_loc[0],
        stat_loc[1],
        f"y = {slope:.2f}x {intercept:+.2f}\n$R^2$ = {values['R^2']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": BORDER_COLOR, "alpha": 0.92},
    )
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    if title:
        ax.set_title(title, pad=2, fontsize=8.3)
    ax.set_xlabel("Ground Truth Counts" if show_xlabel else "", fontsize=7.0)
    ax.set_ylabel("Predicted Counts" if show_ylabel else "", fontsize=7.0, labelpad=2)
    ax.tick_params(labelsize=6.6)


def plot_count_bars(trackers: list[str], metrics: dict[str, dict[str, dict[str, Any]]], output_dir: Path) -> None:
    x = np.arange(len(trackers), dtype=float) * 0.54
    width = 0.115
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    gt_values = [metrics[tracker][MODES[0]]["GT_sum"] for tracker in trackers]

    fig, ax = plt.subplots(figsize=(6.7, 2.45))
    style_axis(ax, grid=True)
    ax.bar(x + offsets[0], gt_values, width, label="GT", color=MODE_COLORS["GT"], edgecolor="white", linewidth=0.45, alpha=BAR_ALPHA)

    max_value = max(gt_values)
    for offset, mode in zip(offsets[1:], MODES):
        values = [metrics[tracker][mode]["Pred_sum"] for tracker in trackers]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
            edgecolor="white",
            linewidth=0.45,
            alpha=BAR_ALPHA,
        )
        max_value = max(max_value, max(values))
        for bar, tracker in zip(bars, trackers):
            ax.annotate(
                f"{metrics[tracker][mode]['Accuracy']:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7.4,
                color="#222222",
            )

    ax.set_xticks(x, [tracker_label(tracker) for tracker in trackers])
    ax.set_xlabel("")
    ax.set_ylabel("Counts")
    ax.set_yscale("symlog", linthresh=10000, linscale=0.9)
    ax.set_yticks([0, 5000, 10000, 20000, 50000])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, max_value * 2.05)
    ax.set_xlim(x[0] + offsets[0] - width * 1.25, x[-1] + offsets[-1] + width * 1.25)
    ax.legend(
        ncol=4,
        loc="upper right",
        bbox_to_anchor=(0.925, 0.985),
        edgecolor=BORDER_COLOR,
        handlelength=1.2,
        fontsize=7.6,
    )
    fig.tight_layout(pad=0.25)
    save_figure(fig, output_dir, "blueberry_count_by_tracker")


def plot_area_tracker_performance(
    trackers: list[str],
    metrics: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> None:
    colors = [REFERENCE_HIGHLIGHT_COLOR if tracker.lower() == "berrytracker" else REFERENCE_BAR_COLOR for tracker in trackers]
    labels = [tracker_label(tracker) for tracker in trackers]
    geh_values = [metrics[tracker]["area"]["GEH"] for tracker in trackers]
    rmse_values = [metrics[tracker]["area"]["RMSE"] for tracker in trackers]

    regression_colors = [REGRESSION_BERRY_COLOR if tracker.lower() == "berrytracker" else REGRESSION_OTHER_COLOR for tracker in trackers]

    fig = plt.figure(figsize=(6.85, 4.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.82, 2.18], hspace=0.40)
    top = outer[0].subgridspec(1, 2, wspace=0.30)
    bottom = outer[1].subgridspec(2, 3, hspace=0.50, wspace=0.34)
    ax_geh = fig.add_subplot(top[0, 0])
    ax_rmse = fig.add_subplot(top[0, 1])

    for ax, values, ylabel, letter in [(ax_geh, geh_values, "(a) GEH", " "), (ax_rmse, rmse_values, "(b) RMSE", " ")]:
        style_axis(ax, grid=True)
        x = np.arange(len(trackers))
        bars = ax.bar(x, values, color=colors, width=0.62, edgecolor="white", linewidth=0.5, alpha=BAR_ALPHA)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel, fontsize=8.0, labelpad=2)
        ax.set_ylim(0, max(values) * 1.22)
        ax.tick_params(axis="x", labelsize=6.8)
        ax.tick_params(axis="y", labelsize=7.4)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max(values) * 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(0.01, 0.95, letter, transform=ax.transAxes, ha="left", va="top", fontsize=8)

    regression_axes: list[plt.Axes] = []
    for idx, tracker in enumerate(trackers):
        ax = fig.add_subplot(bottom[idx // 3, idx % 3])
        regression_axes.append(ax)
        plot_regression_panel(
            ax=ax,
            values=metrics[tracker]["area"],
            color=regression_colors[idx],
            title=labels[idx],
            show_xlabel=idx // 3 == 1,
            show_ylabel=idx % 3 == 0,
            stat_loc=(0.05, 0.95),
            scatter_size=24,
            show_grid=False,
        )

    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.52, 0.665), handlelength=2.0, fontsize=7.4)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.085, top=0.98)
    geh_bounds = ax_geh.get_position()
    ax_geh.set_position([geh_bounds.x0 + 0.025, geh_bounds.y0, geh_bounds.width, geh_bounds.height])
    rmse_bounds = ax_rmse.get_position()
    ax_rmse.set_position([rmse_bounds.x0 - 0.025, rmse_bounds.y0, rmse_bounds.width, rmse_bounds.height])

    for idx, axis in enumerate(regression_axes):
        bounds = axis.get_position()
        col = idx % 3
        row = idx // 3
        x_shift = 0.026 + (0.022 if col == 0 else -0.038 if col == 2 else -0.006)
        y_shift = 0.044 if row == 1 else -0.012
        axis.set_position([bounds.x0 + x_shift, bounds.y0 + y_shift, bounds.width * 0.92, bounds.height * 0.88])

    regression_bounds = [axis.get_position() for axis in regression_axes]
    border_left = max(0.015, min(bounds.x0 for bounds in regression_bounds) - 0.060)
    border_right = min(0.978, max(bounds.x1 for bounds in regression_bounds) + 0.006)
    border_bottom = max(0.010, min(bounds.y0 for bounds in regression_bounds) - 0.085)
    border_top = 0.672
    fig.add_artist(
        Rectangle(
            (border_left, border_bottom),
            border_right - border_left,
            border_top - border_bottom,
            transform=fig.transFigure,
            fill=False,
            edgecolor="#C8CDD4",
            linewidth=0.9,
            linestyle=(0, (3, 3)),
            zorder=0,
        )
    )
    fig.text(border_left + 0.006, 0.652, "(c) R^2", ha="left", va="center", fontsize=8)
    save_figure(fig, output_dir, "area_tracker_count_performance")


def plot_berrytracker_mode_category_regression(
    metrics: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(7.1, 4.45))
    for row, mode in enumerate(MODES):
        for col, stage in enumerate(STAGES):
            ax = axes[row, col]
            plot_regression_panel(
                ax=ax,
                values=metrics[mode][stage],
                color=STAGE_COLORS[stage],
                title=stage if row == 0 else None,
                show_xlabel=row == len(MODES) - 1,
                show_ylabel=col == 0,
                stat_loc=(0.05, 0.95),
                scatter_size=18,
                show_grid=False,
            )
            if col == 0:
                ax.text(-0.32, 0.5, MODE_LABELS[mode], transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=7.8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.972), handlelength=2.0, fontsize=7.6)
    fig.tight_layout(rect=(0, 0, 1, 0.942), h_pad=0.32, w_pad=0.30)
    save_figure(fig, output_dir, "berrytracker_mode_category_regression")


def plot_berrytracker_mode_category_metrics(
    metrics: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> None:
    metric_names = ["Accuracy", "GEH", "RMSE"]
    fig, axes = plt.subplots(3, 1, figsize=(5.05, 3.65))
    x = np.arange(len(STAGES), dtype=float) * 0.39
    width = 0.095
    offsets = [-0.1, 0.0, 0.1]
    axis_limits = {"Accuracy": 116.0, "GEH": 13.0, "RMSE": 260.0}
    axis_ticks = {
        "Accuracy": [0, 20, 40, 60, 80, 100],
        "GEH": [0, 0.5, 1, 2, 4, 10],
        "RMSE": [0, 5, 10, 20, 50, 100, 200],
    }

    for ax, metric_name in zip(axes, metric_names):
        style_axis(ax, grid=True)
        for offset, mode in zip(offsets, MODES):
            values = [metrics[mode][stage][metric_name] for stage in STAGES]
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=MODE_LABELS[mode],
                color=MODE_COLORS[mode],
                edgecolor="white",
                linewidth=0.45,
                alpha=BAR_ALPHA,
            )
            for bar, value in zip(bars, values):
                ax.annotate(
                    f"{value:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.6,
                    clip_on=False,
                )
        ax.set_xticks(x, STAGES)
        ax.set_ylabel(metric_name, fontsize=8.8)
        ax.tick_params(axis="x", labelsize=8.2)
        ax.tick_params(axis="y", labelsize=8.0)
        if metric_name == "GEH":
            ax.set_yscale("symlog", linthresh=1.0, linscale=0.75)
        elif metric_name == "RMSE":
            ax.set_yscale("symlog", linthresh=10.0, linscale=0.85)
        ax.set_ylim(0, axis_limits[metric_name])
        ax.set_xlim(x[0] - 0.19, x[-1] + 0.19)
        ax.set_yticks(axis_ticks[metric_name])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}" if metric_name == "Accuracy" else f"{value:.1f}"))
        if ax is axes[0]:
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.955), handlelength=1.2, fontsize=8.2)
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=0.28)
    save_figure(fig, output_dir, "berrytracker_mode_category_metrics")


def main() -> None:
    setup_logging()
    apply_style()
    args = parse_args()
    trackers, total_metrics = load_total_metrics(args.summary)
    berrytracker_metrics = load_berrytracker_metrics(args.berrytracker)
    plot_count_bars(trackers, total_metrics, args.output_dir)
    plot_area_tracker_performance(trackers, total_metrics, args.output_dir)
    plot_berrytracker_mode_category_regression(berrytracker_metrics, args.output_dir)
    plot_berrytracker_mode_category_metrics(berrytracker_metrics, args.output_dir)


if __name__ == "__main__":
    main()
