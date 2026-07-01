#!/usr/bin/env python3
# Purpose: visualize tracker appearance features and association costs for tracking explainability figures.

import argparse
import importlib
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(REPO_ROOT / "trackers") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "trackers"))

from loguru import logger as loguru_logger

loguru_logger.disable("boxmot.motion.cmc.ecc")

from boxmot.tracker_zoo import create_tracker
from boxmot.utils.association import cal_cost_matrix, compute_aw_max_metric, speed_direction_batch
from tools.video_detector import VideoDetector


LOGGER = logging.getLogger("tracker_visu2")
DATA_ROOT = Path(r"/home/wh1234_/data/blueberry_mot_stitched_walk/test")
SEQUENCES = ("Blueberry-Test-8", "Blueberry-Test-15")
FEATURE_SEQUENCES = ("Blueberry-Test-8", "Blueberry-Test-15")
OUTPUT_DIR = REPO_ROOT / "output" / "visualize" / "tracker_visu2"
CONFIG_DIR = REPO_ROOT / "configs" / "trackers"
DETECTOR_WEIGHTS = REPO_ROOT / "weights" / "berrydet_s.pt"
REID_WEIGHTS = REPO_ROOT / "weights" / "osnet_ain_x1_0_blueberry.pt"
TRACKERS = ("deepocsort", "mytrack")
DISPLAY_NAMES = {"deepocsort": "Deep OC-SORT", "mytrack": "BerryTracker"}
STAGE_NAMES = {0: "Flower", 1: "Green", 2: "Light Purple", 3: "Blue"}
STAGE_COLORS = {
    0: "#DDAA33",
    1: "#228833",
    2: "#AA4499",
    3: "#4477AA",
}
CVPR_ID_PALETTE = (
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#000000",
    "#88CCEE",
    "#CC6677",
    "#DDCC77",
    "#117733",
    "#332288",
    "#AA4499",
    "#44AA99",
    "#999933",
    "#882255",
    "#661100",
    "#6699CC",
    "#888888",
)
TRACKER_COLORS = {"deepocsort": "#7B879D", "mytrack": "#4477AA"}
SPINE_COLOR = "#4B5563"
GRID_COLOR = "#D7DCE2"
TEXT_COLOR = "#1F2937"
OUTPUT_DPI = 600
PREVIEW_DPI = 600
TSNE_FIGSIZE = (6.9, 3.15)
SIMILARITY_FIGSIZE = (6.9, 2.85)
MARGIN_FIGSIZE = (3.35, 2.75)
MATRIX_FIGSIZE = (3.55, 3.10)
COMBINED_MATRIX_FIGSIZE = (6.9, 6.25)
MATRIX_FRAMES = (10, 20)


@dataclass
class FeatureSample:
    tracker: str
    sequence: str
    frame_id: int
    track_id: int
    stage: int
    feature: np.ndarray

    @property
    def identity(self) -> Tuple[str, int]:
        return self.sequence, self.track_id

    @property
    def identity_label(self) -> str:
        seq_tag = self.sequence.replace("Blueberry-Test-", "T")
        return f"{seq_tag}:ID {self.track_id}"


@dataclass
class MatrixSample:
    tracker: str
    sequence: str
    frame_id: int
    final_cost: np.ndarray
    iou: np.ndarray
    angle: np.ndarray
    emb: np.ndarray
    track_ids: List[int]
    det_ids: List[int]


class AssociationProbe:
    """Capture first-stage association matrices without editing tracker code."""

    def __init__(self, tracker_name: str, tracker, max_samples: int, min_size: int, stride: int, target_frames: Sequence[int]) -> None:
        self.tracker_name = tracker_name
        self.tracker = tracker
        self.max_samples = max_samples
        self.min_size = min_size
        self.stride = max(1, stride)
        self.target_frames = {int(frame) for frame in target_frames if int(frame) > 0}
        self.samples: List[MatrixSample] = []
        self.sequence = ""
        self.frame_id = 0
        self._module = None
        self._original = None
        if tracker_name == "mytrack":
            self._original = tracker._associate_tracks
            tracker._associate_tracks = self._wrapped_mytrack_associate
        elif tracker_name == "deepocsort":
            self._module = importlib.import_module("boxmot.trackers.deepocsort.deepocsort")
            self._original = self._module.associate
            self._module.associate = self._wrapped_deepocsort_associate
        else:
            raise ValueError(f"Unsupported association probe tracker: {tracker_name}")

    def close(self) -> None:
        if self.tracker_name == "mytrack" and self._original is not None:
            self.tracker._associate_tracks = self._original
        if self.tracker_name == "deepocsort" and self._module is not None and self._original is not None:
            self._module.associate = self._original

    def set_frame(self, sequence: str, frame_id: int) -> None:
        self.sequence = sequence
        self.frame_id = frame_id

    def _wrapped_mytrack_associate(self, dets, trks, dets_embs, trk_embs, velocities, k_observations, stage1_emb_cost):
        if self._should_capture(dets, trks):
            sample = self._build_mytrack_sample(dets, trks, dets_embs, trk_embs, velocities, k_observations, stage1_emb_cost)
            if sample is not None:
                self.samples.append(sample)
        return self._original(dets, trks, dets_embs, trk_embs, velocities, k_observations, stage1_emb_cost)

    def _wrapped_deepocsort_associate(
        self,
        detections,
        trackers,
        asso_func,
        iou_threshold,
        velocities,
        previous_obs,
        vdc_weight,
        w,
        h,
        emb_cost=None,
        w_assoc_emb=None,
        aw_off=None,
        aw_param=None,
    ):
        if self._should_capture(detections, trackers):
            sample = self._build_deepocsort_sample(
                detections,
                trackers,
                asso_func,
                velocities,
                previous_obs,
                vdc_weight,
                emb_cost,
                w_assoc_emb,
                aw_off,
                aw_param,
            )
            if sample is not None:
                self.samples.append(sample)
        return self._original(
            detections,
            trackers,
            asso_func,
            iou_threshold,
            velocities,
            previous_obs,
            vdc_weight,
            w,
            h,
            emb_cost,
            w_assoc_emb,
            aw_off,
            aw_param,
        )

    def _should_capture(self, dets: np.ndarray, trks: np.ndarray) -> bool:
        if len(self.samples) >= self.max_samples:
            return False
        if self.target_frames:
            should_sample = self.frame_id in self.target_frames
        else:
            should_sample = self.frame_id % self.stride == 0
        if not should_sample:
            return False
        return min(len(dets), len(trks)) >= self.min_size

    def _build_mytrack_sample(
        self,
        dets: np.ndarray,
        trks: np.ndarray,
        dets_embs: np.ndarray,
        trk_embs: np.ndarray,
        velocities: np.ndarray,
        previous_obs: np.ndarray,
        stage1_emb_cost: Optional[np.ndarray],
    ) -> Optional[MatrixSample]:
        if len(dets) == 0 or len(trks) == 0:
            return None
        try:
            iou = np.asarray(self.tracker.asso_func(dets[:, 0:5], trks), dtype=np.float32)
            angle = self._angle_cost(dets, trks, velocities, previous_obs, self.tracker.inertia)
            emb = self._embedding_cost(dets, trks, dets_embs, trk_embs, stage1_emb_cost, iou)
            final_cost = -(iou + angle + emb)
        except Exception as exc:
            LOGGER.debug("Skip matrix capture | sequence=%s | frame=%d | error=%s", self.sequence, self.frame_id, exc)
            return None

        track_ids = [int(track.id) for track in self.tracker.active_tracks[: trks.shape[0]]]
        det_ids = [int(i) for i in range(dets.shape[0])]
        return MatrixSample(self.tracker_name, self.sequence, self.frame_id, final_cost.T, iou.T, angle.T, emb.T, track_ids, det_ids)

    def _build_deepocsort_sample(
        self,
        detections: np.ndarray,
        trackers: np.ndarray,
        asso_func,
        velocities: np.ndarray,
        previous_obs: np.ndarray,
        vdc_weight: float,
        emb_cost: Optional[np.ndarray],
        w_assoc_emb: Optional[float],
        aw_off: Optional[bool],
        aw_param: Optional[float],
    ) -> Optional[MatrixSample]:
        if len(detections) == 0 or len(trackers) == 0:
            return None
        try:
            iou = np.asarray(asso_func(detections, trackers), dtype=np.float32)
            angle = self._angle_cost(detections, trackers, velocities, previous_obs, vdc_weight)
            emb = self._deepocsort_embedding_cost(emb_cost, iou, w_assoc_emb, aw_off, aw_param)
            final_cost = -(iou + angle + emb)
        except Exception as exc:
            LOGGER.debug("Skip matrix capture | tracker=%s | sequence=%s | frame=%d | error=%s", self.tracker_name, self.sequence, self.frame_id, exc)
            return None

        track_ids = [int(track.id) for track in self.tracker.active_tracks[: trackers.shape[0]]]
        det_ids = [int(i) for i in range(detections.shape[0])]
        return MatrixSample(self.tracker_name, self.sequence, self.frame_id, final_cost.T, iou.T, angle.T, emb.T, track_ids, det_ids)

    def _angle_cost(
        self,
        dets: np.ndarray,
        trks: np.ndarray,
        velocities: np.ndarray,
        previous_obs: np.ndarray,
        vdc_weight: float,
    ) -> np.ndarray:
        y, x = speed_direction_batch(dets[:, 0:5], previous_obs)
        inertia_y, inertia_x = velocities[:, 0], velocities[:, 1]
        inertia_y = np.repeat(inertia_y[:, np.newaxis], y.shape[1], axis=1)
        inertia_x = np.repeat(inertia_x[:, np.newaxis], x.shape[1], axis=1)
        diff_angle_cos = np.clip(inertia_x * x + inertia_y * y, a_min=-1, a_max=1)
        diff_angle = (np.pi / 2.0 - np.abs(np.arccos(diff_angle_cos))) / np.pi
        valid_mask = np.ones(previous_obs.shape[0])
        valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
        valid_mask = np.repeat(valid_mask[:, np.newaxis], x.shape[1], axis=1)
        scores = np.repeat(dets[:, -1][:, np.newaxis], trks.shape[0], axis=1)
        return ((valid_mask * diff_angle) * vdc_weight).T * scores

    def _embedding_cost(
        self,
        dets: np.ndarray,
        trks: np.ndarray,
        dets_embs: np.ndarray,
        trk_embs: np.ndarray,
        stage1_emb_cost: Optional[np.ndarray],
        iou: np.ndarray,
    ) -> np.ndarray:
        valid_embs = (
            dets_embs is not None
            and trk_embs is not None
            and dets_embs.ndim == 2
            and trk_embs.ndim == 2
            and dets_embs.shape[0] == dets.shape[0]
            and trk_embs.shape[0] == trks.shape[0]
            and dets_embs.shape[1] == trk_embs.shape[1]
        )
        if self.tracker.aarm_open and valid_embs:
            _, emb = cal_cost_matrix(dets_embs, trk_embs, self.tracker.aarm_open)
            return emb.T * self.tracker.w_association_emb
        if stage1_emb_cost is None:
            return np.zeros_like(iou)
        emb = np.asarray(stage1_emb_cost, dtype=np.float32).copy()
        emb[iou <= 0] = 0
        if self.tracker.aw_off:
            return emb * self.tracker.w_association_emb
        return compute_aw_max_metric(emb, self.tracker.w_association_emb, bottom=self.tracker.aw_param)

    @staticmethod
    def _deepocsort_embedding_cost(
        emb_cost: Optional[np.ndarray],
        iou: np.ndarray,
        w_assoc_emb: Optional[float],
        aw_off: Optional[bool],
        aw_param: Optional[float],
    ) -> np.ndarray:
        if emb_cost is None:
            return np.zeros_like(iou)
        emb = np.asarray(emb_cost, dtype=np.float32).copy()
        emb[iou <= 0] = 0
        if aw_off:
            return emb * float(w_assoc_emb or 0.0)
        return compute_aw_max_metric(emb, float(w_assoc_emb or 0.0), bottom=float(aw_param or 0.5))


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "figure.dpi": PREVIEW_DPI,
            "savefig.dpi": OUTPUT_DPI,
            "font.size": 8.0,
            "axes.titlesize": 9.6,
            "axes.titleweight": "normal",
            "axes.labelsize": 8.0,
            "axes.labelweight": "normal",
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.8,
            "legend.title_fontsize": 7.0,
            "font.weight": "normal",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
        }
    )


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.set_facecolor("white")
    ax.tick_params(axis="both", colors=TEXT_COLOR, pad=1.5)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.75)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.45, alpha=0.78)


def parse_figsize(values: Sequence[float], fallback: Tuple[float, float]) -> Tuple[float, float]:
    if len(values) != 2:
        return fallback
    width, height = float(values[0]), float(values[1])
    if width <= 0 or height <= 0:
        return fallback
    return width, height


def ensure_file(path: Path, label: str) -> None:
    if not path.exists() or path.is_file() and path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found or empty: {path}")


def sequence_name_candidates(sequence: str) -> List[str]:
    candidates = [sequence]
    prefix, sep, suffix = sequence.rpartition("-")
    if sep and suffix.isdigit():
        number = int(suffix)
        candidates.extend([f"{prefix}-{number}", f"{prefix}-{number:02d}"])
    return list(dict.fromkeys(candidates))


def resolve_sequence_dir(data_root: Path, sequence: str) -> Path:
    for candidate in sequence_name_candidates(sequence):
        sequence_dir = data_root / candidate
        if sequence_dir.exists():
            return sequence_dir
    return data_root / sequence


def load_tracker_args(config_dir: Path, tracker_name: str) -> Dict[str, object]:
    config_path = config_dir / f"{tracker_name}.yaml"
    ensure_file(config_path, "Tracker config")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    tracker_args = {key: value["default"] for key, value in config.items()}
    if tracker_name == "deepocsort":
        tracker_args["embedding_off"] = False
        tracker_args["aw_off"] = False
    if tracker_name == "mytrack":
        tracker_args["embedding_off"] = False
        tracker_args["aw_off"] = False
        tracker_args["aarm_open"] = True
    return tracker_args


def build_tracker(args: argparse.Namespace, tracker_name: str):
    tracker_args = load_tracker_args(args.tracker_config_dir, tracker_name)
    reid_weights = args.ours_reid_weights if tracker_name == "mytrack" else args.baseline_reid_weights
    return create_tracker(
        tracker_type=tracker_name,
        tracker_config=str(args.tracker_config_dir / f"{tracker_name}.yaml"),
        reid_weights=reid_weights,
        device=args.tracker_device,
        half=args.half,
        per_class=False,
        evolve_param_dict=tracker_args,
    )


def load_frame_paths(sequence_dir: Path) -> List[Path]:
    img_dir = sequence_dir / "img1"
    if not img_dir.exists():
        img_dir = sequence_dir
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(path for path in img_dir.iterdir() if path.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No frames found in {img_dir}")
    return paths


def normalize_detections(detections: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(detections, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, :6]


def normalize_tracks(tracks: np.ndarray) -> np.ndarray:
    arr = np.asarray(tracks, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 8), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def extract_feature_samples(
    tracker_name: str,
    tracker,
    detections: np.ndarray,
    tracks: np.ndarray,
    frame: np.ndarray,
    sequence: str,
    frame_id: int,
) -> List[FeatureSample]:
    tracks = normalize_tracks(tracks)
    if len(tracks) == 0 or len(detections) == 0 or tracker.model is None:
        return []
    features = tracker.model.get_features(detections[:, 0:4], frame)
    samples: List[FeatureSample] = []
    for row in tracks:
        if row.shape[0] < 8:
            continue
        det_ind = int(row[7])
        if det_ind < 0 or det_ind >= len(features):
            continue
        samples.append(
            FeatureSample(
                tracker=tracker_name,
                sequence=sequence,
                frame_id=frame_id,
                track_id=int(row[4]),
                stage=int(row[6]) if row.shape[0] > 6 else 0,
                feature=np.asarray(features[det_ind], dtype=np.float32).reshape(-1),
            )
        )
    return samples


def select_top_identity_samples(samples: List[FeatureSample], args: argparse.Namespace) -> List[FeatureSample]:
    """Keep all samples from the top-N most frequent IDs in each selected sequence."""
    by_key: Dict[Tuple[str, str, int], List[FeatureSample]] = defaultdict(list)
    for sample in samples:
        if sample.sequence not in args.feature_sequences:
            continue
        by_key[(sample.tracker, sample.sequence, sample.track_id)].append(sample)

    kept: List[FeatureSample] = []
    for tracker in TRACKERS:
        for sequence in args.feature_sequences:
            groups = [
                (key, group)
                for key, group in by_key.items()
                if key[0] == tracker and key[1] == sequence
            ]
            for (_, _, track_id), group in sorted(groups, key=lambda item: len(item[1]), reverse=True)[: args.top_feature_ids]:
                kept.extend(sorted(group, key=lambda sample: sample.frame_id))
                LOGGER.info(
                    "Selected feature ID | tracker=%s | sequence=%s | id=%d | samples=%d",
                    tracker,
                    sequence,
                    track_id,
                    len(group),
                )
    return kept


def compute_tsne(samples: List[FeatureSample], args: argparse.Namespace) -> np.ndarray:
    features = np.vstack([sample.feature for sample in samples])
    features = StandardScaler().fit_transform(features)
    if features.shape[1] > args.pca_dim and len(samples) > args.pca_dim:
        features = PCA(n_components=args.pca_dim, random_state=args.seed).fit_transform(features)
    perplexity = min(args.perplexity, max(5, (len(samples) - 1) // 3))
    LOGGER.info("Running t-SNE | samples=%d | dim=%d | perplexity=%d", len(samples), features.shape[1], perplexity)
    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=args.seed,
    ).fit_transform(features)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: Iterable[str], dpi: int = OUTPUT_DPI) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt.lstrip('.')}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.025)
        LOGGER.info("Saved figure | path=%s", path)


def plot_tsne_by_id(samples: List[FeatureSample], coords: np.ndarray, args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(1, 2, figsize=args.tsne_figsize, dpi=PREVIEW_DPI, sharex=True, sharey=True, constrained_layout=True)
    counts = Counter((sample.tracker, sample.identity) for sample in samples)
    top_ids = {
        tracker: [identity for (_, identity), _ in counts_for_tracker.most_common(args.top_legend_ids)]
        for tracker in TRACKERS
        for counts_for_tracker in [Counter({key: value for key, value in counts.items() if key[0] == tracker})]
    }

    for ax, tracker in zip(axes, TRACKERS):
        tracker_idx = [i for i, sample in enumerate(samples) if sample.tracker == tracker]
        identities = sorted({samples[i].identity for i in tracker_idx})
        color_map = {identity: CVPR_ID_PALETTE[j % len(CVPR_ID_PALETTE)] for j, identity in enumerate(identities)}
        label_map = {sample.identity: sample.identity_label for sample in samples if sample.tracker == tracker}
        for identity in identities:
            idx = [i for i in tracker_idx if samples[i].identity == identity]
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                s=11,
                alpha=0.84,
                color=color_map[identity],
                edgecolors="white",
                linewidths=0.16,
                rasterized=True,
            )
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=label_map[identity],
                markerfacecolor=color_map[identity],
                markeredgecolor="white",
                markeredgewidth=0.25,
                markersize=4.2,
            )
            for identity in top_ids[tracker]
            if identity in color_map
        ]
        ax.set_title(DISPLAY_NAMES[tracker])
        ax.set_xlabel("t-SNE 1")
        style_axis(ax)
        if handles:
            legend = ax.legend(handles=handles, frameon=True, loc="best", title="Top IDs", borderpad=0.25, handletextpad=0.25, labelspacing=0.2)
            legend.get_frame().set_edgecolor("#D8DEE6")
            legend.get_frame().set_linewidth(0.6)
    axes[0].set_ylabel("t-SNE 2")
    save_figure(fig, args.output_dir, "tsne_by_id", args.formats, args.dpi)
    plt.close(fig)


def plot_tsne_by_stage(samples: List[FeatureSample], coords: np.ndarray, args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(1, 2, figsize=args.tsne_figsize, dpi=PREVIEW_DPI, sharex=True, sharey=True, constrained_layout=True)
    for ax, tracker in zip(axes, TRACKERS):
        tracker_idx = [i for i, sample in enumerate(samples) if sample.tracker == tracker]
        for stage, label in STAGE_NAMES.items():
            idx = [i for i in tracker_idx if samples[i].stage == stage]
            if idx:
                ax.scatter(
                    coords[idx, 0],
                    coords[idx, 1],
                    s=12,
                    alpha=0.86,
                    color=STAGE_COLORS[stage],
                    label=label,
                    edgecolors="white",
                    linewidths=0.16,
                    rasterized=True,
                )
        ax.set_title(DISPLAY_NAMES[tracker])
        ax.set_xlabel("t-SNE 1")
        style_axis(ax)
        legend = ax.legend(frameon=True, loc="best", borderpad=0.25, handletextpad=0.25, labelspacing=0.2)
        legend.get_frame().set_edgecolor("#D8DEE6")
        legend.get_frame().set_linewidth(0.6)
    axes[0].set_ylabel("t-SNE 2")
    save_figure(fig, args.output_dir, "tsne_by_stage", args.formats, args.dpi)
    plt.close(fig)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def sample_similarity_pairs(samples: List[FeatureSample], tracker: str, args: argparse.Namespace) -> Tuple[List[float], List[float]]:
    """Sample same-ID positives and same-stage different-ID hard negatives."""
    rng = np.random.default_rng(args.seed)
    tracker_samples = [sample for sample in samples if sample.tracker == tracker]
    by_identity: Dict[Tuple[str, int], List[FeatureSample]] = defaultdict(list)
    by_stage: Dict[int, List[FeatureSample]] = defaultdict(list)
    for sample in tracker_samples:
        by_identity[sample.identity].append(sample)
        by_stage[sample.stage].append(sample)

    positive_groups = [group for group in by_identity.values() if len(group) >= 2]
    positives: List[float] = []
    for _ in range(args.max_similarity_pairs):
        if not positive_groups:
            break
        group = positive_groups[int(rng.integers(len(positive_groups)))]
        i, j = rng.choice(len(group), size=2, replace=False)
        positives.append(cosine_similarity(group[int(i)].feature, group[int(j)].feature))

    negatives: List[float] = []
    candidate_stages = [stage for stage, group in by_stage.items() if len({sample.identity for sample in group}) >= 2]
    for _ in range(args.max_similarity_pairs):
        if not candidate_stages:
            break
        stage = candidate_stages[int(rng.integers(len(candidate_stages)))]
        group = by_stage[stage]
        first = group[int(rng.integers(len(group)))]
        different = [sample for sample in group if sample.identity != first.identity]
        if not different:
            continue
        second = different[int(rng.integers(len(different)))]
        negatives.append(cosine_similarity(first.feature, second.feature))

    return positives, negatives


def draw_violin(ax, values: Sequence[Sequence[float]], labels: Sequence[str], colors: Sequence[str], ylabel: str) -> None:
    parts = ax.violinplot(values, showmeans=True, showmedians=False, showextrema=True)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#222222")
        body.set_linewidth(0.55)
        body.set_alpha(0.78)
    for key in ("cmeans", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("#222222")
        parts[key].set_linewidth(0.8)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    style_axis(ax, grid_axis="y")


def plot_feature_similarity(samples: List[FeatureSample], args: argparse.Namespace) -> None:
    stats = {tracker: sample_similarity_pairs(samples, tracker, args) for tracker in TRACKERS}
    if any(len(stats[tracker][0]) == 0 or len(stats[tracker][1]) == 0 for tracker in TRACKERS):
        LOGGER.warning("Skip feature similarity plot because one tracker lacks enough positive or negative pairs.")
        return

    fig, axes = plt.subplots(1, 2, figsize=args.similarity_figsize, dpi=PREVIEW_DPI, constrained_layout=True)
    labels = [DISPLAY_NAMES[tracker] for tracker in TRACKERS]
    colors = [TRACKER_COLORS[tracker] for tracker in TRACKERS]
    draw_violin(
        axes[0],
        [stats[tracker][0] for tracker in TRACKERS],
        labels,
        colors,
        "Same-ID cosine similarity",
    )
    draw_violin(
        axes[1],
        [stats[tracker][1] for tracker in TRACKERS],
        labels,
        colors,
        "Different-ID same-stage similarity",
    )
    axes[0].set_title("Positive pairs")
    axes[1].set_title("Hard negative pairs")
    for ax in axes:
        ax.set_ylim(-1.0, 1.0)
    save_figure(fig, args.output_dir, "feature_similarity_distribution", args.formats, args.dpi)
    plt.close(fig)

    for tracker in TRACKERS:
        pos = np.asarray(stats[tracker][0], dtype=np.float32)
        neg = np.asarray(stats[tracker][1], dtype=np.float32)
        LOGGER.info(
            "Feature separability | tracker=%s | pos_mean=%.3f | neg_mean=%.3f | gap=%.3f",
            tracker,
            float(pos.mean()),
            float(neg.mean()),
            float(pos.mean() - neg.mean()),
        )


def association_margins(samples: List[MatrixSample], tracker: str) -> List[float]:
    margins: List[float] = []
    for sample in samples:
        if sample.tracker != tracker or sample.final_cost.shape[0] < 2:
            continue
        for det_col in range(sample.final_cost.shape[1]):
            ordered = np.sort(sample.final_cost[:, det_col])
            margins.append(float(ordered[1] - ordered[0]))
    return margins


def plot_association_margin(samples: List[MatrixSample], args: argparse.Namespace) -> None:
    margins = {tracker: association_margins(samples, tracker) for tracker in TRACKERS}
    if any(len(margins[tracker]) == 0 for tracker in TRACKERS):
        LOGGER.warning("Skip association margin plot because one tracker has no valid matrix margins.")
        return

    fig, ax = plt.subplots(figsize=args.margin_figsize, dpi=PREVIEW_DPI, constrained_layout=True)
    labels = [DISPLAY_NAMES[tracker] for tracker in TRACKERS]
    draw_violin(ax, [margins[tracker] for tracker in TRACKERS], labels, [TRACKER_COLORS[tracker] for tracker in TRACKERS], "Best-to-second cost margin")
    ax.set_title("Association Confidence Margin")
    save_figure(fig, args.output_dir, "association_margin_distribution", args.formats, args.dpi)
    plt.close(fig)

    for tracker in TRACKERS:
        values = np.asarray(margins[tracker], dtype=np.float32)
        LOGGER.info(
            "Association margin | tracker=%s | mean=%.3f | median=%.3f | n=%d",
            tracker,
            float(values.mean()),
            float(np.median(values)),
            len(values),
        )


def matrix_color_limits(samples: Sequence[MatrixSample]) -> Tuple[float, float]:
    values = np.concatenate([np.asarray(sample.final_cost, dtype=np.float32).ravel() for sample in samples])
    vmin, vmax = np.nanpercentile(values, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(float(vmax - vmin)) < 1e-9:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        if abs(vmax - vmin) < 1e-9:
            vmin -= 0.5
            vmax += 0.5
    return float(vmin), float(vmax)


def sparse_tick_positions(length: int, max_labels: int) -> List[int]:
    if length <= max_labels:
        return list(range(length))
    step = max(1, int(np.ceil(length / max_labels)))
    positions = list(range(0, length, step))
    if positions[-1] != length - 1:
        positions.append(length - 1)
    return positions


def set_matrix_ticks(ax: plt.Axes, sample: MatrixSample, max_x_labels: int, max_y_labels: int) -> None:
    x_positions = sparse_tick_positions(len(sample.det_ids), max_x_labels)
    y_positions = sparse_tick_positions(len(sample.track_ids), max_y_labels)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"D{sample.det_ids[idx]}" for idx in x_positions], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"T{sample.track_ids[idx]}" for idx in y_positions])
    ax.tick_params(axis="x", pad=1.0)
    ax.tick_params(axis="y", pad=1.0)


def draw_matrix_axis(
    ax: plt.Axes,
    sample: MatrixSample,
    args: argparse.Namespace,
    vmin: float,
    vmax: float,
    show_ylabel: bool,
    title: str,
):
    values = np.asarray(sample.final_cost, dtype=np.float32)
    im = ax.imshow(values, cmap=args.heatmap_cmap, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(title, pad=4.0)
    ax.set_xlabel("Current detections")
    ax.set_ylabel("Historical tracklets" if show_ylabel else "")
    set_matrix_ticks(ax, sample, args.matrix_max_xticks, args.matrix_max_yticks)
    style_axis(ax, grid_axis="")
    ax.set_xticks(np.arange(-0.5, len(sample.det_ids), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sample.track_ids), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.35, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def plot_matrix(sample: MatrixSample, args: argparse.Namespace, index: int) -> None:
    fig, ax = plt.subplots(figsize=args.matrix_figsize, dpi=PREVIEW_DPI, constrained_layout=True)
    vmin, vmax = matrix_color_limits([sample])
    im = draw_matrix_axis(ax, sample, args, vmin, vmax, True, f"{DISPLAY_NAMES[sample.tracker]} | Frame {sample.frame_id}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Association cost")
    cbar.outline.set_edgecolor(SPINE_COLOR)
    cbar.outline.set_linewidth(0.6)
    stem = f"{sample.tracker}_{sample.sequence}_frame_{sample.frame_id:06d}_association_cost_{index:02d}"
    save_figure(fig, args.output_dir / "association_matrices", stem, args.formats, args.dpi)
    plt.close(fig)


def best_matrix_by_frame(samples: Sequence[MatrixSample]) -> Dict[int, MatrixSample]:
    best: Dict[int, MatrixSample] = {}
    for sample in samples:
        current = best.get(sample.frame_id)
        if current is None or sample.final_cost.size > current.final_cost.size:
            best[sample.frame_id] = sample
    return best


def plot_combined_matrices(samples: List[MatrixSample], args: argparse.Namespace) -> None:
    by_group: Dict[Tuple[str, str], List[MatrixSample]] = defaultdict(list)
    for sample in samples:
        if sample.frame_id in args.matrix_frames:
            by_group[(sample.sequence, sample.tracker)].append(sample)

    sequences = sorted({sample.sequence for sample in samples})
    for sequence in sequences:
        selected: List[MatrixSample] = []
        frame_lookup: Dict[Tuple[str, int], MatrixSample] = {}
        for tracker in TRACKERS:
            frame_lookup_for_tracker = best_matrix_by_frame(by_group.get((sequence, tracker), []))
            for frame_id in args.matrix_frames:
                sample = frame_lookup_for_tracker.get(frame_id)
                if sample is not None:
                    frame_lookup[(tracker, frame_id)] = sample
                    selected.append(sample)
                else:
                    LOGGER.warning("Missing matrix sample | sequence=%s | tracker=%s | frame=%d", sequence, tracker, frame_id)
        if not selected:
            continue

        fig, axes = plt.subplots(
            len(args.matrix_frames),
            len(TRACKERS),
            figsize=args.combined_matrix_figsize,
            dpi=PREVIEW_DPI,
            constrained_layout=True,
            squeeze=False,
        )
        for row, frame_id in enumerate(args.matrix_frames):
            for col, tracker in enumerate(TRACKERS):
                ax = axes[row][col]
                sample = frame_lookup.get((tracker, frame_id))
                if sample is None:
                    ax.axis("off")
                    continue
                title = f"{DISPLAY_NAMES[tracker]} | Frame {frame_id}"
                vmin, vmax = matrix_color_limits([sample])
                im = draw_matrix_axis(ax, sample, args, vmin, vmax, col == 0, title)
                cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.018)
                cbar.set_label("Association cost")
                cbar.outline.set_edgecolor(SPINE_COLOR)
                cbar.outline.set_linewidth(0.6)

        stem = f"{sequence}_association_cost_frames_{'_'.join(str(frame) for frame in args.matrix_frames)}_combined"
        save_figure(fig, args.output_dir / "association_matrices", stem, args.formats, args.dpi)
        plt.close(fig)


def select_matrix_samples(samples: List[MatrixSample], count_per_tracker: int) -> List[MatrixSample]:
    """Interleave single-matrix examples across trackers and sequences."""
    if count_per_tracker <= 0:
        return []
    by_group: Dict[Tuple[str, str], List[MatrixSample]] = defaultdict(list)
    for sample in samples:
        by_group[(sample.tracker, sample.sequence)].append(sample)

    selected: List[MatrixSample] = []
    for tracker in TRACKERS:
        tracker_selected: List[MatrixSample] = []
        sequences = sorted(sequence for group_tracker, sequence in by_group if group_tracker == tracker)
        while len(tracker_selected) < count_per_tracker and any(by_group[(tracker, sequence)] for sequence in sequences):
            for sequence in sequences:
                if by_group[(tracker, sequence)]:
                    tracker_selected.append(by_group[(tracker, sequence)].pop(0))
                    if len(tracker_selected) >= count_per_tracker:
                        break
        selected.extend(tracker_selected)
    return selected


def run_tracker_on_sequence(
    args: argparse.Namespace,
    detector: VideoDetector,
    tracker_name: str,
    sequence_dir: Path,
) -> Tuple[List[FeatureSample], List[MatrixSample]]:
    tracker = build_tracker(args, tracker_name)
    probe = AssociationProbe(tracker_name, tracker, args.matrix_count, args.matrix_min_size, args.matrix_stride, args.matrix_frames)
    frame_paths = load_frame_paths(sequence_dir)
    if args.max_frames_per_seq > 0:
        frame_paths = frame_paths[: args.max_frames_per_seq]

    samples: List[FeatureSample] = []
    start = time.perf_counter()
    try:
        for frame_id, img_path in enumerate(frame_paths, start=1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                LOGGER.warning("Skip unreadable frame | path=%s", img_path)
                continue
            detections = normalize_detections(detector.detect_frame_with_sahi(frame))
            probe.set_frame(sequence_dir.name, frame_id)
            tracks = tracker.update(detections, frame)
            if frame_id % args.sample_stride == 0:
                samples.extend(extract_feature_samples(tracker_name, tracker, detections, tracks, frame, sequence_dir.name, frame_id))
            if frame_id % args.log_interval == 0:
                LOGGER.info(
                    "Progress | tracker=%s | sequence=%s | frame=%d/%d | samples=%d | matrices=%d",
                    tracker_name,
                    sequence_dir.name,
                    frame_id,
                    len(frame_paths),
                    len(samples),
                    len(probe.samples),
                )
    finally:
        probe.close()
    LOGGER.info(
        "Sequence done | tracker=%s | sequence=%s | frames=%d | samples=%d | fps=%.2f",
        tracker_name,
        sequence_dir.name,
        len(frame_paths),
        len(samples),
        len(frame_paths) / max(time.perf_counter() - start, 1e-9),
    )
    return samples, probe.samples


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset test directory not found: {args.data_root}")
    ensure_file(args.detector_weights, "Detector weights")
    ensure_file(args.baseline_reid_weights, "Baseline ReID weights")
    ensure_file(args.ours_reid_weights, "Ours ReID weights")
    for tracker_name in TRACKERS:
        ensure_file(args.tracker_config_dir / f"{tracker_name}.yaml", "Tracker config")
    for sequence in args.sequences:
        sequence_dir = resolve_sequence_dir(args.data_root, sequence)
        if not sequence_dir.exists():
            candidates = ", ".join(str(args.data_root / name) for name in sequence_name_candidates(sequence))
            raise FileNotFoundError(f"Sequence not found. Tried: {candidates}")
    sequence_dirs = {resolve_sequence_dir(args.data_root, sequence).resolve() for sequence in args.sequences}
    for sequence in args.feature_sequences:
        if resolve_sequence_dir(args.data_root, sequence).resolve() not in sequence_dirs:
            raise ValueError(f"Feature sequence {sequence} must also be included in --sequences.")
    plt.get_cmap(args.heatmap_cmap)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw t-SNE feature distributions and association cost matrices for blueberry trackers.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="MOT test directory containing selected sequences.")
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES), help="Sequence names to sample.")
    parser.add_argument("--feature-sequences", nargs="+", default=list(FEATURE_SEQUENCES), help="Sequences used for feature distribution plots.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Figure output directory.")
    parser.add_argument("--tracker-config-dir", type=Path, default=CONFIG_DIR, help="BoxMOT tracker config directory.")
    parser.add_argument("--detector-weights", type=Path, default=DETECTOR_WEIGHTS, help="YOLO detector weights.")
    parser.add_argument("--reid-weights", type=Path, default=REID_WEIGHTS, help="Shared fallback appearance/ReID weights.")
    parser.add_argument("--baseline-reid-weights", type=Path, default=None, help="Baseline Deep OC-SORT ReID weights.")
    parser.add_argument("--ours-reid-weights", type=Path, default=None, help="BerryTracker improved appearance branch weights.")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"], help="Output figure formats.")
    parser.add_argument("--dpi", type=int, default=OUTPUT_DPI, help="Raster output dpi. 600 is suitable for paper figures.")
    parser.add_argument("--tsne-figsize", nargs=2, type=float, default=list(TSNE_FIGSIZE), metavar=("W", "H"), help="t-SNE figure size in inches, tuned for a CVPR double column by default.")
    parser.add_argument("--similarity-figsize", nargs=2, type=float, default=list(SIMILARITY_FIGSIZE), metavar=("W", "H"), help="Feature similarity figure size in inches.")
    parser.add_argument("--margin-figsize", nargs=2, type=float, default=list(MARGIN_FIGSIZE), metavar=("W", "H"), help="Association margin figure size in inches.")
    parser.add_argument("--matrix-figsize", nargs=2, type=float, default=list(MATRIX_FIGSIZE), metavar=("W", "H"), help="Association matrix figure size in inches, tuned for a single column by default.")
    parser.add_argument("--combined-matrix-figsize", nargs=2, type=float, default=list(COMBINED_MATRIX_FIGSIZE), metavar=("W", "H"), help="Combined association matrix figure size in inches.")
    parser.add_argument("--heatmap-cmap", type=str, default="viridis", help="Matplotlib colormap for association cost heatmaps.")
    parser.add_argument("--conf", type=float, default=0.1, help="SAHI detection confidence threshold.")
    parser.add_argument("--slice-size", type=int, default=640, help="SAHI slice height and width.")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI slice overlap ratio.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Detector device.")
    parser.add_argument("--tracker-device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Tracker/ReID device.")
    parser.add_argument("--half", action="store_true", help="Use half precision for ReID inference.")
    parser.add_argument("--sample-stride", type=int, default=1, help="Collect feature samples every N frames.")
    parser.add_argument("--max-frames-per-seq", type=int, default=0, help="Limit frames per sequence; 0 means all frames.")
    parser.add_argument("--top-feature-ids", type=int, default=4, help="Top frequent IDs kept per tracker and feature sequence.")
    parser.add_argument("--top-legend-ids", type=int, default=12, help="Number of most frequent IDs shown in the legend.")
    parser.add_argument("--max-similarity-pairs", type=int, default=5000, help="Maximum sampled feature pairs per tracker and pair type.")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity upper bound.")
    parser.add_argument("--pca-dim", type=int, default=50, help="PCA dimension before t-SNE.")
    parser.add_argument("--matrix-count", type=int, default=6, help="Number of association matrices to save per tracker.")
    parser.add_argument("--matrix-min-size", type=int, default=4, help="Minimum detections/tracklets needed for matrix capture.")
    parser.add_argument("--matrix-stride", type=int, default=10, help="Capture association matrices every N frames.")
    parser.add_argument("--matrix-frames", nargs="+", type=int, default=list(MATRIX_FRAMES), help="Specific frame IDs to capture and merge in association matrix figures.")
    parser.add_argument("--matrix-max-xticks", type=int, default=14, help="Maximum visible x-axis tick labels in association matrices.")
    parser.add_argument("--matrix-max-yticks", type=int, default=16, help="Maximum visible y-axis tick labels in association matrices.")
    parser.add_argument("--skip-single-matrices", action="store_true", help="Only save combined association matrix figures.")
    parser.add_argument("--log-interval", type=int, default=100, help="Progress log interval in frames.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling and t-SNE.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    setup_plot_style()
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.feature_sequences = list(args.feature_sequences)
    args.output_dir = args.output_dir.resolve()
    args.tracker_config_dir = args.tracker_config_dir.resolve()
    args.detector_weights = args.detector_weights.resolve()
    args.reid_weights = args.reid_weights.resolve()
    args.baseline_reid_weights = (args.baseline_reid_weights or args.reid_weights).resolve()
    args.ours_reid_weights = (args.ours_reid_weights or args.reid_weights).resolve()
    args.dpi = max(300, int(args.dpi))
    args.tsne_figsize = parse_figsize(args.tsne_figsize, TSNE_FIGSIZE)
    args.similarity_figsize = parse_figsize(args.similarity_figsize, SIMILARITY_FIGSIZE)
    args.margin_figsize = parse_figsize(args.margin_figsize, MARGIN_FIGSIZE)
    args.matrix_figsize = parse_figsize(args.matrix_figsize, MATRIX_FIGSIZE)
    args.combined_matrix_figsize = parse_figsize(args.combined_matrix_figsize, COMBINED_MATRIX_FIGSIZE)
    args.matrix_frames = sorted({int(frame) for frame in args.matrix_frames if int(frame) > 0})
    if not args.matrix_frames:
        raise ValueError("--matrix-frames must contain at least one positive frame ID.")
    args.matrix_count = max(int(args.matrix_count), len(args.matrix_frames))
    args.matrix_max_xticks = max(2, int(args.matrix_max_xticks))
    args.matrix_max_yticks = max(2, int(args.matrix_max_yticks))
    validate_args(args)
    sequence_dirs = [resolve_sequence_dir(args.data_root, sequence).resolve() for sequence in args.sequences]
    args.feature_sequences = [resolve_sequence_dir(args.data_root, sequence).resolve().name for sequence in args.feature_sequences]

    LOGGER.info("Loading SAHI detector | weights=%s | device=%s", args.detector_weights, args.device)
    detector = VideoDetector(
        yolo_weights_path=str(args.detector_weights),
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
        conf_threshold=args.conf,
        device=args.device,
    )

    all_samples: List[FeatureSample] = []
    matrix_samples: List[MatrixSample] = []
    for tracker_name in TRACKERS:
        for sequence_dir in sequence_dirs:
            samples, matrices = run_tracker_on_sequence(args, detector, tracker_name, sequence_dir)
            all_samples.extend(samples)
            matrix_samples.extend(matrices)

    selected = select_top_identity_samples(all_samples, args)
    if len(selected) < max(20, args.perplexity + 2):
        raise RuntimeError(f"Not enough selected feature samples for t-SNE: {len(selected)}")
    LOGGER.info("Feature samples | raw=%d | selected=%d", len(all_samples), len(selected))
    coords = compute_tsne(selected, args)
    plot_tsne_by_id(selected, coords, args)
    plot_tsne_by_stage(selected, coords, args)
    plot_feature_similarity(selected, args)

    plot_combined_matrices(matrix_samples, args)

    selected_matrices = select_matrix_samples(matrix_samples, args.matrix_count)
    if not selected_matrices:
        LOGGER.warning("No association matrices were captured. Try lowering --matrix-min-size or --matrix-stride.")
    if not args.skip_single_matrices:
        for index, sample in enumerate(selected_matrices, start=1):
            plot_matrix(sample, args, index)
    plot_association_margin(matrix_samples, args)
    LOGGER.info("Done | output=%s", args.output_dir)


if __name__ == "__main__":
    main()
