#!/usr/bin/env python3
"""
Synthesize a MOT20-style blueberry dataset from static YOLO images.

The script samples static images and YOLO labels, applies smooth camera motion,
optional blur, and fake leaf occlusions, then exports the result in MOT20-like
directory layout.

Ground-truth rows follow:
    frame,id,x,y,w,h,mark,category,visibility
Detection rows follow MOT20 standard format:
    frame,id,x,y,w,h,confidence,-1,-1,-1
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_IMAGE_DIR = Path("/home/wh1234_/data/blueberry_yolo_data/images/test")
DEFAULT_LABEL_DIR = Path("/home/wh1234_/data/blueberry_yolo_data/labels/test")
DEFAULT_LEAF_DIR = Path("/home/wh1234_/data/blueberry_yolo_data/leaf")
DEFAULT_OUTPUT_DIR = Path("/home/wh1234_/data/blueberry_mot3")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Box:
    track_id: int
    class_id: int
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class LeafAsset:
    rgba: np.ndarray


@dataclass(frozen=True)
class MotionSpec:
    dx_total: float
    dy_step: float
    angle_amp: float
    zoom_amp: float
    angle_phase: float
    zoom_phase: float
    jitter_x: float
    jitter_y: float
    jitter_phase_x: float
    jitter_phase_y: float


@dataclass(frozen=True)
class LeafTrack:
    asset_index: int
    scale: float
    flip: bool
    start_x: int
    end_x: int
    center_y: int
    sway_amp: float
    sway_phase: float
    start_frame: int
    end_frame: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize MOT20 blueberry sequences.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--leaf-dir", type=Path, default=DEFAULT_LEAF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-sequences", type=int, default=25)
    parser.add_argument("--frames-per-seq", type=int, default=90)
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3408)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-name", default="train")
    return parser.parse_args()


def validate_dir(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {path}")


def list_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()])


def corresponding_label_path(image_path: Path, label_dir: Path) -> Path:
    return label_dir / f"{image_path.stem}.txt"


def load_yolo_boxes(label_path: Path, width: int, height: int) -> List[Box]:
    boxes: List[Box] = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))
            xc = float(parts[1]) * width
            yc = float(parts[2]) * height
            bw = float(parts[3]) * width
            bh = float(parts[4]) * height
            x = xc - bw / 2.0
            y = yc - bh / 2.0

            if bw <= 1 or bh <= 1:
                continue

            boxes.append(Box(track_id=idx, class_id=class_id, x=x, y=y, w=bw, h=bh))

    return boxes


def sample_motion_spec(width: int, height: int, rng: random.Random) -> MotionSpec:
    # travel = min(width, height) * rng.uniform(0.45, 0.7)
    travel = min(width, height) * rng.uniform(0.32, 0.5)
    return MotionSpec(
        dx_total=travel,
        # dy_step=height * rng.uniform(0.015, 0.035),
        dy_step=height * rng.uniform(0.008, 0.02),
        angle_amp=rng.uniform(-4.5, 4.5),
        zoom_amp=rng.uniform(-0.08, 0.12),
        angle_phase=rng.uniform(-0.9, 0.9),
        zoom_phase=rng.uniform(-0.6, 0.6),
        # jitter_x=width * rng.uniform(0.006, 0.018),
        # jitter_y=height * rng.uniform(0.006, 0.018),
        jitter_x=width * rng.uniform(0.002, 0.008),
        jitter_y=height * rng.uniform(0.002, 0.008),
        jitter_phase_x=rng.uniform(0.0, math.tau),
        jitter_phase_y=rng.uniform(0.0, math.tau),
    )


def build_affine(
    width: int,
    height: int,
    frame_idx: int,
    total_frames: int,
    spec: MotionSpec,
    dy_offset: float,
) -> np.ndarray:
    t = 0.0 if total_frames <= 1 else frame_idx / (total_frames - 1)
    dx = spec.dx_total * t + spec.jitter_x * math.sin(math.tau * t + spec.jitter_phase_x)
    dy = dy_offset + spec.jitter_y * math.sin(math.tau * t + spec.jitter_phase_y)
    angle = spec.angle_amp * math.sin(math.pi * (t + spec.angle_phase))
    scale = 1.0 + spec.zoom_amp * math.sin(math.pi * (t + 0.15 + spec.zoom_phase))

    center = (width / 2.0, height / 2.0)
    affine = cv2.getRotationMatrix2D(center, angle, scale)
    affine[:, 2] += np.array([dx, dy], dtype=np.float32)
    return affine.astype(np.float32)


def build_vertical_offsets(height: int, total_frames: int, spec: MotionSpec, rng: random.Random) -> List[float]:
    if total_frames <= 0:
        return []

    offsets = [0.0]
    current = 0.0
    max_abs = height * rng.uniform(0.08, 0.18)

    for _ in range(1, total_frames):
        direction = rng.choice([-1.0, 1.0])
        step_scale = rng.uniform(0.45, 1.0)
        current += direction * spec.dy_step * step_scale
        current = max(-max_abs, min(max_abs, current))
        offsets.append(current)

    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
    smoothed = np.convolve(np.array(offsets, dtype=np.float32), kernel, mode="same")
    smoothed[0] = offsets[0]
    smoothed[-1] = offsets[-1]
    return smoothed.tolist()


def transform_box_raw(box: Box, affine: np.ndarray) -> Tuple[float, float, float, float]:
    points = np.array(
        [
            [box.x, box.y],
            [box.x + box.w, box.y],
            [box.x + box.w, box.y + box.h],
            [box.x, box.y + box.h],
        ],
        dtype=np.float32,
    )
    transformed = cv2.transform(points[None, :, :], affine)[0]

    x1 = float(np.min(transformed[:, 0]))
    y1 = float(np.min(transformed[:, 1]))
    x2 = float(np.max(transformed[:, 0]))
    y2 = float(np.max(transformed[:, 1]))
    return x1, y1, x2, y2


def clip_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[float, float, float, float] | None:
    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    x2 = max(0.0, min(x2, width - 1.0))
    y2 = max(0.0, min(y2, height - 1.0))

    new_w = x2 - x1
    new_h = y2 - y1
    if new_w < 2.0 or new_h < 2.0:
        return None
    return x1, y1, new_w, new_h


def transform_box(box: Box, affine: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = transform_box_raw(box, affine)
    return clip_bbox(x1, y1, x2, y2, width, height)


def reflect_bbox(raw_bbox: Tuple[float, float, float, float], width: int, height: int, side: str) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = raw_bbox
    if side == "left":
        return -x2, y1, -x1, y2
    if side == "right":
        edge = width - 1.0
        return 2.0 * edge - x2, y1, 2.0 * edge - x1, y2
    if side == "top":
        return x1, -y2, x2, -y1
    if side == "bottom":
        edge = height - 1.0
        return x1, 2.0 * edge - y2, x2, 2.0 * edge - y1
    raise ValueError(f"Unsupported reflection side: {side}")


def collect_box_instances(
    box: Box,
    affine: np.ndarray,
    width: int,
    height: int,
    reflection_id_map: dict[Tuple[int, str], int],
    next_track_id: int,
) -> Tuple[List[Tuple[int, Tuple[float, float, float, float]]], int]:
    raw_bbox = transform_box_raw(box, affine)
    instances: List[Tuple[int, Tuple[float, float, float, float]]] = []

    primary = clip_bbox(*raw_bbox, width, height)
    if primary is not None:
        instances.append((box.track_id, primary))

    x1, y1, x2, y2 = raw_bbox
    overflow_sides: List[str] = []
    if x1 < 0.0:
        overflow_sides.append("left")
    if x2 > width - 1.0:
        overflow_sides.append("right")
    if y1 < 0.0:
        overflow_sides.append("top")
    if y2 > height - 1.0:
        overflow_sides.append("bottom")

    for side in overflow_sides:
        reflected = clip_bbox(*reflect_bbox(raw_bbox, width, height, side), width, height)
        if reflected is None:
            continue
        key = (box.track_id, side)
        if key not in reflection_id_map:
            reflection_id_map[key] = next_track_id
            next_track_id += 1
        instances.append((reflection_id_map[key], reflected))

    return instances, next_track_id


def maybe_blur(image: np.ndarray, frame_idx: int, total_frames: int, rng: random.Random) -> np.ndarray:
    if total_frames <= 1:
        return image

    t = frame_idx / (total_frames - 1)
    out = image

    if rng.random() < 0.35:
        sigma = 0.6 + 1.2 * math.sin(math.pi * t) ** 2
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)

    if rng.random() < 0.25:
        ksize = 3 if t < 0.5 else 5
        if rng.random() < 0.5:
            kernel = np.zeros((ksize, ksize), dtype=np.float32)
            kernel[ksize // 2, :] = 1.0 / ksize
        else:
            kernel = np.zeros((ksize, ksize), dtype=np.float32)
            kernel[:, ksize // 2] = 1.0 / ksize
        out = cv2.filter2D(out, -1, kernel)

    return out


def load_leaf_assets(leaf_dir: Path) -> List[LeafAsset]:
    assets: List[LeafAsset] = []
    for image_path in sorted(leaf_dir.iterdir()):
        if image_path.suffix.lower() != ".png" or not image_path.is_file():
            continue
        rgba = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            continue
        assets.append(LeafAsset(rgba=rgba))
    return assets


def alpha_blend(base: np.ndarray, overlay_rgba: np.ndarray, x: int, y: int, occ_mask: np.ndarray | None = None) -> None:
    h, w = base.shape[:2]
    oh, ow = overlay_rgba.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + ow)
    y2 = min(h, y + oh)
    if x1 >= x2 or y1 >= y2:
        return

    overlay_crop = overlay_rgba[y1 - y : y2 - y, x1 - x : x2 - x]
    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0
    rgb = overlay_crop[:, :, :3].astype(np.float32)
    roi = base[y1:y2, x1:x2].astype(np.float32)
    blended = alpha * rgb + (1.0 - alpha) * roi
    base[y1:y2, x1:x2] = blended.astype(np.uint8)
    if occ_mask is not None:
        occ_mask[y1:y2, x1:x2] |= overlay_crop[:, :, 3] > 24


def overlay_leaf_occlusions(
    image: np.ndarray,
    leaf_assets: Sequence[LeafAsset],
    leaf_tracks: Sequence[LeafTrack],
    frame_idx: int,
    total_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not leaf_assets or not leaf_tracks:
        return image, np.zeros(image.shape[:2], dtype=bool)

    h, w = image.shape[:2]
    output = image.copy()
    occ_mask = np.zeros((h, w), dtype=bool)

    for track in leaf_tracks:
        if frame_idx < track.start_frame or frame_idx > track.end_frame:
            continue

        local_span = max(1, track.end_frame - track.start_frame)
        t = (frame_idx - track.start_frame) / local_span
        rgba = leaf_assets[track.asset_index].rgba
        scale = track.scale * min(w / max(rgba.shape[1], 1), h / max(rgba.shape[0], 1))
        new_w = max(16, int(rgba.shape[1] * scale))
        new_h = max(16, int(rgba.shape[0] * scale))
        resized = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if track.flip:
            resized = cv2.flip(resized, 1)

        x = int(track.start_x + (track.end_x - track.start_x) * t)
        y = track.center_y + int(track.sway_amp * h * math.sin(math.tau * t + track.sway_phase))
        alpha_blend(output, resized, x, y, occ_mask)

    return output, occ_mask


def is_box_occluded(box_xywh: Tuple[float, float, float, float], occ_mask: np.ndarray, threshold: float = 0.6) -> bool:
    x, y, w, h = box_xywh
    x1 = max(0, min(int(math.floor(x)), occ_mask.shape[1] - 1))
    y1 = max(0, min(int(math.floor(y)), occ_mask.shape[0] - 1))
    x2 = max(0, min(int(math.ceil(x + w)), occ_mask.shape[1]))
    y2 = max(0, min(int(math.ceil(y + h)), occ_mask.shape[0]))
    if x2 <= x1 or y2 <= y1:
        return False
    region = occ_mask[y1:y2, x1:x2]
    occluded_area = int(np.count_nonzero(region))
    box_area = max(1, (x2 - x1) * (y2 - y1))
    return (occluded_area / box_area) > threshold


def sample_leaf_tracks(
    width: int,
    height: int,
    total_frames: int,
    leaf_assets: Sequence[LeafAsset],
    rng: random.Random,
) -> List[LeafTrack]:
    min_leaves = min(3, len(leaf_assets))
    max_leaves = min(4, len(leaf_assets))
    num_leaves = rng.randint(min_leaves, max_leaves)
    tracks: List[LeafTrack] = []
    chosen_centers: List[int] = []
    min_center_gap = max(24, int(height * 0.18))
    min_duration = max(18, int(total_frames * 0.28))
    max_duration = max(min_duration, int(total_frames * 0.6))
    entry_slots = np.linspace(0, max(0, total_frames - min_duration), num_leaves, dtype=int).tolist()

    if len(entry_slots) > 1:
        jitter = max(1, total_frames // 18)
        entry_slots = [max(0, min(total_frames - min_duration, slot + rng.randint(-jitter, jitter))) for slot in entry_slots]
        entry_slots = sorted(entry_slots)

    for _ in range(num_leaves):
        center_y = int((0.2 + 0.6 * rng.random()) * height)
        for _attempt in range(20):
            if all(abs(center_y - prev_center) >= min_center_gap for prev_center in chosen_centers):
                break
            center_y = int((0.15 + 0.7 * rng.random()) * height)
        chosen_centers.append(center_y)

        start_x = int((-0.2 + rng.uniform(-0.05, 0.05)) * width)
        end_x = int((0.85 + rng.uniform(0.05, 0.3)) * width)
        if tracks:
            prev_track = tracks[-1]
            horizontal_gap = max(20, int(width * 0.12))
            start_x = min(start_x, prev_track.start_x - horizontal_gap)
            end_x = min(end_x, prev_track.end_x - horizontal_gap)

        start_frame = entry_slots[len(tracks)] if entry_slots else 0
        duration = rng.randint(min_duration, max_duration)
        end_frame = min(total_frames - 1, start_frame + duration)

        tracks.append(
            LeafTrack(
                asset_index=rng.randrange(len(leaf_assets)),
                scale=rng.uniform(0.15, 0.45),
                flip=rng.random() < 0.5,
                start_x=start_x,
                end_x=end_x,
                center_y=center_y,
                sway_amp=rng.uniform(0.01, 0.05),
                sway_phase=rng.uniform(0.0, math.tau),
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
    return tracks


def write_seqinfo(seq_dir: Path, seq_name: str, width: int, height: int, seq_len: int, frame_rate: int) -> None:
    config = configparser.ConfigParser()
    config["Sequence"] = {
        "name": seq_name,
        "imDir": "img1",
        "frameRate": str(frame_rate),
        "seqLength": str(seq_len),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": ".jpg",
    }
    with (seq_dir / "seqinfo.ini").open("w", encoding="utf-8") as handle:
        config.write(handle)


def write_mot_annotations(seq_dir: Path, gt_lines: List[str], det_lines: List[str]) -> None:
    gt_dir = seq_dir / "gt"
    det_dir = seq_dir / "det"
    gt_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    (gt_dir / "gt.txt").write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""), encoding="utf-8")
    (det_dir / "det.txt").write_text("\n".join(det_lines) + ("\n" if det_lines else ""), encoding="utf-8")


def export_seqmap(output_dir: Path, train_name: str, sequence_names: Sequence[str]) -> None:
    seqmaps_dir = output_dir / "seqmaps"
    seqmaps_dir.mkdir(parents=True, exist_ok=True)
    content = ["name", *sequence_names]
    (seqmaps_dir / f"{train_name}.txt").write_text("\n".join(content) + "\n", encoding="utf-8")


def create_sequence(
    image_path: Path,
    label_path: Path,
    leaf_assets: Sequence[LeafAsset],
    seq_dir: Path,
    seq_name: str,
    frames_per_seq: int,
    frame_rate: int,
    rng: random.Random,
) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    height, width = image.shape[:2]
    boxes = load_yolo_boxes(label_path, width, height)

    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)

    gt_lines: List[str] = []
    det_lines: List[str] = []
    used_identity_ids: set[int] = set()
    reflection_id_map: dict[Tuple[int, str], int] = {}
    next_track_id = max((box.track_id for box in boxes), default=0) + 1

    motion_rng = random.Random(rng.randint(0, 10**9))
    occlusion_rng = random.Random(rng.randint(0, 10**9))
    blur_rng = random.Random(rng.randint(0, 10**9))

    motion_spec = sample_motion_spec(width, height, motion_rng)
    vertical_offsets = build_vertical_offsets(height, frames_per_seq, motion_spec, motion_rng)
    leaf_tracks = sample_leaf_tracks(width, height, frames_per_seq, leaf_assets, occlusion_rng)
    motion_params = [
        build_affine(width, height, frame_idx, frames_per_seq, motion_spec, vertical_offsets[frame_idx])
        for frame_idx in range(frames_per_seq)
    ]

    for frame_idx, affine in enumerate(motion_params, start=1):
        frame = cv2.warpAffine(
            image,
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        frame, occ_mask = overlay_leaf_occlusions(frame, leaf_assets, leaf_tracks, frame_idx - 1, frames_per_seq)
        frame = maybe_blur(frame, frame_idx - 1, frames_per_seq, blur_rng)

        frame_name = f"{frame_idx:06d}.jpg"
        if not cv2.imwrite(str(img_dir / frame_name), frame):
            raise RuntimeError(f"Failed to write frame: {img_dir / frame_name}")

        for box in boxes:
            instances, next_track_id = collect_box_instances(
                box, affine, width, height, reflection_id_map, next_track_id
            )
            for instance_track_id, transformed in instances:
                if is_box_occluded(transformed, occ_mask):
                    continue
                used_identity_ids.add(instance_track_id)
                x, y, bw, bh = transformed
                visibility = 1.0
                gt_lines.append(
                    f"{frame_idx},{instance_track_id},{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},1,{box.class_id},{visibility:.2f}"
                )
                det_lines.append(
                    f"{frame_idx},{instance_track_id},{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},1.00,-1,-1,-1"
                )

    write_seqinfo(seq_dir, seq_name, width, height, frames_per_seq, frame_rate)
    write_mot_annotations(seq_dir, gt_lines, det_lines)

    return {
        "sequence": seq_name,
        "source_image": str(image_path),
        "source_label": str(label_path),
        "width": width,
        "height": height,
        "frames": frames_per_seq,
        "objects": len(used_identity_ids),
    }


def collect_candidates(image_dir: Path, label_dir: Path) -> List[Tuple[Path, Path]]:
    candidates: List[Tuple[Path, Path]] = []
    for image_path in list_images(image_dir):
        label_path = corresponding_label_path(image_path, label_dir)
        if not label_path.exists():
            continue
        candidates.append((image_path, label_path))
    return candidates


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    validate_dir(args.image_dir, "image_dir")
    validate_dir(args.label_dir, "label_dir")
    validate_dir(args.leaf_dir, "leaf_dir")

    candidates = collect_candidates(args.image_dir, args.label_dir)
    if len(candidates) < args.num_sequences:
        raise ValueError(
            f"Not enough image/label pairs. Need {args.num_sequences}, found {len(candidates)}."
        )

    leaf_assets = load_leaf_assets(args.leaf_dir)
    if not leaf_assets:
        raise ValueError(f"No valid RGBA leaf PNG assets found in: {args.leaf_dir}")

    rng = random.Random(args.seed)
    sampled = rng.sample(candidates, args.num_sequences)

    prepare_output_dir(args.output_dir, args.overwrite)
    train_root = args.output_dir / args.train_name
    train_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    sequence_names = []

    for seq_idx, (image_path, label_path) in enumerate(sampled, start=1):
        seq_name = f"Blueberry-{seq_idx:02d}"
        sequence_names.append(seq_name)
        seq_dir = train_root / seq_name
        manifest.append(
            create_sequence(
                image_path=image_path,
                label_path=label_path,
                leaf_assets=leaf_assets,
                seq_dir=seq_dir,
                seq_name=seq_name,
                frames_per_seq=args.frames_per_seq,
                frame_rate=args.frame_rate,
                rng=random.Random(rng.randint(0, 10**9)),
            )
        )

    export_seqmap(args.output_dir, args.train_name, sequence_names)

    summary = {
        "dataset_root": str(args.output_dir),
        "split": args.train_name,
        "num_sequences": len(sequence_names),
        "frames_per_sequence": args.frames_per_seq,
        "frame_rate": args.frame_rate,
        "seed": args.seed,
        "sequences": manifest,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
