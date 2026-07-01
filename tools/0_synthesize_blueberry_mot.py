#!/usr/bin/env python3
"""
Build train/test MOT20-style blueberry splits from stitched image pairs.

Compared with 3_synthesize_blueberry_mot.py, this variant:
- stitches two static images for every sequence
- only uses expanded black canvas behavior
- simulates handheld walking motion with controlled single/double directions
- removes gt/det rows when leaf occlusion exceeds 30% for that frame
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from synthesize_blueberry_mot import (
    Box,
    LeafTrack,
    MotionSpec,
    clip_bbox,
    collect_candidates,
    export_seqmap,
    is_box_occluded,
    load_leaf_assets,
    load_yolo_boxes,
    maybe_blur,
    overlay_leaf_occlusions,
    prepare_output_dir,
    transform_box_raw,
    validate_dir,
    write_mot_annotations,
    write_seqinfo,
)


DEFAULT_TRAIN_IMAGE_DIR = Path("/home/wh1234_/data/blueberry_yolo_data/images/train")
DEFAULT_TRAIN_LABEL_DIR = Path("/home/wh1234_/data/blueberry_yolo_data/labels/train")
DEFAULT_TEST_IMAGE_DIR = Path("/home/wh1234_/data/blueberry_yolo_data/images/test")
DEFAULT_TEST_LABEL_DIR = Path("/home/wh1234_/data/blueberry_yolo_data/labels/test")
DEFAULT_LEAF_DIR = Path("/home/wh1234_/data/blueberry_yolo_data/leaf")
DEFAULT_OUTPUT_DIR = Path("/home/wh1234_/data/blueberry_mot_stitched_walk")

SINGLE_PATTERNS = ["left", "right", "left_up", "right_up", "left_down", "right_down"]
DOUBLE_PATTERNS = ["right_up+right_down", "left_up+left_down"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/test stitched-pair blueberry MOT20-style dataset.")
    parser.add_argument("--train-image-dir", type=Path, default=DEFAULT_TRAIN_IMAGE_DIR)
    parser.add_argument("--train-label-dir", type=Path, default=DEFAULT_TRAIN_LABEL_DIR)
    parser.add_argument("--test-image-dir", type=Path, default=DEFAULT_TEST_IMAGE_DIR)
    parser.add_argument("--test-label-dir", type=Path, default=DEFAULT_TEST_LABEL_DIR)
    parser.add_argument("--leaf-dir", type=Path, default=DEFAULT_LEAF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-sequences", type=int, default=20)
    parser.add_argument("--test-sequences", type=int, default=20)
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3414)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sample_handheld_motion_spec(width: int, height: int, rng: random.Random) -> MotionSpec:
    sway_amp_x = width * rng.uniform(0.006, 0.018)
    return MotionSpec(
        dx_total=sway_amp_x,
        dy_step=height * rng.uniform(0.0008, 0.0028),
        angle_amp=rng.uniform(-1.1, 1.1),
        zoom_amp=rng.uniform(-0.012, 0.022),
        angle_phase=rng.uniform(-0.35, 0.35),
        zoom_phase=rng.uniform(-0.25, 0.25),
        jitter_x=width * rng.uniform(0.001, 0.0035),
        jitter_y=height * rng.uniform(0.0015, 0.0045),
        jitter_phase_x=rng.uniform(0.0, math.tau * 0.6),
        jitter_phase_y=rng.uniform(0.0, math.tau * 0.6),
    )


def build_local_camera_offsets(width: int, height: int, total_frames: int, spec: MotionSpec, rng: random.Random) -> tuple[list[float], list[float]]:
    if total_frames <= 0:
        return [], []
    if total_frames == 1:
        return [0.0], [0.0]

    x_offsets: list[float] = []
    y_offsets: list[float] = []
    phase_x2 = rng.uniform(0.0, math.tau)
    phase_y = rng.uniform(0.0, math.tau)
    walk_cycles = rng.uniform(1.8, 3.2)
    bounce_cycles = rng.uniform(2.4, 4.6)
    drift_y = 0.0
    max_abs_drift_y = height * rng.uniform(0.008, 0.025)
    bounce_amp = height * rng.uniform(0.004, 0.012)

    for frame_idx in range(total_frames):
        t = frame_idx / max(total_frames - 1, 1)
        x_wave = spec.dx_total * math.sin(1.1 * math.pi * t + spec.angle_phase)
        x_wave += 0.32 * spec.dx_total * math.sin(walk_cycles * math.pi * t + phase_x2)

        drift_y += rng.choice([-1.0, 1.0]) * spec.dy_step * rng.uniform(0.3, 0.8)
        drift_y = max(-max_abs_drift_y, min(max_abs_drift_y, drift_y))
        y_wave = bounce_amp * math.sin(bounce_cycles * math.pi * t + phase_y)
        x_offsets.append(x_wave)
        y_offsets.append(drift_y + y_wave)

    kernel = np.array([0.18, 0.64, 0.18], dtype=np.float32)
    x_smoothed = np.convolve(np.asarray(x_offsets, dtype=np.float32), kernel, mode="same")
    y_smoothed = np.convolve(np.asarray(y_offsets, dtype=np.float32), kernel, mode="same")
    x_smoothed[0] = x_offsets[0]
    y_smoothed[0] = y_offsets[0]
    x_smoothed[-1] = x_offsets[-1]
    y_smoothed[-1] = y_offsets[-1]
    return x_smoothed.tolist(), y_smoothed.tolist()


def build_affine(width: int, height: int, frame_idx: int, total_frames: int, spec: MotionSpec, dx_local: float, dy_local: float) -> np.ndarray:
    t = 0.0 if total_frames <= 1 else frame_idx / (total_frames - 1)
    dx = dx_local + spec.jitter_x * math.sin(1.9 * math.tau * t + spec.jitter_phase_x)
    dy = dy_local + spec.jitter_y * math.sin(2.4 * math.tau * t + spec.jitter_phase_y)
    angle = spec.angle_amp * math.sin(1.2 * math.pi * t + spec.angle_phase)
    scale = 1.0 + spec.zoom_amp * math.sin(1.15 * math.pi * t + spec.zoom_phase)

    center = (width / 2.0, height / 2.0)
    affine = cv2.getRotationMatrix2D(center, angle, scale)
    affine[:, 2] += np.array([dx, dy], dtype=np.float32)
    return affine.astype(np.float32)


def sequence_moves_left(pattern: str) -> bool:
    return pattern.startswith("left")


def sample_leaf_tracks_custom(
    width: int,
    height: int,
    total_frames: int,
    leaf_assets,
    leaf_move_left: bool,
    active_start_frame: int,
    active_end_frame: int,
    rng: random.Random,
) -> list[LeafTrack]:
    min_leaves = min(3, len(leaf_assets))
    max_leaves = min(5, len(leaf_assets))
    num_leaves = rng.randint(min_leaves, max_leaves)
    tracks: list[LeafTrack] = []
    chosen_centers: list[int] = []
    min_center_gap = max(24, int(height * 0.18))
    active_start_frame = max(0, min(active_start_frame, total_frames - 1))
    active_end_frame = max(active_start_frame, min(active_end_frame, total_frames - 1))
    active_frames = active_end_frame - active_start_frame + 1
    min_duration = min(active_frames, max(18, int(active_frames * 0.28)))
    max_duration = min(active_frames, max(min_duration, int(active_frames * 0.6)))
    entry_slots = np.linspace(
        active_start_frame,
        max(active_start_frame, active_end_frame - min_duration + 1),
        num_leaves,
        dtype=int,
    ).tolist()

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

        if leaf_move_left:
            start_x = int((0.85 + rng.uniform(0.05, 0.3)) * width)
            end_x = int((-0.2 + rng.uniform(-0.05, 0.05)) * width)
        else:
            start_x = int((-0.2 + rng.uniform(-0.05, 0.05)) * width)
            end_x = int((0.85 + rng.uniform(0.05, 0.3)) * width)
        if tracks:
            prev_track = tracks[-1]
            horizontal_gap = max(20, int(width * 0.12))
            if leaf_move_left:
                start_x = max(start_x, prev_track.start_x + horizontal_gap)
                end_x = max(end_x, prev_track.end_x + horizontal_gap)
            else:
                start_x = min(start_x, prev_track.start_x - horizontal_gap)
                end_x = min(end_x, prev_track.end_x - horizontal_gap)
        start_frame = entry_slots[len(tracks)] if entry_slots else active_start_frame
        duration = rng.randint(min_duration, max_duration)
        end_frame = min(active_end_frame, start_frame + duration - 1)

        tracks.append(
            LeafTrack(
                asset_index=rng.randrange(len(leaf_assets)),
                scale=rng.uniform(0.18, 0.42),
                flip=rng.random() < 0.5,
                start_x=start_x,
                end_x=end_x,
                center_y=center_y,
                sway_amp=rng.uniform(0.01, 0.04),
                sway_phase=rng.uniform(0.0, math.tau),
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
    return tracks


def reindex_and_shift_boxes(boxes: Sequence[Box], offset_x: int, offset_y: int, next_track_id: int) -> tuple[list[Box], int]:
    shifted: list[Box] = []
    for box in boxes:
        shifted.append(
            Box(
                track_id=next_track_id,
                class_id=box.class_id,
                x=box.x + offset_x,
                y=box.y + offset_y,
                w=box.w,
                h=box.h,
            )
        )
        next_track_id += 1
    return shifted, next_track_id


def build_stitched_pair(
    image_a: np.ndarray,
    boxes_a: Sequence[Box],
    image_b: np.ndarray,
    boxes_b: Sequence[Box],
) -> tuple[np.ndarray, list[Box], int, int]:
    h1, w1 = image_a.shape[:2]
    h2, w2 = image_b.shape[:2]
    patch_h = max(h1, h2)
    patch_w = w1 + w2
    stitched = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)

    y1 = (patch_h - h1) // 2
    y2 = (patch_h - h2) // 2
    stitched[y1 : y1 + h1, 0:w1] = image_a
    stitched[y2 : y2 + h2, w1 : w1 + w2] = image_b

    stitched_boxes: list[Box] = []
    next_track_id = 1
    left_boxes, next_track_id = reindex_and_shift_boxes(boxes_a, 0, y1, next_track_id)
    right_boxes, next_track_id = reindex_and_shift_boxes(boxes_b, w1, y2, next_track_id)
    stitched_boxes.extend(left_boxes)
    stitched_boxes.extend(right_boxes)

    frame_w = max(w1, w2)
    frame_h = patch_h
    return stitched, stitched_boxes, frame_w, frame_h


def build_expanded_canvas(image: np.ndarray, boxes: Sequence[Box], frame_w: int, frame_h: int, rng: random.Random):
    patch_h, patch_w = image.shape[:2]
    pad_x = max(frame_w, int(patch_w * rng.uniform(0.28, 0.45)))
    pad_y = max(frame_h // 2, int(patch_h * rng.uniform(0.3, 0.5)))

    canvas = np.zeros((patch_h + 2 * pad_y, patch_w + 2 * pad_x, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + patch_h, pad_x : pad_x + patch_w] = image

    shifted_boxes = [
        Box(
            track_id=box.track_id,
            class_id=box.class_id,
            x=box.x + pad_x,
            y=box.y + pad_y,
            w=box.w,
            h=box.h,
        )
        for box in boxes
    ]
    return canvas, shifted_boxes, pad_x, pad_y


def transform_box_to_patch(box: Box, affine: np.ndarray, width: int, height: int, crop_x: int, crop_y: int):
    x1, y1, x2, y2 = transform_box_raw(box, affine)
    return clip_bbox(x1 - crop_x, y1 - crop_y, x2 - crop_x, y2 - crop_y, width, height)


def paste_patch_to_black(patch: np.ndarray, offset_x: int, offset_y: int, out_w: int, out_h: int) -> np.ndarray:
    output = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    patch_h, patch_w = patch.shape[:2]

    dst_x1 = max(0, offset_x)
    dst_y1 = max(0, offset_y)
    dst_x2 = min(out_w, offset_x + patch_w)
    dst_y2 = min(out_h, offset_y + patch_h)
    if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
        return output

    src_x1 = dst_x1 - offset_x
    src_y1 = dst_y1 - offset_y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)
    output[dst_y1:dst_y2, dst_x1:dst_x2] = patch[src_y1:src_y2, src_x1:src_x2]
    return output


def translate_box(box_xywh, offset_x: int, offset_y: int, out_w: int, out_h: int):
    x, y, w, h = box_xywh
    return clip_bbox(x + offset_x, y + offset_y, x + offset_x + w, y + offset_y + h, out_w, out_h)


def compute_visible_ratio(
    patch_x: float,
    patch_y: float,
    patch_w: int,
    patch_h: int,
    frame_w: int,
    frame_h: int,
) -> float:
    x1 = max(0.0, patch_x)
    y1 = max(0.0, patch_y)
    x2 = min(float(frame_w), patch_x + patch_w)
    y2 = min(float(frame_h), patch_y + patch_h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    visible_area = (x2 - x1) * (y2 - y1)
    frame_area = max(1.0, float(frame_w * frame_h))
    return visible_area / frame_area


def find_visibility_window(
    patch_x_positions: Sequence[float],
    patch_y_positions: Sequence[float],
    patch_w: int,
    patch_h: int,
    frame_w: int,
    frame_h: int,
    threshold: float = 0.2,
) -> tuple[int, int]:
    visible_indices = [
        idx
        for idx, (patch_x, patch_y) in enumerate(zip(patch_x_positions, patch_y_positions))
        if compute_visible_ratio(patch_x, patch_y, patch_w, patch_h, frame_w, frame_h) >= threshold
    ]
    if not visible_indices:
        last_idx = max(0, len(patch_x_positions) - 1)
        return 0, last_idx
    return visible_indices[0], visible_indices[-1]


def sample_motion_recipe(is_double: bool, rng: random.Random) -> dict:
    recipe = {
        "is_double": is_double,
        "pattern": rng.choice(DOUBLE_PATTERNS if is_double else SINGLE_PATTERNS),
        "speed_px": rng.uniform(25.0, 35.0),
        "vertical_limit_ratio": rng.uniform(0.05, 0.12),
        "sway_ratio": rng.uniform(0.008, 0.02),
    }
    if is_double:
        recipe["vertical_step_ratio"] = rng.uniform(0.0015, 0.0045)
    else:
        if recipe["pattern"] in {"left_up", "right_up", "left_down", "right_down"}:
            recipe["vertical_target_ratio"] = rng.uniform(0.045, 0.11)
        else:
            recipe["vertical_target_ratio"] = rng.uniform(0.0, 0.025)
    return recipe


def build_motion_plan(frame_w: int, frame_h: int, patch_w: int, recipe: dict, rng: random.Random) -> dict:
    pattern = recipe["pattern"]
    move_left = pattern.startswith("left")
    start_x = float(frame_w) if move_left else float(-patch_w)
    end_x = float(-patch_w) if move_left else float(frame_w)
    total_distance = abs(end_x - start_x)
    frames = max(48, int(math.ceil(total_distance / max(float(recipe["speed_px"]), 1.0))))

    progress = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    if frames >= 8:
        edge_span = min(0.22, 5.0 / max(1.0, float(frames - 1)))
        front_vals = [0.0, edge_span * 0.6, edge_span, edge_span * 1.35]
        back_vals = [1.0 - edge_span * 1.35, 1.0 - edge_span, 1.0 - edge_span * 0.6, 1.0]
        anchor_idx = np.array([0, 1, 2, 3, frames - 4, frames - 3, frames - 2, frames - 1], dtype=np.float32)
        anchor_vals = np.array([*front_vals, *back_vals], dtype=np.float32)
        progress = np.interp(np.arange(frames, dtype=np.float32), anchor_idx, anchor_vals).astype(np.float32)
    x_positions = start_x + (end_x - start_x) * progress

    max_abs_y = frame_h * float(recipe["vertical_limit_ratio"])
    sway_amp = frame_h * float(recipe["sway_ratio"])
    sway_phase = rng.uniform(0.0, math.tau)
    sway = sway_amp * np.sin(np.linspace(0.0, 2.0 * math.pi, frames, dtype=np.float32) + sway_phase)

    if recipe["is_double"]:
        current_y = 0.0
        base_y = [0.0]
        step_ratio = float(recipe["vertical_step_ratio"])
        for _ in range(1, frames):
            step = frame_h * step_ratio * rng.uniform(0.5, 1.0)
            current_y += rng.choice([-1.0, 1.0]) * step
            current_y = max(-max_abs_y, min(max_abs_y, current_y))
            base_y.append(current_y)
        y_positions = np.asarray(base_y, dtype=np.float32)
    else:
        if pattern.endswith("_up"):
            target_y = -frame_h * float(recipe["vertical_target_ratio"])
        elif pattern.endswith("_down"):
            target_y = frame_h * float(recipe["vertical_target_ratio"])
        else:
            target_y = frame_h * rng.uniform(-float(recipe["vertical_target_ratio"]), float(recipe["vertical_target_ratio"]))
        y_positions = np.linspace(0.0, target_y, frames, dtype=np.float32)

    y_positions = y_positions + sway
    if len(y_positions) >= 3:
        kernel = np.array([0.16, 0.68, 0.16], dtype=np.float32)
        y_positions = np.convolve(y_positions, kernel, mode="same")
    y_positions = np.clip(y_positions, -max_abs_y, max_abs_y)
    y_positions[0] = float(y_positions[0])
    y_positions[-1] = float(y_positions[-1])

    return {
        "pattern": pattern,
        "frames": frames,
        "patch_x": [float(x) for x in x_positions],
        "patch_y": [float(y) for y in y_positions],
    }


def create_sequence(
    item: dict,
    leaf_assets,
    seq_dir: Path,
    seq_name: str,
    frame_rate: int,
) -> dict:
    image_a = cv2.imread(str(item["image_path_a"]), cv2.IMREAD_COLOR)
    image_b = cv2.imread(str(item["image_path_b"]), cv2.IMREAD_COLOR)
    if image_a is None:
        raise RuntimeError(f"Failed to load image: {item['image_path_a']}")
    if image_b is None:
        raise RuntimeError(f"Failed to load image: {item['image_path_b']}")

    h1, w1 = image_a.shape[:2]
    h2, w2 = image_b.shape[:2]
    boxes_a = load_yolo_boxes(item["label_path_a"], w1, h1)
    boxes_b = load_yolo_boxes(item["label_path_b"], w2, h2)

    patch_rng = random.Random(item["seed"] + 11)
    motion_rng = random.Random(item["seed"] + 23)
    leaf_rng = random.Random(item["seed"] + 37)
    canvas_rng = random.Random(item["seed"] + 53)
    blur_rng = random.Random(item["seed"] + 71)

    stitched, stitched_boxes, frame_w, frame_h = build_stitched_pair(image_a, boxes_a, image_b, boxes_b)
    patch_h, patch_w = stitched.shape[:2]
    motion_plan = build_motion_plan(frame_w, frame_h, patch_w, item["motion_recipe"], patch_rng)
    frames_per_seq = motion_plan["frames"]
    leaf_move_left = not sequence_moves_left(motion_plan["pattern"])
    leaf_start_frame, leaf_end_frame = find_visibility_window(
        motion_plan["patch_x"],
        motion_plan["patch_y"],
        patch_w,
        patch_h,
        frame_w,
        frame_h,
        threshold=0.1,
    )

    motion_spec = sample_handheld_motion_spec(patch_w, patch_h, motion_rng)
    local_x_offsets, local_y_offsets = build_local_camera_offsets(patch_w, patch_h, frames_per_seq, motion_spec, motion_rng)
    leaf_tracks = sample_leaf_tracks_custom(
        patch_w,
        patch_h,
        frames_per_seq,
        leaf_assets,
        leaf_move_left,
        leaf_start_frame,
        leaf_end_frame,
        leaf_rng,
    )
    canvas, shifted_boxes, crop_x, crop_y = build_expanded_canvas(stitched, stitched_boxes, frame_w, frame_h, canvas_rng)
    canvas_h, canvas_w = canvas.shape[:2]

    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)

    gt_lines: list[str] = []
    det_lines: list[str] = []
    used_identity_ids: set[int] = set()
    output_identity_map: dict[int, int] = {}
    next_output_identity = 1

    for frame_idx in range(frames_per_seq):
        affine = build_affine(
            canvas_w,
            canvas_h,
            frame_idx,
            frames_per_seq,
            motion_spec,
            local_x_offsets[frame_idx],
            local_y_offsets[frame_idx],
        )
        warped_canvas = cv2.warpAffine(
            canvas,
            affine,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        patch = warped_canvas[crop_y : crop_y + patch_h, crop_x : crop_x + patch_w].copy()

        patch_boxes = []
        for box in shifted_boxes:
            transformed = transform_box_to_patch(box, affine, patch_w, patch_h, crop_x, crop_y)
            if transformed is not None:
                patch_boxes.append((box.track_id, box.class_id, transformed))

        patch, patch_occ_mask = overlay_leaf_occlusions(patch, leaf_assets, leaf_tracks, frame_idx, frames_per_seq)
        patch_offset_x = int(round(motion_plan["patch_x"][frame_idx]))
        patch_offset_y = int(round(motion_plan["patch_y"][frame_idx]))
        frame = paste_patch_to_black(patch, patch_offset_x, patch_offset_y, frame_w, frame_h)
        frame = maybe_blur(frame, frame_idx, frames_per_seq, blur_rng)

        frame_name = f"{frame_idx + 1:06d}.jpg"
        if not cv2.imwrite(str(img_dir / frame_name), frame):
            raise RuntimeError(f"Failed to write frame: {img_dir / frame_name}")

        for internal_identity_id, class_id, patch_box in patch_boxes:
            if is_box_occluded(patch_box, patch_occ_mask, threshold=0.3):
                continue
            final_box = translate_box(patch_box, patch_offset_x, patch_offset_y, frame_w, frame_h)
            if final_box is None:
                continue
            if internal_identity_id not in output_identity_map:
                output_identity_map[internal_identity_id] = next_output_identity
                next_output_identity += 1
            identity_id = output_identity_map[internal_identity_id]
            used_identity_ids.add(identity_id)
            x, y, bw, bh = final_box
            gt_lines.append(
                f"{frame_idx + 1},{identity_id},{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},1,{class_id},1.00"
            )
            det_lines.append(
                f"{frame_idx + 1},{identity_id},{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},1.00,-1,-1,-1"
            )

    write_seqinfo(seq_dir, seq_name, frame_w, frame_h, frames_per_seq, frame_rate)
    write_mot_annotations(seq_dir, gt_lines, det_lines)

    return {
        "sequence": seq_name,
        "source_images": [str(item["image_path_a"]), str(item["image_path_b"])],
        "source_labels": [str(item["label_path_a"]), str(item["label_path_b"])],
        "width": frame_w,
        "height": frame_h,
        "frames": frames_per_seq,
        "objects": len(used_identity_ids),
        "pattern": motion_plan["pattern"],
        "is_double_direction": bool(item["motion_recipe"]["is_double"]),
    }


def build_motion_recipes(total_sequences: int, rng: random.Random) -> list[dict]:
    if total_sequences <= 0:
        return []

    double_count = int(round(total_sequences * 0.4))
    flags = [True] * double_count + [False] * (total_sequences - double_count)
    rng.shuffle(flags)
    recipes = [sample_motion_recipe(flag, random.Random(rng.randint(0, 10**9))) for flag in flags]
    return recipes


def group_candidates_by_height(split_name: str, candidates: Sequence[tuple[Path, Path]]) -> dict[int, list[tuple[Path, Path]]]:
    groups: dict[int, list[tuple[Path, Path]]] = defaultdict(list)
    for image_path, label_path in candidates:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image while grouping {split_name} candidates: {image_path}")
        height = int(image.shape[0])
        groups[height].append((image_path, label_path))

    valid_groups = {height: items for height, items in groups.items() if len(items) >= 2}
    if candidates and not valid_groups:
        raise ValueError(
            f"No same-height image pairs available for {split_name}. Need at least one height group containing 2 images."
        )
    return valid_groups


def sample_sequence_items(
    split_name: str,
    candidates_by_height: dict[int, list[tuple[Path, Path]]],
    num_sequences: int,
    motion_recipes: Sequence[dict],
    rng: random.Random,
) -> list[dict]:
    max_unique_pair_count = sum(len(items) // 2 for items in candidates_by_height.values())
    if num_sequences > 0 and max_unique_pair_count < 1:
        raise ValueError(
            f"Not enough same-height image/label pairs for {split_name}. Need at least 2 images in one height group."
        )
    if num_sequences > max_unique_pair_count:
        raise ValueError(
            f"Not enough unique same-height image pairs for {split_name}. "
            f"Need {num_sequences} pairs, but only {max_unique_pair_count} non-overlapping pairs are available."
        )
    if len(motion_recipes) != num_sequences:
        raise ValueError(
            f"Motion recipe count mismatch for {split_name}: expected {num_sequences}, got {len(motion_recipes)}."
        )

    remaining_by_height = {
        height: list(items)
        for height, items in candidates_by_height.items()
        if len(items) >= 2
    }
    items = []
    for seq_idx in range(num_sequences):
        eligible_heights = [height for height, entries in remaining_by_height.items() if len(entries) >= 2]
        if not eligible_heights:
            raise ValueError(
                f"Ran out of unique image pairs while sampling {split_name}. "
                f"Requested {num_sequences}, sampled {len(items)}."
            )
        weights = [len(remaining_by_height[height]) for height in eligible_heights]
        chosen_height = rng.choices(eligible_heights, weights=weights, k=1)[0]
        chosen_pair = rng.sample(remaining_by_height[chosen_height], 2)
        (image_path_a, label_path_a), (image_path_b, label_path_b) = chosen_pair
        for pair in chosen_pair:
            remaining_by_height[chosen_height].remove(pair)
        items.append(
            {
                "sequence": f"Blueberry-{split_name.capitalize()}-{seq_idx + 1:02d}",
                "image_path_a": image_path_a,
                "label_path_a": label_path_a,
                "image_path_b": image_path_b,
                "label_path_b": label_path_b,
                "image_height": chosen_height,
                "motion_recipe": motion_recipes[seq_idx],
                "seed": rng.randint(0, 10**9),
            }
        )
    return items


def generate_split(
    split_name: str,
    items: Sequence[dict],
    leaf_assets,
    output_dir: Path,
    frame_rate: int,
) -> dict:
    split_root = output_dir / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    sequence_names = []
    for item in items:
        seq_name = item["sequence"]
        sequence_names.append(seq_name)
        seq_dir = split_root / seq_name
        manifest.append(
            create_sequence(
                item=item,
                leaf_assets=leaf_assets,
                seq_dir=seq_dir,
                seq_name=seq_name,
                frame_rate=frame_rate,
            )
        )

    export_seqmap(output_dir, split_name, sequence_names)
    summary = {
        "split": split_name,
        "num_sequences": len(sequence_names),
        "num_double_direction_sequences": sum(1 for item in items if item["motion_recipe"]["is_double"]),
        "sequences": manifest,
    }
    (output_dir / f"{split_name}_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()

    validate_dir(args.train_image_dir, "train_image_dir")
    validate_dir(args.train_label_dir, "train_label_dir")
    validate_dir(args.test_image_dir, "test_image_dir")
    validate_dir(args.test_label_dir, "test_label_dir")
    validate_dir(args.leaf_dir, "leaf_dir")

    leaf_assets = load_leaf_assets(args.leaf_dir)
    if not leaf_assets:
        raise ValueError(f"No valid RGBA leaf PNG assets found in: {args.leaf_dir}")

    train_candidates = collect_candidates(args.train_image_dir, args.train_label_dir)
    test_candidates = collect_candidates(args.test_image_dir, args.test_label_dir)
    train_candidates_by_height = group_candidates_by_height("train", train_candidates)
    test_candidates_by_height = group_candidates_by_height("test", test_candidates)

    prepare_output_dir(args.output_dir, args.overwrite)

    sampling_rng = random.Random(args.seed)
    total_sequences = args.train_sequences + args.test_sequences
    motion_recipes = build_motion_recipes(total_sequences, random.Random(sampling_rng.randint(0, 10**9)))
    train_motion_recipes = motion_recipes[: args.train_sequences]
    test_motion_recipes = motion_recipes[args.train_sequences :]

    train_items = sample_sequence_items(
        "train",
        train_candidates_by_height,
        args.train_sequences,
        train_motion_recipes,
        random.Random(sampling_rng.randint(0, 10**9)),
    )
    test_items = sample_sequence_items(
        "test",
        test_candidates_by_height,
        args.test_sequences,
        test_motion_recipes,
        random.Random(sampling_rng.randint(0, 10**9)),
    )

    train_summary = generate_split("train", train_items, leaf_assets, args.output_dir, args.frame_rate)
    test_summary = generate_split("test", test_items, leaf_assets, args.output_dir, args.frame_rate)

    summary = {
        "dataset_root": str(args.output_dir),
        "seed": args.seed,
        "double_direction_ratio_target": 0.4,
        "double_direction_sequences": sum(1 for recipe in motion_recipes if recipe["is_double"]),
        "total_sequences": total_sequences,
        "train": train_summary,
        "test": test_summary,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
