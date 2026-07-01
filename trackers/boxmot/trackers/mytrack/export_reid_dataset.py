"""Export blueberry MOT annotations into an image-folder ReID dataset.

Default assumptions match ``dataset/blueberry_mot_stitched_walk``:
  - MOT-style sequence layout with ``img1/``, ``gt/gt.txt`` and ``seqinfo.ini``
  - class ids 0..3 mapped to Flower / Green / Light Purple / Blue
  - validation split carved from the training split unless told otherwise

By default, identities are split by ``(sequence, track_id, class_id)`` instead
of only ``(sequence, track_id)``. This is deliberate: for blueberry maturity
labels, a track that receives different class labels across frames introduces
label noise for ReID training, so the exporter keeps those class segments apart.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASS_NAMES = {
    0: "Flower",
    1: "Green",
    2: "Light_Purple",
    3: "Blue",
}


@dataclass(frozen=True)
class CropRecord:
    source_split: str
    target_split: str
    sequence: str
    frame: int
    track_id: int
    class_id: int
    class_name: str
    image_path: Path
    x: float
    y: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export blueberry MOT tracks into a ReID image-folder dataset."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/blueberry_mot_stitched_walk"),
        help="Root folder containing MOT-style train/ and test/ sequence folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/blueberry_reid"),
        help="Target image-folder dataset root.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest path. Defaults to <data-root>/train_manifest.json when using tail split.",
    )
    parser.add_argument(
        "--val-source",
        type=str,
        default="tail",
        choices=["tail", "test"],
        help="Validation images come from the tail of the train split or from test/ sequences.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=4,
        help="Number of tail train sequences used for val when --val-source=tail.",
    )
    parser.add_argument(
        "--class-ids",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3],
        help="Keep only these category ids.",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=8.0,
        help="Minimum width and height for a crop to be exported.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=64.0,
        help="Minimum box area for a crop to be exported.",
    )
    parser.add_argument(
        "--visibility-thresh",
        type=float,
        default=0.0,
        help="Minimum MOT visibility value to keep a box.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.0,
        help="Minimum MOT confidence/mark value to keep a box.",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.08,
        help="Relative margin added on each side before cropping.",
    )
    parser.add_argument(
        "--min-samples-per-id",
        type=int,
        default=2,
        help="Drop identities with fewer than this many crops inside a split.",
    )
    parser.add_argument(
        "--split-track-by-class",
        action="store_true",
        help="Split one MOT track into separate ReID identities when class labels differ across frames.",
    )
    parser.add_argument(
        "--merge-track-classes",
        dest="split_track_by_class",
        action="store_false",
        help="Keep one identity per (sequence, track_id) even if class labels vary across frames.",
    )
    parser.set_defaults(split_track_by_class=True)
    parser.add_argument(
        "--image-format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Image format used when saving crops.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality used when --image-format=jpg.",
    )
    return parser.parse_args()


def load_manifest_sequences(manifest_path: Path) -> list[str]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    sequences = [item["sequence"] for item in payload.get("sequences", [])]
    if not sequences:
        raise ValueError(f"No sequences found in manifest: {manifest_path}")
    return sequences


def load_seqmap_sequences(seqmap_path: Path) -> list[str]:
    if not seqmap_path.is_file():
        return []
    sequences: list[str] = []
    with seqmap_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.lower() == "name":
                continue
            sequences.append(line)
    return sequences


def existing_sequence_dirs(split_root: Path) -> list[str]:
    if not split_root.is_dir():
        return []
    return sorted(path.name for path in split_root.iterdir() if path.is_dir())


def resolve_train_sequences(data_root: Path, manifest: Path | None) -> list[str]:
    candidates: list[str] = []
    if manifest is not None and manifest.is_file():
        candidates = load_manifest_sequences(manifest)
    if not candidates:
        candidates = load_seqmap_sequences(data_root / "seqmaps" / "train.txt")
    if not candidates:
        candidates = existing_sequence_dirs(data_root / "train")

    existing = set(existing_sequence_dirs(data_root / "train"))
    return [name for name in candidates if name in existing]


def resolve_test_sequences(data_root: Path) -> list[str]:
    candidates = load_seqmap_sequences(data_root / "seqmaps" / "test.txt")
    if not candidates:
        candidates = existing_sequence_dirs(data_root / "test")
    existing = set(existing_sequence_dirs(data_root / "test"))
    return [name for name in candidates if name in existing]


def read_seqinfo(seq_dir: Path) -> tuple[int, int, str]:
    parser = ConfigParser()
    seqinfo_path = seq_dir / "seqinfo.ini"
    if not seqinfo_path.is_file():
        raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo_path}")
    parser.read(seqinfo_path, encoding="utf-8")
    section = parser["Sequence"]
    width = int(section.get("imWidth"))
    height = int(section.get("imHeight"))
    im_dir = section.get("imDir", section.get("imdir", "img1"))
    return width, height, im_dir


def collect_image_paths(img_dir: Path) -> dict[int, Path]:
    frame_to_path: dict[int, Path] = {}
    for path in sorted(img_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            frame_idx = int(path.stem)
        except ValueError:
            continue
        frame_to_path[frame_idx] = path
    return frame_to_path


def clamp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1_i = max(0, min(width - 1, int(math.floor(x1))))
    y1_i = max(0, min(height - 1, int(math.floor(y1))))
    x2_i = max(1, min(width, int(math.ceil(x2))))
    y2_i = max(1, min(height, int(math.ceil(y2))))
    if x2_i <= x1_i or y2_i <= y1_i:
        return None
    return x1_i, y1_i, x2_i, y2_i


def parse_gt_records(
    seq_dir: Path,
    source_split: str,
    target_split: str,
    allowed_classes: set[int],
    class_names: dict[int, str],
    min_box_size: float,
    min_area: float,
    visibility_thresh: float,
    conf_thresh: float,
) -> list[CropRecord]:
    width, height, im_dir_name = read_seqinfo(seq_dir)
    frame_to_path = collect_image_paths(seq_dir / im_dir_name)
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing gt.txt: {gt_path}")

    records: list[CropRecord] = []
    with gt_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue

            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            class_id = int(float(parts[7]))
            visibility = float(parts[8]) if len(parts) > 8 else 1.0

            if track_id <= 0:
                continue
            if class_id not in allowed_classes:
                continue
            if conf < conf_thresh:
                continue
            if visibility < visibility_thresh:
                continue
            if min(w, h) < min_box_size:
                continue
            if w * h < min_area:
                continue

            image_path = frame_to_path.get(frame)
            if image_path is None:
                continue

            records.append(
                CropRecord(
                    source_split=source_split,
                    target_split=target_split,
                    sequence=seq_dir.name,
                    frame=frame,
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_names.get(class_id, f"class_{class_id}"),
                    image_path=image_path,
                    x=max(0.0, min(x, width - 1.0)),
                    y=max(0.0, min(y, height - 1.0)),
                    w=max(0.0, min(w, width)),
                    h=max(0.0, min(h, height)),
                )
            )
    return records


def write_crop(
    record: CropRecord,
    output_path: Path,
    crop_margin: float,
    jpeg_quality: int,
) -> bool:
    image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False

    img_h, img_w = image.shape[:2]
    margin_w = record.w * crop_margin
    margin_h = record.h * crop_margin
    box = clamp_box(
        record.x - margin_w,
        record.y - margin_h,
        record.x + record.w + margin_w,
        record.y + record.h + margin_h,
        img_w,
        img_h,
    )
    if box is None:
        return False

    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    else:
        params = []
    return bool(cv2.imwrite(str(output_path), crop, params))


def identity_key(record: CropRecord, split_track_by_class: bool) -> str:
    if split_track_by_class:
        return f"{record.class_name}__{record.sequence}__{record.track_id:06d}"
    return f"{record.sequence}__{record.track_id:06d}"


def export_split(
    records: list[CropRecord],
    output_root: Path,
    crop_margin: float,
    image_format: str,
    jpeg_quality: int,
    min_samples_per_id: int,
    split_track_by_class: bool,
) -> dict[str, object]:
    by_identity: dict[str, list[CropRecord]] = defaultdict(list)
    track_class_sets: dict[tuple[str, int], set[int]] = defaultdict(set)

    for record in records:
        by_identity[identity_key(record, split_track_by_class)].append(record)
        track_class_sets[(record.sequence, record.track_id)].add(record.class_id)

    exported_identities = 0
    exported_images = 0
    skipped_identities = 0
    class_image_counts: dict[str, int] = defaultdict(int)
    class_identity_counts: dict[str, int] = defaultdict(int)
    metadata: dict[str, dict[str, object]] = {}
    split_multi_class_tracks = sum(
        1 for classes in track_class_sets.values() if split_track_by_class and len(classes) > 1
    )

    output_root.mkdir(parents=True, exist_ok=True)
    for identity, items in sorted(by_identity.items()):
        items.sort(key=lambda item: (item.frame, item.image_path.name))
        if len(items) < min_samples_per_id:
            skipped_identities += 1
            continue

        dominant_class = max(
            (item.class_name for item in items),
            key=lambda class_name: sum(1 for item in items if item.class_name == class_name),
        )
        sequence = items[0].sequence
        track_id = items[0].track_id
        original_track_classes = sorted(track_class_sets[(sequence, track_id)])
        identity_dir = output_root / identity
        saved = 0
        class_histogram: dict[int, int] = defaultdict(int)
        for index, item in enumerate(items, start=1):
            class_histogram[item.class_id] += 1
            filename = f"{index:06d}.{image_format}"
            output_path = identity_dir / filename
            if write_crop(item, output_path, crop_margin, jpeg_quality):
                saved += 1

        if saved < min_samples_per_id:
            skipped_identities += 1
            if identity_dir.exists():
                shutil.rmtree(identity_dir)
            continue

        exported_identities += 1
        exported_images += saved
        class_image_counts[dominant_class] += saved
        class_identity_counts[dominant_class] += 1
        metadata[identity] = {
            "sequence": sequence,
            "track_id": track_id,
            "dominant_class": dominant_class,
            "class_histogram": {str(k): int(v) for k, v in sorted(class_histogram.items())},
            "original_track_classes": original_track_classes,
            "num_images": saved,
        }

    return {
        "num_identities": exported_identities,
        "num_images": exported_images,
        "skipped_identities": skipped_identities,
        "class_identity_counts": dict(sorted(class_identity_counts.items())),
        "class_image_counts": dict(sorted(class_image_counts.items())),
        "split_multi_class_tracks": split_multi_class_tracks,
        "identities": metadata,
    }


def clear_split_dir(split_dir: Path) -> None:
    if split_dir.exists():
        shutil.rmtree(split_dir)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    manifest_path = (
        (args.manifest.resolve() if args.manifest is not None else None)
        or (data_root / "train_manifest.json")
    )
    output_root = args.output_root.resolve()
    class_names = {class_id: DEFAULT_CLASS_NAMES.get(class_id, f"class_{class_id}") for class_id in args.class_ids}
    allowed_classes = set(args.class_ids)

    train_sequence_names = resolve_train_sequences(data_root, manifest_path)
    if not train_sequence_names:
        raise ValueError(f"No train sequences found under {data_root / 'train'}")

    if args.val_source == "tail":
        if len(train_sequence_names) <= args.val_count:
            raise ValueError(
                f"val-count={args.val_count} leaves no training sequences in {data_root / 'train'}"
            )
        train_split_sequences = train_sequence_names[:-args.val_count]
        val_split_sequences = train_sequence_names[-args.val_count:]
        val_source_split = "train"
    else:
        train_split_sequences = train_sequence_names
        val_split_sequences = resolve_test_sequences(data_root)
        val_source_split = "test"
        if not val_split_sequences:
            raise ValueError(f"No test sequences found under {data_root / 'test'}")

    all_records: dict[str, list[CropRecord]] = {"train": [], "val": []}
    sequence_plan = (
        ("train", "train", train_split_sequences),
        (val_source_split, "val", val_split_sequences),
    )
    for source_split, target_split, sequence_names in sequence_plan:
        for sequence_name in sequence_names:
            seq_dir = data_root / source_split / sequence_name
            if not seq_dir.is_dir():
                continue
            split_records = parse_gt_records(
                seq_dir=seq_dir,
                source_split=source_split,
                target_split=target_split,
                allowed_classes=allowed_classes,
                class_names=class_names,
                min_box_size=args.min_box_size,
                min_area=args.min_area,
                visibility_thresh=args.visibility_thresh,
                conf_thresh=args.conf_thresh,
            )
            all_records[target_split].extend(split_records)

    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "data_root": str(data_root),
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "class_ids": sorted(allowed_classes),
        "class_names": {str(k): v for k, v in sorted(class_names.items())},
        "min_box_size": args.min_box_size,
        "min_area": args.min_area,
        "visibility_thresh": args.visibility_thresh,
        "conf_thresh": args.conf_thresh,
        "crop_margin": args.crop_margin,
        "min_samples_per_id": args.min_samples_per_id,
        "val_source": args.val_source,
        "val_count": args.val_count,
        "split_track_by_class": args.split_track_by_class,
        "train_sequences": train_split_sequences,
        "val_sequences": val_split_sequences,
    }

    for split_name in ("train", "val"):
        split_output = output_root / split_name
        clear_split_dir(split_output)
        split_summary = export_split(
            records=all_records[split_name],
            output_root=split_output,
            crop_margin=args.crop_margin,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
            min_samples_per_id=args.min_samples_per_id,
            split_track_by_class=args.split_track_by_class,
        )
        summary[split_name] = split_summary

    summary_path = output_root / "metadata.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
