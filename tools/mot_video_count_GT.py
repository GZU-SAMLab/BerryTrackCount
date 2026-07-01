#!/usr/bin/env python3
"""Generate per-sequence counting ground truth CSV for blueberry MOT data."""

from __future__ import annotations

import argparse
import configparser
import csv
import logging
from collections import Counter
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "Flower", 1: "Green", 2: "Light Purple", 3: "Blue"}
CSV_HEADERS = ["video_name", "frames_num", "Flower", "Green", "Light Purple", "Blue", "total"]
DEFAULT_MOT_ROOT = Path("/home/wh1234_/data/blueberry_mot_stitched_walk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MOT counting ground truth CSV.")
    parser.add_argument("--mot-root", type=Path, default=DEFAULT_MOT_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=Path("dataset/mot_Count_GT/stitched_walk_count_GT.csv"))
    return parser.parse_args()


def discover_sequences(mot_root: Path, split: str) -> list[Path]:
    split_dir = mot_root / split
    if split_dir.is_dir():
        seq_dirs = sorted(path for path in split_dir.iterdir() if (path / "gt" / "gt.txt").exists())
        if seq_dirs:
            return seq_dirs

    seq_dirs = sorted(path.parent.parent for path in mot_root.rglob("gt.txt") if path.parent.name == "gt")
    return [path for path in seq_dirs if path.is_dir()]


def read_frames_num(seq_dir: Path) -> int:
    seqinfo_path = seq_dir / "seqinfo.ini"
    if not seqinfo_path.exists():
        return 0

    config = configparser.ConfigParser()
    config.read(seqinfo_path, encoding="utf-8")
    return config.getint("Sequence", "seqLength", fallback=0)


def count_sequence(seq_dir: Path) -> dict[str, int | str]:
    gt_path = seq_dir / "gt" / "gt.txt"
    track_to_class: dict[int, int] = {}

    with gt_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            parts = line.strip().split(",")
            if len(parts) < 8:
                logger.warning("Skip malformed row: %s:%d", gt_path, line_no)
                continue

            track_id = int(float(parts[1]))
            class_id = int(float(parts[7]))
            prev_class = track_to_class.get(track_id)
            if prev_class is not None and prev_class != class_id:
                logger.warning(
                    "Inconsistent class for seq=%s track_id=%d: %d -> %d",
                    seq_dir.name,
                    track_id,
                    prev_class,
                    class_id,
                )
                continue
            track_to_class[track_id] = class_id

    counts = Counter(CLASS_NAMES[class_id] for class_id in track_to_class.values() if class_id in CLASS_NAMES)
    row = {
        "video_name": seq_dir.name,
        "frames_num": read_frames_num(seq_dir),
        "Flower": counts.get("Flower", 0),
        "Green": counts.get("Green", 0),
        "Light Purple": counts.get("Light Purple", 0),
        "Blue": counts.get("Blue", 0),
    }
    row["total"] = row["Flower"] + row["Green"] + row["Light Purple"] + row["Blue"]
    return row


def write_csv(rows: list[dict[str, int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seq_dirs = discover_sequences(args.mot_root, args.split)
    if not seq_dirs:
        raise FileNotFoundError(f"No MOT sequences found under: {args.mot_root}")

    logger.info("Found %d sequences under %s", len(seq_dirs), args.mot_root)
    rows = []
    for index, seq_dir in enumerate(seq_dirs, start=1):
        row = count_sequence(seq_dir)
        rows.append(row)
        logger.info(
            "[%d/%d] %s | frames=%d | Flower=%d | Green=%d | Light Purple=%d | Blue=%d | total=%d",
            index,
            len(seq_dirs),
            row["video_name"],
            row["frames_num"],
            row["Flower"],
            row["Green"],
            row["Light Purple"],
            row["Blue"],
            row["total"],
        )

    rows.sort(key=lambda item: str(item["video_name"]))
    write_csv(rows, args.output)
    logger.info("Saved CSV to %s", args.output)


if __name__ == "__main__":
    main()
