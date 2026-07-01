"""Run the blueberry ReID pipeline with an osnet_ain_x1_0 backbone.

This script is an orchestration entrypoint for the workflow:
1. Export MOT tracks into an image-folder ReID dataset.
2. Train a BoxMOT-compatible ReID checkpoint with ``osnet_ain_x1_0``.

The underlying implementations live under ``trackers/boxmot/trackers/mytrack``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "trackers" / "boxmot").is_dir():
            return candidate
    raise RuntimeError(f"Failed to infer repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__))
EXPORT_SCRIPT = REPO_ROOT / "trackers" / "boxmot" / "trackers" / "mytrack" / "export_reid_dataset.py"
TRAIN_SCRIPT = REPO_ROOT / "trackers" / "boxmot" / "trackers" / "mytrack" / "train_reid.py"
DEFAULT_DATA_ROOT = REPO_ROOT / "dataset" / "blueberry_mot_stitched_walk"
DEFAULT_REID_ROOT = REPO_ROOT / "dataset" / "blueberry_reid_ain"
DEFAULT_OUTPUT = REPO_ROOT / "weights" / "osnet_ain_x1_0_blueberry.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export blueberry ReID crops and train an osnet_ain_x1_0 model."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--reid-root", type=Path, default=DEFAULT_REID_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--last-output",
        type=Path,
        default=None,
        help="Optional explicit path to the last checkpoint. Defaults to <output_stem>_last.pt.",
    )
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="Optional local checkpoint used to initialize matching layers.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume automatically from the last checkpoint if it exists.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use osnet_ain_x1_0 ImageNet initialization when supported by the environment.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip crop export and train using the existing ReID dataset.",
    )
    parser.add_argument("--val-source", type=str, default="tail", choices=["tail", "test"])
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--class-ids", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--min-box-size", type=float, default=8.0)
    parser.add_argument("--min-area", type=float, default=64.0)
    parser.add_argument("--visibility-thresh", type=float, default=0.0)
    parser.add_argument("--conf-thresh", type=float, default=0.0)
    parser.add_argument("--crop-margin", type=float, default=0.08)
    parser.add_argument("--min-samples-per-id", type=int, default=2)
    parser.add_argument("--merge-track-classes", action="store_true")
    parser.add_argument("--image-format", type=str, default="jpg", choices=["jpg", "png"])
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--instances-per-identity", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--tri-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Keep aligned with BoxMOT inference preprocessing unless you also change inference.",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--color-jitter", type=float, default=0.1)
    parser.add_argument("--random-erasing", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--early-stop-patience", type=int, default=50)
    parser.add_argument("--early-stop-min-delta", type=float, default=5e-4)
    return parser.parse_args()


def ensure_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output = args.output.resolve()
    if "osnet_ain_x1_0" not in output.name:
        raise ValueError(
            f"Output filename must contain 'osnet_ain_x1_0' for BoxMOT compatibility: {output}"
        )
    last_output = (
        args.last_output.resolve()
        if args.last_output is not None
        else output.with_name(f"{output.stem}_last{output.suffix}")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    args.reid_root.resolve().mkdir(parents=True, exist_ok=True)
    return output, last_output


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("running:", " ".join(f'"{part}"' if " " in part else part for part in command))
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def reid_dataset_ready(reid_root: Path) -> bool:
    train_dir = reid_root / "train"
    val_dir = reid_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        return False

    train_ids = [path for path in train_dir.iterdir() if path.is_dir()]
    val_ids = [path for path in val_dir.iterdir() if path.is_dir()]
    return bool(train_ids and val_ids)


def build_export_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--data-root",
        str(args.data_root.resolve()),
        "--output-root",
        str(args.reid_root.resolve()),
        "--val-source",
        args.val_source,
        "--val-count",
        str(args.val_count),
        "--class-ids",
        *[str(class_id) for class_id in args.class_ids],
        "--min-box-size",
        str(args.min_box_size),
        "--min-area",
        str(args.min_area),
        "--visibility-thresh",
        str(args.visibility_thresh),
        "--conf-thresh",
        str(args.conf_thresh),
        "--crop-margin",
        str(args.crop_margin),
        "--min-samples-per-id",
        str(args.min_samples_per_id),
        "--image-format",
        args.image_format,
        "--jpeg-quality",
        str(args.jpeg_quality),
    ]
    if args.merge_track_classes:
        command.append("--merge-track-classes")
    return command


def build_train_command(args: argparse.Namespace, output: Path, last_output: Path) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data-root",
        str(args.reid_root.resolve()),
        "--output",
        str(output),
        "--arch",
        "osnet_ain_x1_0",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--instances-per-identity",
        str(args.instances_per_identity),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--margin",
        str(args.margin),
        "--ce-weight",
        str(args.ce_weight),
        "--tri-weight",
        str(args.tri_weight),
        "--label-smoothing",
        str(args.label_smoothing),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--hflip-prob",
        str(args.hflip_prob),
        "--color-jitter",
        str(args.color_jitter),
        "--random-erasing",
        str(args.random_erasing),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-min-delta",
        str(args.early_stop_min_delta),
    ]
    if args.pretrained and args.init_weights is None:
        command.append("--pretrained")
    if args.init_weights is not None:
        command.extend(["--init-weights", str(args.init_weights.resolve())])
    if args.resume and last_output.is_file():
        command.extend(["--resume", str(last_output)])
    return command


def main() -> None:
    args = parse_args()
    output, last_output = ensure_paths(args)

    env = os.environ.copy()
    trackers_root = REPO_ROOT / "trackers"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{trackers_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(trackers_root)
    )

    print(f"repo_root={REPO_ROOT}")
    print(f"data_root={args.data_root.resolve()}")
    print(f"reid_root={args.reid_root.resolve()}")
    print(f"output={output}")
    print(f"last_output={last_output}")
    print("arch=osnet_ain_x1_0")

    should_skip_export = args.skip_export or reid_dataset_ready(args.reid_root.resolve())
    if should_skip_export:
        if args.skip_export:
            print("skip_export=True, using existing ReID dataset.")
        else:
            print("detected existing ReID dataset under reid_root, skipping export.")
    else:
        run_command(build_export_command(args), env)

    run_command(build_train_command(args, output, last_output), env)


if __name__ == "__main__":
    main()
