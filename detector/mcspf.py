"""
Training helper for mcspf-YOLO8/mcspf-YOLO11 models across n/s/m/l scales.
"""

import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from typing import Dict, Any
import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO


# SUPPORTED_SCALES = ("n", "s", "m", "l")
SUPPORTED_SCALES = ("s", "m")

BASE_TRAIN_ARGS: Dict[str, Any] = {
    "data": "configs/data/mydata.yaml",
    "epochs": 600,
    "imgsz": 640,
    "device": 0,
    "project": "runs/mcspf",
    "optimizer": "AdamW",
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "seed": 42,
    "close_mosaic": 10,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "hsv_h": 0.01,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "amp": True,
    "fraction": 1.0,
    "dropout": 0.0,
}


mcspf_CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "v8": {
        "n": {
            "config": "configs/detector/MCSPF/mcspf-yolov8n.yaml",
            "name": "mcspf-YOLOv8n",
            "train_args": {
                "batch": 64,
                "workers": 14,
                "lr0": 0.001,
                "patience": 50,
                "lrf": 0.1,
                "mosaic": 1.0,
                "mixup": 0.1,
            },
        },
        "s": {
            "config": "configs/detector/MCSPF/mcspf-yolov8s.yaml",
            "name": "mcspf-YOLOv8s",
            "train_args": {
                "batch": 48,
                "workers": 12,
                "lr0": 0.001,
                "patience": 50,
                "lrf": 0.1,
                "mosaic": 1.0,
                "mixup": 0.1,
            },
        },
        "m": {
            "config": "configs/detector/MCSPF/mcspf-yolov8m.yaml",
            "name": "mcspf-YOLOv8m",
            "train_args": {
                "batch": 32,
                "workers": 10,
                "lr0": 0.0008,
                "patience": 80,
                "lrf": 0.1,
                "mosaic": 1.0,
                "mixup": 0.1,
            },
        },
        "l": {
            "config": "configs/detector/MCSPF/mcspf-yolov8l.yaml",
            "name": "mcspf-YOLOv8l",
            "train_args": {
                "batch": 24,
                "workers": 8,
                "lr0": 0.0006,
                "patience": 100,
                "lrf": 0.1,
                "mosaic": 1.0,
                "mixup": 0.1,
            },
        },
    },
    "11": {
        "n": {
            "config": "configs/detector/MCSPF/mcspf-yolo11n.yaml",
            "name": "mcspf-YOLO11n",
            "train_args": {
                "batch": 64,
                "workers": 14,
                "lr0": 0.001,
                "patience": 50,
                "lrf": 0.01,
                "mosaic": 1.0,
                "mixup": 0.15,
                "degrees": 15.0,
            },
        },
        "s": {
            "config": "configs/detector/MCSPF/mcspf-yolo11s.yaml",
            "name": "mcspf-YOLO11s",
            "train_args": {
                "batch": 48,
                "workers": 12,
                "lr0": 0.001,
                "patience": 50,
                "lrf": 0.01,
                "mosaic": 1.0,
                "mixup": 0.15,
                "degrees": 15.0,
            },
        },
        "m": {
            "config": "configs/detector/MCSPF/mcspf-yolo11m.yaml",
            "name": "mcspf-YOLO11m",
            "train_args": {
                "batch": 32,
                "workers": 10,
                "lr0": 0.0008,
                "patience": 80,
                "lrf": 0.01,
                "mosaic": 1.0,
                "mixup": 0.15,
                "degrees": 15.0,
            },
        },
        "l": {
            "config": "configs/detector/MCSPF/mcspf-yolo11l.yaml",
            "name": "mcspf-YOLO11l",
            "train_args": {
                "batch": 24,
                "workers": 8,
                "lr0": 0.0006,
                "patience": 100,
                "lrf": 0.01,
                "mosaic": 1.0,
                "mixup": 0.15,
                "degrees": 15.0,
            },
        },
    },
}


def build_train_args(version_key: str, model_size: str) -> Dict[str, Any]:
    """Merge base arguments with model- and scale-specific overrides."""
    cfg = mcspf_CONFIGS[version_key][model_size]
    args = BASE_TRAIN_ARGS.copy()
    args.update(cfg["train_args"])
    args["name"] = cfg["name"]
    return args


def _train(version_key: str, model_size: str):
    cfg = mcspf_CONFIGS[version_key][model_size]
    args = build_train_args(version_key, model_size)
    model = YOLO(cfg["config"])

    print("=" * 60)
    print(f"Training {cfg['name']}")
    print("=" * 60)
    result = model.train(**args)
    print(f"{cfg['name']} training complete")
    return result


def train_mcspf_model(model_size: str, model_version: str = "both"):
    """Train mcspf model(s) for a specific scale."""
    if model_size not in SUPPORTED_SCALES:
        raise ValueError(f"Unsupported model size {model_size}, choose from {SUPPORTED_SCALES}.")

    allowed_versions = ("v8", "11", "both")
    if model_version not in allowed_versions:
        raise ValueError(f"model_version must be in {allowed_versions}")

    results = {}
    if model_version in ("v8", "both"):
        results["v8"] = _train("v8", model_size)
    if model_version in ("11", "both"):
        results["11"] = _train("11", model_size)
    return results


def train_all_models(model_version: str = "both"):
    """Train n/s/m/l sequentially."""
    summary = {}
    for size in SUPPORTED_SCALES:
        try:
            summary[size] = train_mcspf_model(size, model_version=model_version)
        except Exception as exc:
            print(f"Training {size} failed: {exc}")
            summary[size] = None
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("mcspf-YOLO training")
    print("1. Train specific scale")
    print("2. Train all scales (default)")
    print("=" * 60)
    # mode_choice = input("Select mode (1/2, default 2): ").strip()
    mode_choice = "2"

    print("\nModel version:")
    print("1. mcspf-YOLO8")
    print("2. mcspf-YOLO11")
    print("3. Train both (default)")
    # version_choice = input("Select version (1/2/3, default 3): ").strip()
    version_choice = "3"
    version_map = {"1": "v8", "2": "11", "3": "both"}
    model_version = version_map.get(version_choice, "both")
    print(f"Training {model_version} models")

    if mode_choice == "1":
        scale = input("Enter model size (n/s/m/l): ").strip().lower()
        train_mcspf_model(scale, model_version=model_version)
    else:
        train_all_models(model_version=model_version)
