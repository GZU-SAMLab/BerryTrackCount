"""
Training helper for FBRT-YOLO8/FBRT-YOLO11 models across n/s/m/l scales.
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

allowed_versions = ("v8", "11", "both")

BASE_TRAIN_ARGS: Dict[str, Any] = {
    "data": "configs/data/mydata.yaml",
    "epochs": 300,
    "imgsz": 640,
    "device": 0,
    "project": "runs/FCM",
    "optimizer": "AdamW",
    "seed": 42,
    "amp": True,
}


CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "v8": {
        "n": {
            "config": "configs/detector/FCM/fcm-yolov8n.yaml",
            "weights": "/home/wh1234_/code/Counting/weights/yolov8n.pt",
            "name": "FCM-YOLOv8n",
            "train_args": {
                "batch": 64,
                "workers": 14,
                "lr0": 0.001,
                "patience": 50,
                "lrf": 0.1,
            },
        },
        "s": {
            "config": "configs/detector/FCM/fcm-yolov8s.yaml",
            "weights": "/home/wh1234_/code/Counting/weights/yolov8s.pt",
            "name": "FCM-YOLOv8s",
            "train_args": {
                "batch": 48,
                "workers": 12,
                "lr0": 0.001,
                "patience": 50,
                "lrf": 0.1,
            },
        },
        "m": {
            "config": "configs/detector/FCM/fcm-yolov8m.yaml",
            "weights": "/home/wh1234_/code/Counting/weights/yolov8m.pt",
            "name": "FCM-YOLOv8m",
            "train_args": {
                "batch": 32,
                "workers": 10,
                "lr0": 0.0008,
                "patience": 80,
                "lrf": 0.1,
            },
        },
        "l": {
            "config": "configs/detector/FCM/fcm-yolov8l.yaml",
            "weights": "/home/wh1234_/code/Counting/weights/yolov8l.pt",
            "name": "FCM-YOLOv8l",
            "train_args": {
                "batch": 24,
                "workers": 8,
                "lr0": 0.0006,
                "patience": 100,
                "lrf": 0.1,
            },
        },
    },
    "11": {
        "n": {
            "config": "configs/detector/FCM/fcm-yolo11n.yaml",
            "weights": "/home/wh1234_/code/Counting/weights/yolo11n.pt",
            "name": "FCM-YOLO11n",
            "train_args": {
                "batch": 64,
                "workers": 14,
                "lr0": 0.001,
                "patience": 50,
                "lrf": 0.01,
            },
        },
        "s": {
            "config": "configs/detector/FCM/fcm-yolo11s.yaml",
            "weights": "/home/wh1234_/code/Counting/weights/yolo11s.pt",
            "name": "FCM-YOLO11s",
            "train_args": {
                "batch": 48,
                "workers": 12,
                "lr0": 0.001,
                "patience": 50,
                "lrf": 0.01,
            },
        },
        "m": {
            "config": "configs/detector/FCM/fcm-yolo11m.yaml",
            "weights": "/home/wh1234_/code/Counting/weights/yolo11m.pt",
            "name": "FCM-YOLO11m",
            "train_args": {
                "batch": 32,
                "workers": 10,
                "lr0": 0.0008,
                "patience": 80,
                "lrf": 0.01,
            },
        },
        "l": {
            "config": "configs/detector/FCM/fcm-yolo11l.yaml",
            "weights": "/home/wh1234_/code/Counting/weights/yolo11l.pt",
            "name": "FCM-YOLO11l",
            "train_args": {
                "batch": 24,
                "workers": 8,
                "lr0": 0.0006,
                "patience": 100,
                "lrf": 0.01,
            },
        },
    },
}


def get_pretrained_weight_path(version_key: str, model_size: str) -> str:
    """Get the path to pretrained COCO weights."""
    if version_key == "v8":
        weight_file = CONFIGS[version_key][model_size]["weights"]
    elif version_key == "11":
        weight_file = CONFIGS[version_key][model_size]["weights"]
    else:
        raise ValueError(f"Unsupported version_key: {version_key}")
    return weight_file


def build_train_args(version_key: str, model_size: str, use_pretrained: bool = False) -> Dict[str, Any]:
    """Merge base arguments with model- and scale-specific overrides."""
    cfg = CONFIGS[version_key][model_size]
    args = BASE_TRAIN_ARGS.copy()
    args.update(cfg["train_args"])
    
    # Modify name to indicate pretrained status
    if use_pretrained:
        args["name"] = f"{cfg['name']}-pretrained"
    else:
        args["name"] = f"{cfg['name']}-scratch"
    
    return args


def train_single_model(version_key: str, model_size: str, use_pretrained: bool = False):
    """Train a single model with or without pretrained weights."""
    cfg = CONFIGS[version_key][model_size]
    args = build_train_args(version_key, model_size, use_pretrained)
    
    # Load model with or without pretrained weights
    if use_pretrained:
        weight_path = get_pretrained_weight_path(version_key, model_size)
        print(f"Loading pretrained COCO weights from: {weight_path}")
        model = YOLO(cfg["config"])
        model.load(cfg["weights"])
    else:
        model = YOLO(cfg["config"])
    
    print("=" * 60)
    status = "with pretrained COCO weights" if use_pretrained else "from scratch"
    print(f"Training {cfg['name']} {status}")
    print("=" * 60)
    result = model.train(**args)
    print(f"{cfg['name']} training complete ({status})")
    return result


def train_fbrt_model(model_size: str, model_version: str = "both", load_pretrained: int = 0):
    """
    Train model(s) for a specific scale.
    
    Args:
        model_size: Model size (n/s/m/l)
        model_version: Model version ("v8", "11", or "both")
        load_pretrained: 0 = no pretrained, 1 = with pretrained, 2 = both
    """
    if model_size not in SUPPORTED_SCALES:
        raise ValueError(f"Unsupported model size {model_size}, choose from {SUPPORTED_SCALES}.")

    if model_version not in allowed_versions:
        raise ValueError(f"model_version must be in {allowed_versions}")
    
    if load_pretrained not in (0, 1, 2):
        raise ValueError(f"load_pretrained must be 0, 1, or 2, got {load_pretrained}")

    results = {}
    
    # Determine training modes based on load_pretrained flag
    if load_pretrained == 2:
        train_modes = [False, True]  # Train both from scratch and with pretrained
    elif load_pretrained == 1:
        train_modes = [True]  # Only with pretrained
    else:  # load_pretrained == 0
        train_modes = [False]  # Only from scratch
    
    for use_pretrained in train_modes:
        if model_version in ("v8", "both"):
            result_key = "v8" + ("_pretrained" if use_pretrained else "_scratch")
            if "v8" not in results:
                results["v8"] = {}
            results["v8"][result_key] = train_single_model("v8", model_size, use_pretrained)

        if model_version in ("11", "both"):
            result_key = "11" + ("_pretrained" if use_pretrained else "_scratch")
            if "11" not in results:
                results["11"] = {}
            results["11"][result_key] = train_single_model("11", model_size, use_pretrained)
        
    return results


def train_all_models(model_version: str = "both", load_pretrained: int = 0):
    """Train n/s/m/l sequentially."""
    summary = {}
    for size in SUPPORTED_SCALES:
        try:
            summary[size] = train_fbrt_model(size, model_version=model_version, load_pretrained=load_pretrained)
        except Exception as exc:
            print(f"Training {size} failed: {exc}")
            summary[size] = None
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("FCM模块替换backbone中的C2f/C3k2模块")
    print("1. Train specific scale")
    print("2. Train all scales (default)")
    print("=" * 60)
    # mode_choice = input("Select mode (1/2, default 2): ").strip()
    mode_choice = "2"

    print("\nModel version:")
    print("1. YOLO8")
    print("2. YOLO11")
    print("3. Train both (default)")
    # version_choice = input("Select version (1/2/3, default 3): ").strip()
    version_choice = "3"
    version_map = {"1": "v8", "2": "11", "3": "both"}
    model_version = version_map.get(version_choice, "both")
    print(f"Training {model_version} models")

    print("\nPretrained weights loading:")
    print("0. Train from scratch (no pretrained weights)")
    print("1. Train with pretrained COCO weights")
    print("2. Train both (from scratch and with pretrained weights)")
    # pretrained_choice = input("Select pretrained loading mode (0/1/3, default 0): ").strip()
    pretrained_choice = "2"
    pretrained_map = {"0": 0, "1": 1, "2": 2}
    load_pretrained = pretrained_map.get(pretrained_choice, 0)
    
    pretrained_desc = {
        0: "from scratch",
        1: "with pretrained COCO weights",
        2: "both (from scratch and with pretrained weights)"
    }
    print(f"Pretrained loading mode: {load_pretrained} - {pretrained_desc[load_pretrained]}")

    if mode_choice == "1":
        scale = input("Enter model size (n/s/m/l): ").strip().lower()
        train_fbrt_model(scale, model_version=model_version, load_pretrained=load_pretrained)
    else:
        train_all_models(model_version=model_version, load_pretrained=load_pretrained)
