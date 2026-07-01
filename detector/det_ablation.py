"""
Training script for detector ablation and baseline experiments.
"""

import gc
import logging
import os
import shutil
import sys
import time
import warnings
from typing import Any

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

SUPPORTED_SCALES = ("n", "s", "m", "l")

BASE_TRAIN_ARGS: dict[str, Any] = {
    "data": "configs/data/mydata_731.yaml",
    "epochs": 300,
    "imgsz": 640,
    "device": 0,
    "project": "runs/det_ablation",
    "optimizer": "AdamW",
    "seed": 42,
    "amp": True,
}

SCALE_ARGS: dict[str, dict[str, Any]] = {
    "n": {"batch": 64, "workers": 14, "lr0": 0.001, "patience": 50, "lrf": 0.01},
    "s": {"batch": 48, "workers": 12, "lr0": 0.001, "patience": 50, "lrf": 0.01},
    "m": {"batch": 32, "workers": 10, "lr0": 0.0008, "patience": 80, "lrf": 0.01},
    "l": {"batch": 24, "workers": 8, "lr0": 0.0006, "patience": 100, "lrf": 0.01},
}

EXPERIMENTS: dict[str, dict[str, Any]] = {
    # "fcm": {
    #     "scales": ("n", "s", "m", "l"),
    #     "base_config": "configs/detector/ablation/fcm-yolo11.yaml",
    #     "config": "configs/detector/ablation/fcm-yolo11{scale}.yaml",
    #     "name_prefix": "FCM-YOLO11",
    #     "weights": "weights/yolo11{scale}.pt",
    # },
    # "msdaf": {
    #     "scales": ("n", "s", "m", "l"),
    #     "base_config": "configs/detector/ablation/msdaf-yolo11.yaml",
    #     "config": "configs/detector/ablation/msdaf-yolo11{scale}.yaml",
    #     "name_prefix": "MSDAF-YOLO11",
    #     "weights": "/home/wh1234_/code/Counting/weights/yolo11{scale}.pt",
    # },
    # "yolov8": {
    #     "scales": ("n", "s", "m"),
    #     "config": "ultralytics/cfg/models/v8/yolov8{scale}.yaml",
    #     "name_prefix": "YOLOV8",
    #     "weights": "weights/yolov8{scale}.pt",
    # },
    # "yolov9": {
    #     "scales": ("s", "m"),
    #     "config": "ultralytics/cfg/models/v9/yolov9{scale}.yaml",
    #     "name_prefix": "YOLOV9",
    #     "weights": "weights/yolov9{scale}.pt",
    # },
    # "yolov10": {
    #     "scales": ("n", "s", "m"),
    #     "config": "ultralytics/cfg/models/v10/yolov10{scale}.yaml",
    #     "name_prefix": "YOLOV10",
    #     "weights": "weights/yolov10{scale}.pt",
    # },
    "yolo11": {
        "scales": ("n", "s", "m", "l"),
        "config": "ultralytics/cfg/models/11/yolo11{scale}.yaml",
        "name_prefix": "YOLO11",
        "weights": "weights/yolo11{scale}.pt",
    },
    # "yolo12": {
    #     "scales": ("n", "s"),
    #     "config": "ultralytics/cfg/models/12/yolo12{scale}.yaml",
    #     "name_prefix": "YOLO12",
    #     "weights": "weights/yolo12{scale}.pt",
    # },
}


def ensure_scale_config(base_config: str, scale_config: str) -> str:
    """Create a scale-specific config file from the base config if needed."""
    base_path = os.path.join(ROOT, base_config)
    scale_path = os.path.join(ROOT, scale_config)
    if not os.path.exists(scale_path):
        shutil.copyfile(base_path, scale_path)
        LOGGER.info("Create scale config | source=%s | target=%s", base_config, scale_config)
    return scale_config


def resolve_config_path(experiment: dict[str, Any], scale: str) -> str:
    """Resolve the model config path for the current experiment."""
    config_path = experiment["config"].format(scale=scale)
    if "base_config" in experiment:
        return ensure_scale_config(experiment["base_config"], config_path)
    return config_path


def resolve_experiment_weight_path(experiment: dict[str, Any], scale: str) -> str:
    """Resolve the pretrained weight path for the current experiment."""
    return os.path.join(ROOT, experiment["weights"].format(scale=scale))


def build_train_args(experiment_key: str, scale: str, use_pretrained: bool) -> dict[str, Any]:
    """Build training arguments for one experiment and scale."""
    mode = "pretrained" if use_pretrained else "scratch"
    return {
        **BASE_TRAIN_ARGS,
        **SCALE_ARGS[scale],
        "name": f"{EXPERIMENTS[experiment_key]['name_prefix']}-{scale}-{mode}",
    }


def cleanup() -> None:
    """Release resources between sequential training runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(0.5)


def train_experiment(experiment_key: str, scale: str, use_pretrained: bool = True):
    """Train one experiment on one scale."""
    experiment = EXPERIMENTS[experiment_key]
    config_path = resolve_config_path(experiment, scale)
    weight_path = resolve_experiment_weight_path(experiment, scale)
    effective_pretrained = use_pretrained and os.path.exists(weight_path)
    train_args = build_train_args(experiment_key, scale, effective_pretrained)

    LOGGER.info("Start training | experiment=%s | scale=%s | pretrained=%s", experiment_key, scale, effective_pretrained)
    LOGGER.info("Model config | path=%s", config_path)

    model = None
    try:
        model = YOLO(config_path)
        if effective_pretrained:
            LOGGER.info("Load weights | path=%s", weight_path)
            model.load(weight_path)
        elif use_pretrained:
            LOGGER.warning("Pretrained weights not found, train from scratch | path=%s", weight_path)
        result = model.train(**train_args)
        LOGGER.info("Training finished | name=%s", train_args["name"])
        return result
    except Exception:
        LOGGER.exception("Training failed | experiment=%s | scale=%s", experiment_key, scale)
        return None
    finally:
        del model
        cleanup()


def run_ablation_study(scales: tuple[str, ...] = SUPPORTED_SCALES, use_pretrained: bool = True) -> dict[str, dict[str, Any]]:
    """Run all configured ablation experiments."""
    LOGGER.info("Ablation study started | experiments=%s | scales=%s | pretrained=%s", list(EXPERIMENTS), scales, use_pretrained)
    results: dict[str, dict[str, Any]] = {}

    for experiment_key, experiment in EXPERIMENTS.items():
        results[experiment_key] = {}
        experiment_scales = tuple(scale for scale in scales if scale in experiment["scales"])
        LOGGER.info("Experiment scales | experiment=%s | scales=%s", experiment_key, experiment_scales)
        for scale in experiment_scales:
            results[experiment_key][scale] = train_experiment(experiment_key, scale, use_pretrained)

    LOGGER.info("Ablation study finished")
    return results


def log_summary(results: dict[str, dict[str, Any]]) -> None:
    """Log the final training summary."""
    LOGGER.info("Training summary")
    for experiment_key, scale_results in results.items():
        for scale, result in scale_results.items():
            status = "SUCCESS" if result is not None else "FAILED"
            LOGGER.info("Summary | experiment=%s | scale=%s | status=%s", experiment_key, scale, status)


if __name__ == "__main__":
    summary = run_ablation_study(use_pretrained=True)
    log_summary(summary)
