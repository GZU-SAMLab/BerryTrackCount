"""
Training helper for FCM-YOLO models with beta ablation study.
Supports YOLO11 with different beta values (0.25, 0.5, 0.75, mixed).
"""

import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import Dict, Any, List
import warnings
import yaml
from pathlib import Path

warnings.filterwarnings("ignore")

from ultralytics import YOLO


# Supported scales
SUPPORTED_SCALES = ("s", "m")

# Beta values for ablation study
BETA_VALUES = [0.25, 0.5, 0.75]
BETA_MIXED = "mixed"  # 前两个0.75，后两个0.25

# Base YAML config file
BASE_CONFIG = "configs/detector/FCM/fcm-bt0.75-yolo11.yaml"

BASE_TRAIN_ARGS: Dict[str, Any] = {
    "data": "configs/data/mydata.yaml",
    "epochs": 300,
    "imgsz": 640,
    "device": 0,
    "project": "runs/newFCM",
    "optimizer": "AdamW",
    "seed": 42,
    "amp": True,
}


def get_beta_str(beta) -> str:
    """Convert beta value to string format for file names."""
    if beta == "mixed":
        return "beta0.75-0.25"
    return f"beta{beta}"


def _get_scale_train_args(scale: str) -> Dict[str, Any]:
    """Get training arguments specific to model scale."""
    scale_args = {
        "n": {
            "batch": 64,
            "workers": 14,
            "lr0": 0.001,
            "patience": 50,
        },
        "s": {
            "batch": 48,
            "workers": 12,
            "lr0": 0.001,
            "patience": 50,
        },
        "m": {
            "batch": 32,
            "workers": 10,
            "lr0": 0.0008,
            "patience": 80,
        },
        "l": {
            "batch": 24,
            "workers": 8,
            "lr0": 0.0006,
            "patience": 100,
        },
    }
    
    args = scale_args.get(scale, scale_args["s"]).copy()
    args["lrf"] = 0.01  # YOLO11 learning rate decay
    
    return args


def modify_fcm_config(base_config_path: str, beta, scale: str = "s") -> str:
    """
    Modify FCM blocks in YAML config file with specified beta values.
    
    Args:
        base_config_path: Path to base YAML config file
        beta: Beta value (float) or "mixed" for mixed beta (0.75 for first two, 0.25 for last two)
        scale: Model scale (n/s/m/l)
    
    Returns:
        Path to modified temporary config file
    """
    # Read base config
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Find and modify FCM blocks in backbone
    # Search for all FCM layers and modify them
    fcm_count = 0
    for i, layer in enumerate(config['backbone']):
        if len(layer) >= 4 and layer[2] == 'FCM':  # Check if it's an FCM layer
            args = layer[3].copy()  # [dim_out, beta] or [dim_out, beta, k1, k2, k3]
            
            # Set beta value
            if beta == "mixed":
                # First two FCM blocks use 0.75, last two use 0.25
                args[1] = 0.75 if fcm_count < 2 else 0.25
            else:
                args[1] = beta
            
            # Update the layer
            config['backbone'][i] = layer[:3] + [args]
            fcm_count += 1
    
    # Create temporary config file
    temp_dir = Path(ROOT) / "temp_configs"
    temp_dir.mkdir(exist_ok=True)
    
    beta_str = get_beta_str(beta)
    temp_config_name = f"fcm-{beta_str}-yolo11{scale}.yaml"
    temp_config_path = temp_dir / temp_config_name
    
    # Write modified config
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return str(temp_config_path)


def train_single_model(scale: str, beta, use_pretrained: bool = True):
    """
    Train a single model with specific beta value.
    
    Args:
        scale: Model scale (n/s/m/l)
        beta: Beta value (float) or "mixed"
        use_pretrained: Whether to use pretrained COCO weights
    """
    # Modify config
    temp_config_path = modify_fcm_config(BASE_CONFIG, beta, scale)
    
    # Build training arguments
    args = BASE_TRAIN_ARGS.copy()
    args.update(_get_scale_train_args(scale))
    
    # Build name
    beta_str = get_beta_str(beta)
    status = "pretrained" if use_pretrained else "scratch"
    args["name"] = f"FCM-YOLO11{scale}-{beta_str}-{status}"
    
    # Load model
    if use_pretrained:
        weight_path = f"/home/wh1234_/code/Counting/weights/yolo11{scale}.pt"
        print(f"Loading pretrained COCO weights from: {weight_path}")
        model = YOLO(temp_config_path)
        model.load(weight_path)
    else:
        model = YOLO(temp_config_path)
    
    print("=" * 80)
    print(f"Training {args['name']}")
    print(f"  Beta: {beta}")
    print(f"  Scale: {scale}")
    print(f"  Pretrained: {use_pretrained}")
    print("=" * 80)
    
    # Train
    result = model.train(**args)
    print(f"{args['name']} training complete")
    
    # Clean up temporary config file
    try:
        os.remove(temp_config_path)
    except:
        pass
    
    return result


def train_beta_ablation(scale: str, beta_list: List = None, use_pretrained: bool = True):
    """
    Train models for beta ablation study.
    
    Args:
        scale: Model scale (n/s/m/l)
        beta_list: List of beta values to train. If None, train all beta values.
        use_pretrained: Whether to use pretrained COCO weights
    """
    if scale not in SUPPORTED_SCALES:
        raise ValueError(f"Unsupported model size {scale}, choose from {SUPPORTED_SCALES}.")
    
    if beta_list is None:
        beta_list = BETA_VALUES + [BETA_MIXED]
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"Beta Ablation Study - Scale: {scale}")
    print(f"Beta values: {beta_list}")
    print(f"{'='*80}\n")
    
    for beta in beta_list:
        print(f"\n{'='*80}")
        print(f"Training with beta={beta}")
        print(f"{'='*80}\n")
        
        result = train_single_model(scale, beta, use_pretrained)
        results[beta] = result
    
    return results


def train_all_scales_beta_ablation(beta_list: List = None, use_pretrained: bool = True):
    """
    Train all scales with beta ablation study.
    
    Args:
        beta_list: List of beta values to train. If None, train all beta values.
        use_pretrained: Whether to use pretrained COCO weights
    """
    summary = {}
    for scale in SUPPORTED_SCALES:
        try:
            print(f"\n{'#'*80}")
            print(f"# Training scale: {scale.upper()}")
            print(f"{'#'*80}\n")
            summary[scale] = train_beta_ablation(scale, beta_list, use_pretrained)
        except Exception as exc:
            print(f"Training {scale} failed: {exc}")
            import traceback
            traceback.print_exc()
            summary[scale] = None
    return summary


if __name__ == "__main__":
    print("=" * 80)
    print("Replacement of C3k2 in YOLO11 Backbone with FCM Module - Beta Ablation")
    print("=" * 80)
    
    # # Configuration
    # scale = "s"  # Change to "m" for medium scale
    # use_pretrained = True
    
    # # Beta values to train (None means all: 0.25, 0.5, 0.75, mixed)
    # beta_list = None  # Or specify: [0.25, 0.5, 0.75, "mixed"]
    
    # # Train single scale
    # print(f"\nTraining scale: {scale}")
    # results = train_beta_ablation(scale, beta_list, use_pretrained)
    
    # # Uncomment to train all scales
    # # print("\nTraining all scales")
    # # summary = train_all_scales_beta_ablation(beta_list, use_pretrained)
    
    # print("\n" + "=" * 80)
    # print("All experiments completed!")
    # print("=" * 80)

    cfg = {
        "11": {
            "config": f"configs/detector/FCM/fcm-bt0.75-yolo11s.yaml",
            "weights": f"/home/wh1234_/code/Counting/weights/yolo11s.pt",
            "train_args": {
                "data": "configs/data/mydata.yaml",
                "epochs": 600,
                "imgsz": 640,
                "device": 0,
                "project": "runs/newFCM",
                "name": "FCM-YOLO11s-beta0.25-0.75-pretrained",
                "optimizer": "AdamW",
                "seed": 42,
                "amp": True,
                "batch": 48,
                "workers": 12,
                "lr0": 0.001,
                "patience": 50,
                "lrf": 0.01,
            }
        }
    }
    use_pretrained = True
    if use_pretrained:
        weight_path = cfg["11"]["weights"]
        print(f"Loading pretrained COCO weights from: {weight_path}")
        model = YOLO(cfg["11"]["config"])
        model.load(cfg["11"]["weights"])
    else:
        model = YOLO(cfg["11"]["config"])
    
    print("=" * 60)
    status = "with pretrained COCO weights" if use_pretrained else "from scratch"
    print(f"Training {cfg['11']['train_args']['name']} {status}")
    print("=" * 60)
    result = model.train(**cfg["11"]["train_args"])

