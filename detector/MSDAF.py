"""
Training helper for MSDAF-YOLO models with beta ablation study.
Supports YOLO8/YOLO11 across n/s/m/l scales with standard beta values and
the YOLO11-specific mixed beta setting 0.75-0.25.
"""

import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from typing import Dict, Any, List, Union, Tuple
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO


# SUPPORTED_SCALES = ("s", "m")
SUPPORTED_SCALES = ("n", "s", "m", "l")

BetaValue = Union[float, str]

# Beta values for ablation study
STANDARD_BETA_VALUES: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
SPECIAL_BETA_VALUES: Tuple[str, ...] = ("0.75-0.25",)
BETA_VALUES: Tuple[BetaValue, ...] = STANDARD_BETA_VALUES + SPECIAL_BETA_VALUES
VERSION_BETA_VALUES: Dict[str, Tuple[BetaValue, ...]] = {
    "v8": STANDARD_BETA_VALUES,
    "11": BETA_VALUES,
}

allowed_versions = ("v8", "11", "both")

BASE_TRAIN_ARGS: Dict[str, Any] = {
    "data": "configs/data/mydata.yaml",
    "epochs": 300,
    "imgsz": 640,
    "device": 0,
    "project": "runs/MSDAF",
    "optimizer": "AdamW",
    "seed": 42,
    "amp": True,
}


def get_beta_str(beta: BetaValue) -> str:
    """Convert beta value to string format for file names (e.g., 0.1 -> 'beta0.1')."""
    beta_text = str(beta)
    return beta_text if beta_text.startswith("beta") else f"beta{beta_text}"


def is_beta_supported(version_key: str, beta: BetaValue) -> bool:
    """Check whether a beta value is available for the given model version."""
    return beta in VERSION_BETA_VALUES[version_key]


def generate_msdaf_configs():
    """
    Generate MSDAF configurations for all combinations of version, scale, and beta.
    Structure: CONFIGS[version][scale][beta] -> config_dict
    
    Note: Config files don't have scale suffix (e.g., msdaf-beta0.5-yolo11.yaml)
    The scale is specified through the 'model' parameter when loading weights.
    """
    configs = {
        "v8": {},
        "11": {}
    }
    
    # YOLOv8 configurations
    for scale in SUPPORTED_SCALES:
        configs["v8"][scale] = {}
        for beta in VERSION_BETA_VALUES["v8"]:
            beta_str = get_beta_str(beta)
            configs["v8"][scale][beta] = {
                "config": f"configs/detector/FCM/msdaf-{beta_str}-yolov8{scale}.yaml",
                "weights": f"/home/wh1234_/code/Counting/weights/yolov8{scale}.pt",
                "name": f"MSDAF-YOLOv8{scale}-{beta_str}",
                "train_args": _get_scale_train_args(scale, version="v8")
            }
    
    # YOLO11 configurations
    for scale in SUPPORTED_SCALES:
        configs["11"][scale] = {}
        for beta in VERSION_BETA_VALUES["11"]:
            beta_str = get_beta_str(beta)
            configs["11"][scale][beta] = {
                "config": f"configs/detector/FCM/msdaf-{beta_str}-yolo11{scale}.yaml",
                "weights": f"/home/wh1234_/code/Counting/weights/yolo11{scale}.pt",
                "name": f"MSDAF-YOLO11{scale}-{beta_str}",
                "train_args": _get_scale_train_args(scale, version="11")
            }
    
    return configs


def _get_scale_train_args(scale: str, version: str) -> Dict[str, Any]:
    """Get training arguments specific to model scale."""
    # Scale-specific hyperparameters
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
    
    # Version-specific learning rate decay
    if version == "v8":
        args["lrf"] = 0.1
    else:  # YOLO11
        args["lrf"] = 0.01
    
    return args


# Generate configurations
CONFIGS = generate_msdaf_configs()


def get_pretrained_weight_path(version_key: str, model_size: str, beta: BetaValue) -> str:
    """Get the path to pretrained COCO weights."""
    cfg = CONFIGS[version_key][model_size][beta]
    return cfg["weights"]


def build_train_args(version_key: str, model_size: str, beta: BetaValue, use_pretrained: bool = False) -> Dict[str, Any]:
    """Merge base arguments with model- and scale-specific overrides."""
    cfg = CONFIGS[version_key][model_size][beta]
    args = BASE_TRAIN_ARGS.copy()
    args.update(cfg["train_args"])
    
    # Modify name to indicate pretrained status
    if use_pretrained:
        args["name"] = f"{cfg['name']}-pretrained"
    else:
        args["name"] = f"{cfg['name']}-scratch"
    
    return args


def train_single_model(version_key: str, model_size: str, beta: BetaValue, use_pretrained: bool = False):
    """Train a single model with specific beta value, with or without pretrained weights."""
    cfg = CONFIGS[version_key][model_size][beta]
    args = build_train_args(version_key, model_size, beta, use_pretrained)
    
    # Load model with or without pretrained weights
    if use_pretrained:
        weight_path = get_pretrained_weight_path(version_key, model_size, beta)
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


def train_beta_ablation(model_size: str, model_version: str = "both", 
                       beta_list: List[BetaValue] = None, load_pretrained: int = 0):
    """
    Train models for beta ablation study.
    
    Args:
        model_size: Model size (n/s/m/l)
        model_version: Model version ("v8", "11", or "both")
        beta_list: List of beta values to train. If None, train all beta values.
        load_pretrained: 0 = no pretrained, 1 = with pretrained, 2 = both
    """
    if model_size not in SUPPORTED_SCALES:
        raise ValueError(f"Unsupported model size {model_size}, choose from {SUPPORTED_SCALES}.")

    if model_version not in allowed_versions:
        raise ValueError(f"model_version must be in {allowed_versions}")
    
    if load_pretrained not in (0, 1, 2):
        raise ValueError(f"load_pretrained must be 0, 1, or 2, got {load_pretrained}")
    
    # Default to all beta values if not specified
    if beta_list is None:
        if model_version == "v8":
            beta_list = list(VERSION_BETA_VALUES["v8"])
        else:
            beta_list = list(VERSION_BETA_VALUES["11"])
    
    # Validate beta values
    for beta in beta_list:
        if model_version == "both":
            if not (is_beta_supported("v8", beta) or is_beta_supported("11", beta)):
                raise ValueError(f"Beta value {beta} not supported. Choose from {BETA_VALUES}")
        elif not is_beta_supported(model_version, beta):
            supported_betas = VERSION_BETA_VALUES[model_version]
            raise ValueError(f"Beta value {beta} not supported for {model_version}. Choose from {supported_betas}")

    results = {}
    
    # Determine training modes based on load_pretrained flag
    if load_pretrained == 2:
        train_modes = [False, True]  # Train both from scratch and with pretrained
    elif load_pretrained == 1:
        train_modes = [True]  # Only with pretrained
    else:  # load_pretrained == 0
        train_modes = [False]  # Only from scratch
    
    # Train for each beta value
    for beta in beta_list:
        beta_str = get_beta_str(beta)
        print(f"\n{'='*80}")
        print(f"Starting ablation study for beta={beta} ({beta_str})")
        print(f"{'='*80}\n")
        
        if beta_str not in results:
            results[beta_str] = {}
        
        for use_pretrained in train_modes:
            if model_version in ("v8", "both") and is_beta_supported("v8", beta):
                result_key = "v8" + ("_pretrained" if use_pretrained else "_scratch")
                if "v8" not in results[beta_str]:
                    results[beta_str]["v8"] = {}
                results[beta_str]["v8"][result_key] = train_single_model("v8", model_size, beta, use_pretrained)

            if model_version in ("11", "both") and is_beta_supported("11", beta):
                result_key = "11" + ("_pretrained" if use_pretrained else "_scratch")
                if "11" not in results[beta_str]:
                    results[beta_str]["11"] = {}
                results[beta_str]["11"][result_key] = train_single_model("11", model_size, beta, use_pretrained)
        
        print(f"\n{'='*80}")
        print(f"Completed ablation study for beta={beta} ({beta_str})")
        print(f"{'='*80}\n")
    
    return results


def train_all_betas_all_scales(model_version: str = "both", 
                                beta_list: List[BetaValue] = None, 
                                load_pretrained: int = 0):
    """
    Train all scales with all beta values (complete ablation study).
    
    Args:
        model_version: Model version ("v8", "11", or "both")
        beta_list: List of beta values to train. If None, train all beta values.
        load_pretrained: 0 = no pretrained, 1 = with pretrained, 2 = both
    """
    summary = {}
    for size in SUPPORTED_SCALES:
        try:
            print(f"\n{'#'*80}")
            print(f"# Training scale: {size.upper()}")
            print(f"{'#'*80}\n")
            summary[size] = train_beta_ablation(
                size, 
                model_version=model_version, 
                beta_list=beta_list, 
                load_pretrained=load_pretrained
            )
        except Exception as exc:
            print(f"Training {size} failed: {exc}")
            import traceback
            traceback.print_exc()
            summary[size] = None
    return summary


def train_specific_beta(beta: BetaValue, model_size: str = None, 
                       model_version: str = "both", load_pretrained: int = 0):
    """
    Train specific beta value across all scales or a specific scale.
    
    Args:
        beta: Beta value (0.1, 0.25, 0.5, 0.75, 0.9, or 0.75-0.25 for YOLO11)
        model_size: Model size (n/s/m/l). If None, train all scales.
        model_version: Model version ("v8", "11", or "both")
        load_pretrained: 0 = no pretrained, 1 = with pretrained, 2 = both
    """
    if model_version == "both":
        if not (is_beta_supported("v8", beta) or is_beta_supported("11", beta)):
            raise ValueError(f"Beta value {beta} not supported. Choose from {BETA_VALUES}")
    elif not is_beta_supported(model_version, beta):
        supported_betas = VERSION_BETA_VALUES[model_version]
        raise ValueError(f"Beta value {beta} not supported for {model_version}. Choose from {supported_betas}")
    
    if model_size is not None:
        # Train specific scale
        return train_beta_ablation(model_size, model_version, [beta], load_pretrained)
    else:
        # Train all scales
        return train_all_betas_all_scales(model_version, [beta], load_pretrained)


if __name__ == "__main__":
    print("=" * 80)
    print("MSDAF module replaces C3k2 modules in the YOLO11 backbone - beta ablation study (first two 0.75, last two 0.25)")
    print("=" * 80)

    beta = "0.75-0.25"
    model_version = "11"
    load_pretrained = 1
    summary = train_specific_beta(beta, None, model_version, load_pretrained)

    # print("=" * 80)
    # print("MSDAF module replaces C3k2 modules in the YOLO11 backbone - beta ablation study")
    # print("=" * 80)
    
    # print("\nTraining mode:")
    # print("1. Train a specific beta value (specific scale)")
    # print("2. Train a specific beta value (all scales)")
    # print("3. Train all beta values (specific scale)")
    # print("4. Complete ablation study (all beta values + all scales) [default]")
    # print("=" * 80)
    # # mode_choice = input("Select mode (1/2/3/4, default 4): ").strip()
    # mode_choice = "4"
    
    # print("\nModel version:")
    # print("1. YOLOv8")
    # print("2. YOLO11")
    # print("3. Train both (default)")
    # # version_choice = input("Select version (1/2/3, default 3): ").strip()
    # version_choice = "2"
    # version_map = {"1": "v8", "2": "11", "3": "both"}
    # model_version = version_map.get(version_choice, "both")
    # print(f"Training model version: {model_version}")

    # print("\nPretrained weight loading:")
    # print("0. Train from scratch (do not use pretrained weights)")
    # print("1. Use pretrained COCO weights")
    # print("2. Train both modes (from scratch + pretrained)")
    # # pretrained_choice = input("Select pretrained loading mode (0/1/2, default 0): ").strip()
    # pretrained_choice = "2"
    # pretrained_map = {"0": 0, "1": 1, "2": 2}
    # load_pretrained = pretrained_map.get(pretrained_choice, 0)
    
    # pretrained_desc = {
    #     0: "Train from scratch",
    #     1: "Use pretrained COCO weights",
    #     2: "Train both modes (from scratch + pretrained)"
    # }
    # print(f"Pretrained loading mode: {load_pretrained} - {pretrained_desc[load_pretrained]}")
    
    # # Execute based on mode
    # if mode_choice == "1":
    #     # Train specific beta + specific scale
    #     print("\nBeta value selection:")
    #     for i, beta in enumerate(BETA_VALUES, 1):
    #         print(f"{i}. beta={beta}")
    #     # beta_choice = input(f"Select beta value (1-{len(BETA_VALUES)}): ").strip()
    #     beta_choice = "3"  # Default to beta=0.5
    #     beta = BETA_VALUES[int(beta_choice) - 1] if beta_choice.isdigit() else 0.5
        
    #     # scale = input("Enter model scale (n/s/m/l): ").strip().lower()
    #     scale = "s"
    #     train_specific_beta(beta, scale, model_version, load_pretrained)
        
    # elif mode_choice == "2":
    #     # Train specific beta + all scales
    #     print("\nBeta value selection:")
    #     for i, beta in enumerate(BETA_VALUES, 1):
    #         print(f"{i}. beta={beta}")
    #     # beta_choice = input(f"Select beta value (1-{len(BETA_VALUES)}): ").strip()
    #     beta_choice = "3"  # Default to beta=0.5
    #     beta = BETA_VALUES[int(beta_choice) - 1] if beta_choice.isdigit() else 0.5
        
    #     train_specific_beta(beta, None, model_version, load_pretrained)
        
    # elif mode_choice == "3":
    #     # Train all betas + specific scale
    #     # scale = input("Enter model scale (n/s/m/l): ").strip().lower()
    #     scale = "s"
    #     train_beta_ablation(scale, model_version, None, load_pretrained)
        
    # else:
    #     # Full ablation study - all betas + all scales
    #     print("\n" + "=" * 80)
    #     print("Starting complete ablation study:")
    #     print(f"  - Model version: {model_version}")
    #     print(f"  - Model scales: {', '.join(SUPPORTED_SCALES)}")
    #     print(f"  - Beta values: {', '.join([str(b) for b in BETA_VALUES])}")
    #     print(f"  - Pretrained mode: {pretrained_desc[load_pretrained]}")
    #     print("=" * 80 + "\n")
        
    #     summary = train_all_betas_all_scales(model_version, None, load_pretrained)
        
    #     print("\n" + "=" * 80)
    #     print("Complete ablation study finished!")
    #     print("=" * 80)
    #     print("\nTraining summary:")
    #     for scale, result in summary.items():
    #         if result is not None:
    #             print(f"  ✓ {scale.upper()}: success")
    #         else:
    #             print(f"  ✗ {scale.upper()}: failed")
    #     print("=" * 80)

