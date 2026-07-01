#!/usr/bin/env python3
# Purpose: generate detector attention heatmaps and detection visualizations for dataset images.

from __future__ import annotations

import argparse
import importlib.util
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GRAD_CAM_ROOT = REPO_ROOT / "pytorch-grad-cam"
if GRAD_CAM_ROOT.exists() and str(GRAD_CAM_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAD_CAM_ROOT))

LOGGER = logging.getLogger("heapmap_visu")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_IMAGE_DIR = REPO_ROOT / "dataset" / "images_640"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "visualize" / "heapmap-detectors"
DINO_OPS_ROOT = REPO_ROOT / "detector" / "DINO" / "models" / "dino" / "ops"
DETECTOR_SPECS = {
    # "yolo11n": {"backend": "yolo", "weights": REPO_ROOT / "weights" / "yolo11n.pt"},
    # "berrydet": {"backend": "yolo", "weights": REPO_ROOT / "weights" / "berrydet_s.pt"},
    # "rtdetr": {"backend": "rtdetr", "weights": REPO_ROOT / "weights" / "rtdetr-l.pt"},
    # "yolox": {
    #     "backend": "yolox",
    #     "weights": REPO_ROOT / "weights" / "yolox-s.pth",
    #     "config": REPO_ROOT / "configs" / "detector" / "yolox" / "yolox_s_exp.py",
    # },
    "dino": {
        "backend": "dino",
        "weights": REPO_ROOT / "weights" / "dino.pth",
        "config": REPO_ROOT / "detector" / "DINO" / "config" / "DINO" / "DINO_4scale_custom.py",
    },
}
STAGE_NAMES = {
    0: "Flower",
    1: "Green",
    2: "Light Purple",
    3: "Blue",
}
STAGE_COLORS_BGR = {
    0: (0, 159, 230),
    1: (115, 158, 0),
    2: (167, 121, 204),
    3: (178, 114, 0),
}
RESAMPLE_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


@dataclass
class DetectionResult:
    boxes: torch.Tensor
    cls: torch.Tensor
    conf: torch.Tensor


@dataclass
class DetectorBundle:
    name: str
    backend: str
    detector: Any
    cam_model: Any
    target_layers: list[torch.nn.Module]
    preprocess: Callable[[Image.Image], torch.Tensor]
    predict: Callable[[Path, Image.Image], DetectionResult]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize detector attention maps on random images.")
    parser.add_argument("--image", type=Path, default=None, help="Single image path to visualize instead of random sampling.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Image root for random sampling.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for saved images.")
    parser.add_argument("--num-images", type=int, default=61, help="Number of images sampled for visualization.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square inference size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--seed", type=int, default=55, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--method",
        choices=("gradcam", "eigengradcam", "eigencam"),
        default="eigencam",
        help="CAM algorithm used for heatmap generation.",
    )
    parser.add_argument("--box-renorm", dest="box_renorm", action="store_true", help="Renormalize CAM inside predicted boxes.")
    parser.add_argument("--no-box-renorm", dest="box_renorm", action="store_false", help="Disable CAM box renormalization.")
    parser.set_defaults(box_renorm=False)
    parser.add_argument("--heatmap-alpha", type=float, default=0.58, help="Heatmap overlay opacity.")
    parser.add_argument("--heatmap-gamma", type=float, default=0.8, help="Gamma correction for smoother hotspots.")
    parser.add_argument("--smooth-radius", type=float, default=8.0, help="Gaussian blur radius for CAM smoothing.")
    parser.add_argument("--draw-scale", type=int, default=3, help="Scale factor for anti-aliased box rendering.")
    parser.add_argument("--box-width", type=int, default=2, help="Box line width at final image scale.")
    parser.add_argument("--label-font-size", type=int, default=13, help="Label font size at final image scale.")
    parser.add_argument("--eigen-smooth", dest="eigen_smooth", action="store_true", help="Use pytorch-grad-cam eigen smoothing.")
    parser.add_argument("--no-eigen-smooth", dest="eigen_smooth", action="store_false", help="Disable pytorch-grad-cam eigen smoothing.")
    parser.set_defaults(eigen_smooth=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Torch device.")
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if args.image is not None and not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if args.image is not None and args.image.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"Unsupported image suffix: {args.image}")
    for name, spec in DETECTOR_SPECS.items():
        path = spec["weights"]
        if not path.exists():
            raise FileNotFoundError(f"{name} weights not found: {path}")
        config = spec.get("config")
        if config is not None and not config.exists():
            raise FileNotFoundError(f"{name} config not found: {config}")


def sample_images(image_dir: Path, count: int, seed: int) -> list[Path]:
    images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No images found under: {image_dir}")
    rng = random.Random(seed)
    return rng.sample(images, min(count, len(images)))


def resolve_image_paths(args: argparse.Namespace) -> list[Path]:
    if args.image is not None:
        image_path = args.image if args.image.is_absolute() else (REPO_ROOT / args.image)
        return [image_path.resolve()]
    return sample_images(args.image_dir, args.num_images, args.seed)


def output_stem(index: int, image_path: Path, single_image: bool) -> str:
    return image_path.stem if single_image else f"{index:02d}_{image_path.stem}"


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def label_path_for_image(image_path: Path, image_root: Path) -> Path:
    try:
        relative = image_path.relative_to(image_root)
    except ValueError:
        relative = image_path
    parts = list(relative.parts)
    if parts and parts[0] == "images":
        parts[0] = "labels"
    else:
        parts = ["labels", *parts]
    return image_root / Path(*parts).with_suffix(".txt")


def load_yolo_labels(label_path: Path, image_size: tuple[int, int]) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not label_path.exists():
        LOGGER.warning("GT label file not found: %s", label_path)
        return []
    width, height = image_size
    labels = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if len(parts) < 5:
                LOGGER.warning("Skip malformed GT row: %s:%d", label_path, line_no)
                continue
            cls, xc, yc, box_w, box_h = int(float(parts[0])), *map(float, parts[1:5])
            x1 = max(0.0, (xc - box_w / 2.0) * width)
            y1 = max(0.0, (yc - box_h / 2.0) * height)
            x2 = min(float(width - 1), (xc + box_w / 2.0) * width)
            y2 = min(float(height - 1), (yc + box_h / 2.0) * height)
            labels.append((cls, (x1, y1, x2, y2)))
    return labels


def draw_gt(
    image: Image.Image,
    labels: list[tuple[int, tuple[float, float, float, float]]],
    draw_scale: int,
    box_width: int,
    font_size: int,
) -> Image.Image:
    items = [
        (box, STAGE_NAMES.get(cls, str(cls)), bgr_to_rgb(STAGE_COLORS_BGR.get(cls, (255, 220, 32))))
        for cls, box in labels
    ]
    return render_boxes(image, items, draw_scale, box_width, font_size)


def save_gt_image(
    image: Image.Image,
    image_path: Path,
    image_root: Path,
    output_path: Path,
    draw_scale: int,
    box_width: int,
    font_size: int,
) -> None:
    label_path = label_path_for_image(image_path, image_root)
    labels = load_yolo_labels(label_path, image.size)
    draw_gt(image, labels, draw_scale, box_width, font_size).save(output_path)
    LOGGER.info("Saved GT image: %s", output_path)


def bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    blue, green, red = color
    return red, green, blue


def load_draw_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def scaled_box(box: tuple[float, float, float, float], scale: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return tuple(round(value * scale) for value in (x1, y1, x2, y2))


def draw_scaled_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    scale: int,
) -> None:
    x, y = xy
    pad_x, pad_y = 4 * scale, 3 * scale
    text_bbox = draw.textbbox((x, y), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    label_y = max(0, y - text_h - 2 * pad_y)
    draw.rounded_rectangle(
        (x, label_y, x + text_w + 2 * pad_x, label_y + text_h + 2 * pad_y),
        radius=2 * scale,
        fill=color,
    )
    draw.text((x + pad_x, label_y + pad_y), text, fill=(0, 0, 0, 255), font=font)


def render_boxes(
    image: Image.Image,
    items: list[tuple[tuple[float, float, float, float], str, tuple[int, int, int]]],
    draw_scale: int,
    box_width: int,
    font_size: int,
) -> Image.Image:
    scale = max(1, int(draw_scale))
    overlay_size = (image.width * scale, image.height * scale)
    overlay = Image.new("RGBA", overlay_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_draw_font(max(1, font_size * scale))
    line_width = max(1, box_width * scale)

    for box, text, color_rgb in items:
        color = (*color_rgb, 255)
        scaled = scaled_box(box, scale)
        draw.rectangle(scaled, outline=color, width=line_width)
        draw_scaled_label(draw, (scaled[0], scaled[1]), text, color, font, scale)

    overlay = overlay.resize(image.size, RESAMPLE_LANCZOS)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def image_to_tensor(image: Image.Image, imgsz: int, device: torch.device) -> torch.Tensor:
    resized = image.resize((imgsz, imgsz), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def jet_colormap(cam: np.ndarray) -> np.ndarray:
    x = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def enhance_cam(cam: np.ndarray, gamma: float) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    low, high = np.percentile(cam, (1.0, 99.5))
    cam = np.clip((cam - low) / (high - low + 1e-7), 0.0, 1.0)
    return np.power(cam, gamma).astype(np.float32)


def smooth_cam(cam: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return scale_cam(cam)
    cam_image = Image.fromarray(np.uint8(scale_cam(cam) * 255))
    cam_image = cam_image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(cam_image, dtype=np.float32) / 255.0


def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.58, gamma: float = 0.65) -> Image.Image:
    cam = enhance_cam(cam, gamma)
    cam_image = Image.fromarray(np.uint8(jet_colormap(cam) * 255)).resize(image.size, Image.BILINEAR)
    base = np.asarray(image, dtype=np.float32) / 255.0
    heat = np.asarray(cam_image, dtype=np.float32) / 255.0
    mixed = np.clip(base * (1.0 - alpha) + heat * alpha, 0.0, 1.0)
    return Image.fromarray(np.uint8(mixed * 255))


def scale_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    cam = cam - float(cam.min())
    return cam / (float(cam.max()) + 1e-7)


def eigen_project_activations(activation_batch: np.ndarray) -> np.ndarray:
    projections = []
    for activations in activation_batch:
        reshaped = activations.reshape(activations.shape[0], -1).transpose()
        reshaped = reshaped - reshaped.mean(axis=0)
        _, _, vt = np.linalg.svd(reshaped, full_matrices=False)
        projection = reshaped @ vt[0, :]
        projection = projection.reshape(activations.shape[1:])
        if abs(float(projection.min())) > abs(float(projection.max())):
            projection = -projection
        projections.append(scale_cam(projection))
    return np.asarray(projections, dtype=np.float32)


def resize_cam_batch(cams: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    resized = []
    for cam in cams:
        image = Image.fromarray(np.uint8(scale_cam(cam) * 255))
        image = image.resize((target_w, target_h), RESAMPLE_BICUBIC)
        resized.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.asarray(resized, dtype=np.float32)


class YoloCamModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(False), torch.enable_grad():
            output = self.model(x)
        return self._differentiable_output(output)

    @staticmethod
    def _flatten_raw_head(raw: Any) -> torch.Tensor | None:
        if isinstance(raw, dict):
            raw = raw.get("one2many", raw.get("one2one"))
        if isinstance(raw, (list, tuple)) and raw and all(isinstance(item, torch.Tensor) for item in raw):
            return torch.cat([item.view(item.shape[0], item.shape[1], -1) for item in raw], dim=2)
        return raw if isinstance(raw, torch.Tensor) else None

    def _differentiable_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            pred = output[0]
            if isinstance(pred, torch.Tensor) and pred.requires_grad:
                return pred
            raw = self._flatten_raw_head(output[1] if len(output) > 1 else None)
            if raw is not None:
                return raw
            return pred
        raw = self._flatten_raw_head(output)
        if raw is None:
            raise RuntimeError("YOLO CAM wrapper could not find a tensor output.")
        return raw


class TensorScoreTarget:
    def __call__(self, model_output: Any) -> torch.Tensor:
        if isinstance(model_output, dict):
            logits = model_output.get("pred_logits")
            if isinstance(logits, torch.Tensor):
                return logits.sigmoid().amax()
            boxes = model_output.get("pred_boxes")
            if isinstance(boxes, torch.Tensor):
                return boxes.flatten().max()
            raise RuntimeError("Cannot find differentiable DINO output for CAM target.")
        if model_output.ndim == 3:
            model_output = model_output[0]
        if model_output.ndim == 2 and model_output.shape[0] > 4:
            if model_output.shape[0] > 64:
                return model_output[64:, :].sigmoid().amax()
            return model_output[4:, :].amax()
        if model_output.ndim == 2 and model_output.shape[1] > 4:
            if model_output.shape[1] > 64:
                return model_output[:, 64:].sigmoid().amax()
            return model_output[:, 4:].amax()
        return model_output.flatten().max()


class DINOInputAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(False), torch.enable_grad():
            return self.model(x)["pred_logits"].sigmoid()


class DINOBackboneEigenCAM:
    def __init__(self, model: torch.nn.Module, nested_tensor_factory: Callable[[list[torch.Tensor]], Any]) -> None:
        self.model = model
        self.nested_tensor_factory = nested_tensor_factory

    def __call__(self, input_tensor: torch.Tensor, targets: Any = None, eigen_smooth: bool = True) -> np.ndarray:
        del targets, eigen_smooth
        with torch.no_grad():
            nested = self.nested_tensor_factory(input_tensor)
            features, _ = self.model.backbone(nested)
            activations = features[-1].tensors.detach().cpu().numpy()
        cams = eigen_project_activations(activations)
        return resize_cam_batch(cams, input_tensor.shape[-2:])


def load_cam_class(method: str) -> Any:
    try:
        from pytorch_grad_cam import EigenCAM, EigenGradCAM, GradCAM
    except Exception as exc:
        raise RuntimeError(f"Failed to import official pytorch-grad-cam from {GRAD_CAM_ROOT}.") from exc
    return {"gradcam": GradCAM, "eigengradcam": EigenGradCAM, "eigencam": EigenCAM}[method]


def build_cam_extractors(
    cam_models: dict[str, Any],
    target_layers: dict[str, list[torch.nn.Module]],
    method: str,
    backends: dict[str, str],
) -> dict[str, Any]:
    needs_official_cam = any(backends.get(name) != "dino" for name in cam_models)
    cam_cls = load_cam_class(method) if needs_official_cam else None
    extractors = {}
    for name, model in cam_models.items():
        if backends.get(name) == "dino":
            if method != "eigencam":
                LOGGER.warning("DINO uses backbone EigenCAM fallback instead of %s to avoid native GradCAM crashes.", method)
            extractors[name] = model
            continue
        if cam_cls is None:
            raise RuntimeError("Official pytorch-grad-cam class was not initialized.")
        extractors[name] = cam_cls(model=model, target_layers=target_layers[name])
    if method != "eigencam":
        for name, cam in extractors.items():
            if backends.get(name) != "dino":
                cam.compute_input_gradient = True
    return extractors


def release_cam_extractors(extractors: dict[str, Any]) -> None:
    for cam in extractors.values():
        if hasattr(cam, "activations_and_grads"):
            cam.activations_and_grads.release()


def generate_cam(cam: Any, tensor: torch.Tensor, method: str, eigen_smooth: bool) -> np.ndarray:
    targets = None if method == "eigencam" else [TensorScoreTarget()]
    return cam(input_tensor=tensor, targets=targets, eigen_smooth=eigen_smooth)[0]


def ultralytics_to_detection(result: Any, device: torch.device) -> DetectionResult:
    if result.boxes is None or len(result.boxes) == 0:
        empty = torch.empty((0,), device=device)
        return DetectionResult(torch.empty((0, 4), device=device), empty.long(), empty)
    return DetectionResult(
        boxes=result.boxes.xyxy.detach().to(device),
        cls=result.boxes.cls.detach().to(device).long(),
        conf=result.boxes.conf.detach().to(device),
    )


def detection_boxes(result: DetectionResult, image_size: tuple[int, int], cam_shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    if result.boxes.numel() == 0:
        return []
    boxes = result.boxes.detach().cpu().numpy()
    image_w, image_h = image_size
    cam_h, cam_w = cam_shape
    scale_x = cam_w / max(image_w, 1)
    scale_y = cam_h / max(image_h, 1)
    normalized = []
    for x1, y1, x2, y2 in boxes:
        left = int(np.clip(round(x1 * scale_x), 0, cam_w - 1))
        top = int(np.clip(round(y1 * scale_y), 0, cam_h - 1))
        right = int(np.clip(round(x2 * scale_x), left + 1, cam_w))
        bottom = int(np.clip(round(y2 * scale_y), top + 1, cam_h))
        normalized.append((left, top, right, bottom))
    return normalized


def renormalize_cam_in_boxes(cam: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    if not boxes:
        return scale_cam(cam)
    renormalized = np.zeros(cam.shape, dtype=np.float32)
    for left, top, right, bottom in boxes:
        box_cam = cam[top:bottom, left:right]
        if box_cam.size:
            renormalized[top:bottom, left:right] = scale_cam(box_cam)
    return scale_cam(renormalized)


def get_target_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    layers = getattr(model, "model", None)
    if layers is None or len(layers) < 2:
        raise ValueError("Cannot find YOLO layer list for target layer selection.")
    detect = layers[-1]
    from_ids = getattr(detect, "f", None)
    if isinstance(from_ids, int):
        from_ids = [from_ids]
    if not from_ids:
        LOGGER.info("Using fallback target layer before detection head: %s", layers[-2].__class__.__name__)
        return [layers[-2]]

    target_ids = [idx for idx in from_ids if idx != -1]
    if not target_ids:
        LOGGER.info("Using fallback target layer before detection head: %s", layers[-2].__class__.__name__)
        return [layers[-2]]

    target_ids = target_ids[-2:] if len(target_ids) > 2 else target_ids
    LOGGER.info(
        "Using target layers feeding detection head: %s (%s)",
        target_ids,
        ", ".join(layers[idx].__class__.__name__ for idx in target_ids),
    )
    return [layers[idx] for idx in target_ids]


def save_detection(
    result: DetectionResult,
    image: Image.Image,
    output_path: Path,
    draw_scale: int,
    box_width: int,
    font_size: int,
) -> None:
    items = []
    if result.boxes.numel() > 0:
        boxes = result.boxes.detach().cpu().numpy()
        classes = result.cls.detach().cpu().numpy().astype(int)
        confs = result.conf.detach().cpu().numpy()
        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(float, box)
            color = bgr_to_rgb(STAGE_COLORS_BGR.get(cls, (255, 220, 32)))
            label = f"{STAGE_NAMES.get(cls, str(cls))} {conf:.2f}"
            items.append(((x1, y1, x2, y2), label, color))
    render_boxes(image, items, draw_scale, box_width, font_size).save(output_path)


def load_torch_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare_dino_ops_path() -> None:
    candidates = [DINO_OPS_ROOT, *DINO_OPS_ROOT.glob("build/lib.*"), *DINO_OPS_ROOT.glob("dist/*.egg")]
    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if importlib.util.find_spec("MultiScaleDeformableAttention") is None:
        compiled = sorted(DINO_OPS_ROOT.rglob("MultiScaleDeformableAttention*.so"))
        compiled += sorted(DINO_OPS_ROOT.rglob("MultiScaleDeformableAttention*.pyd"))
        found = ", ".join(str(path) for path in compiled) if compiled else "none"
        raise ModuleNotFoundError(
            "DINO op MultiScaleDeformableAttention is not importable. "
            f"Found compiled files: {found}. "
            f"Rebuild it in the active Python/CUDA environment: cd {DINO_OPS_ROOT} && python setup.py build install"
        )


def load_ultralytics_bundle(
    name: str,
    backend: str,
    weights: Path,
    imgsz: int,
    conf: float,
    device: torch.device,
    device_arg: str,
) -> DetectorBundle:
    try:
        from ultralytics import RTDETR, YOLO
    except Exception as exc:
        raise RuntimeError("Failed to import ultralytics. Check the local cv2/numpy installation.") from exc

    model_cls = RTDETR if backend == "rtdetr" else YOLO
    detector = model_cls(str(weights))
    detector.model.to(device).eval()
    for param in detector.model.parameters():
        param.requires_grad_(True)

    def preprocess(image: Image.Image) -> torch.Tensor:
        return image_to_tensor(image, imgsz, device)

    def predict(image_path: Path, _: Image.Image) -> DetectionResult:
        result = detector.predict(str(image_path), imgsz=imgsz, conf=conf, device=device_arg, verbose=False)[0]
        return ultralytics_to_detection(result, device)

    return DetectorBundle(
        name=name,
        backend=backend,
        detector=detector,
        cam_model=YoloCamModel(detector.model).to(device).eval(),
        target_layers=get_target_layers(detector.model),
        preprocess=preprocess,
        predict=predict,
    )


def load_yolox_bundle(
    name: str,
    weights: Path,
    config: Path,
    conf: float,
    device: torch.device,
) -> DetectorBundle:
    yolox_root = REPO_ROOT / "detector" / "YOLOX"
    if str(yolox_root) not in sys.path:
        sys.path.insert(0, str(yolox_root))
    from yolox.data.data_augment import ValTransform
    from yolox.exp import get_exp
    from yolox.utils import postprocess

    exp = get_exp(str(config), None)
    exp.num_classes = 4
    model = exp.get_model().to(device).eval()
    checkpoint = load_torch_checkpoint(weights)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    for param in model.parameters():
        param.requires_grad_(True)
    preproc = ValTransform(legacy=False)
    test_size = tuple(exp.test_size)

    def preprocess(image: Image.Image) -> torch.Tensor:
        image_bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        tensor, _ = preproc(image_bgr, None, test_size)
        return torch.from_numpy(tensor).unsqueeze(0).float().to(device)

    def predict(_: Path, image: Image.Image) -> DetectionResult:
        tensor = preprocess(image)
        with torch.no_grad():
            outputs = postprocess(model(tensor), exp.num_classes, conf, exp.nmsthre)
        if outputs[0] is None:
            empty = torch.empty((0,), device=device)
            return DetectionResult(torch.empty((0, 4), device=device), empty.long(), empty)
        output = outputs[0].detach().to(device)
        image_w, image_h = image.size
        ratio = min(test_size[0] / image_h, test_size[1] / image_w)
        boxes = output[:, 0:4] / ratio
        boxes[:, 0::2].clamp_(0, image_w - 1)
        boxes[:, 1::2].clamp_(0, image_h - 1)
        scores = output[:, 4] * output[:, 5]
        return DetectionResult(boxes=boxes, cls=output[:, 6].long(), conf=scores)

    return DetectorBundle(
        name=name,
        backend="yolox",
        detector=model,
        cam_model=YoloCamModel(model).to(device).eval(),
        target_layers=[model.backbone.C3_n3, model.backbone.C3_n4],
        preprocess=preprocess,
        predict=predict,
    )


def load_dino_bundle(
    name: str,
    weights: Path,
    config: Path,
    conf: float,
    device: torch.device,
) -> DetectorBundle:
    dino_root = REPO_ROOT / "detector" / "DINO"
    if str(dino_root) not in sys.path:
        sys.path.insert(0, str(dino_root))
    prepare_dino_ops_path()
    import datasets.transforms as T
    from main import build_model_main
    from util.misc import nested_tensor_from_tensor_list
    from util.slconfig import SLConfig

    dino_args = SLConfig.fromfile(str(config))
    dino_args.device = str(device)
    model, _, postprocessors = build_model_main(dino_args)
    checkpoint = load_torch_checkpoint(weights)
    load_info = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    LOGGER.info("Loaded DINO state dict: missing=%d unexpected=%d", len(load_info.missing_keys), len(load_info.unexpected_keys))
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(True)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def preprocess(image: Image.Image) -> torch.Tensor:
        tensor, _ = transform(image, None)
        return tensor.unsqueeze(0).to(device)

    def predict(_: Path, image: Image.Image) -> DetectionResult:
        tensor = preprocess(image)
        target_sizes = torch.tensor([[image.height, image.width]], dtype=torch.float32, device=device)
        with torch.no_grad():
            output = model(tensor)
            result = postprocessors["bbox"](output, target_sizes)[0]
        keep = result["scores"] > conf
        return DetectionResult(
            boxes=result["boxes"][keep].detach().to(device),
            cls=result["labels"][keep].detach().to(device).long(),
            conf=result["scores"][keep].detach().to(device),
        )

    target_layer = model.backbone[0].body["layer4"]
    return DetectorBundle(
        name=name,
        backend="dino",
        detector=model,
        cam_model=DINOBackboneEigenCAM(model, nested_tensor_from_tensor_list),
        target_layers=[target_layer],
        preprocess=preprocess,
        predict=predict,
    )


def load_detectors(args: argparse.Namespace, device: torch.device) -> dict[str, DetectorBundle]:
    detectors: dict[str, DetectorBundle] = {}
    for name, spec in DETECTOR_SPECS.items():
        backend = spec["backend"]
        weights = spec["weights"]
        LOGGER.info("Loading %s (%s) weights from %s", name, backend, weights)
        try:
            if backend in {"yolo", "rtdetr"}:
                detectors[name] = load_ultralytics_bundle(name, backend, weights, args.imgsz, args.conf, device, args.device)
            elif backend == "yolox":
                detectors[name] = load_yolox_bundle(name, weights, spec["config"], args.conf, device)
            elif backend == "dino":
                detectors[name] = load_dino_bundle(name, weights, spec["config"], args.conf, device)
            else:
                raise ValueError(f"Unsupported detector backend: {backend}")
        except Exception as exc:
            LOGGER.error("Skip %s because loading failed: %s", name, exc)
            continue
        LOGGER.info("Loaded %s with %d target layer(s)", name, len(detectors[name].target_layers))
    if not detectors:
        raise RuntimeError("No detector loaded successfully.")
    return detectors


def draw_missing_detection(path: Path, image: Image.Image, message: str) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((8, 8, min(canvas.width - 1, 520), 34), fill=(255, 255, 255))
    draw.text((14, 15), message, fill=(180, 20, 20), font=font)
    canvas.save(path)


def main() -> None:
    setup_logging()
    args = parse_args()
    args.image_dir = (args.image_dir if args.image_dir.is_absolute() else (REPO_ROOT / args.image_dir)).resolve()
    if args.image is not None:
        args.image = (args.image if args.image.is_absolute() else (REPO_ROOT / args.image)).resolve()
    validate_paths(args)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = resolve_image_paths(args)
    detectors = load_detectors(args, device)
    target_layers = {name: bundle.target_layers for name, bundle in detectors.items()}
    cam_models = {name: bundle.cam_model for name, bundle in detectors.items()}
    backends = {name: bundle.backend for name, bundle in detectors.items()}
    LOGGER.info("Building CAM extractors for: %s", ", ".join(detectors))
    cam_extractors = build_cam_extractors(cam_models, target_layers, args.method, backends)
    LOGGER.info("Built CAM extractors")
    if args.image is not None:
        LOGGER.info("Visualizing single image: %s", image_paths[0])
    else:
        LOGGER.info("Sampled %d images from %s", len(image_paths), args.image_dir)

    try:
        for index, image_path in enumerate(image_paths, start=1):
            image = load_rgb(image_path)
            stem = output_stem(index, image_path, args.image is not None)
            original_path = args.output_dir / f"{stem}_original.jpg"
            image.save(original_path)
            LOGGER.info("Saved original image: %s", original_path)

            gt_path = args.output_dir / f"{stem}_gt.jpg"
            save_gt_image(
                image,
                image_path,
                args.image_dir,
                gt_path,
                args.draw_scale,
                args.box_width,
                args.label_font_size,
            )

            for model_name, bundle in detectors.items():
                tensor = bundle.preprocess(image)
                cam_path = args.output_dir / f"{stem}_{model_name}_{args.method}.jpg"
                cam = generate_cam(cam_extractors[model_name], tensor, args.method, args.eigen_smooth)

                det_path = args.output_dir / f"{stem}_{model_name}_detection.jpg"
                result = None
                try:
                    result = bundle.predict(image_path, image)
                    save_detection(
                        result,
                        image,
                        det_path,
                        args.draw_scale,
                        args.box_width,
                        args.label_font_size,
                    )
                except Exception as exc:
                    LOGGER.warning("Detection plotting failed for %s with %s: %s", image_path, model_name, exc)
                    draw_missing_detection(det_path, image, "Detection plotting failed; see log.")
                LOGGER.info("Saved %s detection result: %s", model_name, det_path)

                if args.box_renorm and result is not None:
                    boxes = detection_boxes(result, image.size, cam.shape)
                    cam = renormalize_cam_in_boxes(cam, boxes)
                cam = smooth_cam(cam, args.smooth_radius)
                overlay_cam(image, cam, args.heatmap_alpha, args.heatmap_gamma).save(cam_path)
                LOGGER.info("Saved %s heatmap: %s", model_name, cam_path)
    finally:
        release_cam_extractors(cam_extractors)

    LOGGER.info("Visualization complete. Output directory: %s", args.output_dir)


if __name__ == "__main__":
    main()

"""
Recommended operation:
python visualize/heapmap_visu.py --method eigencam --heatmap-alpha 0.6 --heatmap-gamma 0.8 --smooth-radius 8
softer:
python visualize/heapmap_visu.py --method eigencam --heatmap-alpha 0.6 --heatmap-gamma 0.9 --smooth-radius 12
"""
