"""Train a BoxMOT-compatible ReID model on image-folder crops.

Expected dataset layout:
    dataset_root/
      train/
        identity_000001/
          000001.jpg
          000002.jpg
      val/
        identity_000001/
          000001.jpg

The saved checkpoint keeps a BoxMOT-friendly ``state_dict`` payload. Make sure
the output filename contains the architecture name, for example
``blueberry_osnet_x0_25.pt``, because BoxMOT infers the backbone from the
checkpoint filename.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import types
from collections import OrderedDict
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


REPO_ROOT = Path(__file__).resolve().parents[4]
TRACKERS_ROOT = REPO_ROOT / "trackers"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(TRACKERS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKERS_ROOT))


class _CompatLogger:
    def __init__(self) -> None:
        self._logger = logging.getLogger("boxmot")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def remove(self, *args, **kwargs):
        return None

    def add(self, *args, **kwargs):
        return 1

    def opt(self, *args, **kwargs):
        return self

    def debug(self, message, *args, **kwargs):
        self._logger.debug(message)

    def info(self, message, *args, **kwargs):
        self._logger.info(message)

    def success(self, message, *args, **kwargs):
        self._logger.info(message)

    def warning(self, message, *args, **kwargs):
        self._logger.warning(message)

    def error(self, message, *args, **kwargs):
        self._logger.error(message)


if "loguru" not in sys.modules:
    loguru_module = types.ModuleType("loguru")
    loguru_module.logger = _CompatLogger()
    sys.modules["loguru"] = loguru_module

if "boxmot" not in sys.modules:
    boxmot_module = types.ModuleType("boxmot")
    boxmot_module.__path__ = [str(TRACKERS_ROOT / "boxmot")]
    sys.modules["boxmot"] = boxmot_module
if "boxmot.reid" not in sys.modules:
    reid_module = types.ModuleType("boxmot.reid")
    reid_module.__path__ = [str(TRACKERS_ROOT / "boxmot" / "reid")]
    sys.modules["boxmot.reid"] = reid_module
if "boxmot.reid.core" not in sys.modules:
    reid_core_module = types.ModuleType("boxmot.reid.core")
    reid_core_module.__path__ = [str(TRACKERS_ROOT / "boxmot" / "reid" / "core")]
    sys.modules["boxmot.reid.core"] = reid_core_module

from boxmot.reid.backbones.mobilenetv2 import mobilenetv2_x1_0, mobilenetv2_x1_4
from boxmot.reid.backbones.osnet import (
    osnet_ibn_x1_0,
    osnet_x0_25,
    osnet_x0_5,
    osnet_x0_75,
    osnet_x1_0,
)
from boxmot.reid.backbones.osnet_ain import osnet_ain_x1_0
from boxmot.reid.backbones.resnet import resnet101, resnet50


MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
TRAIN_MODEL_FACTORY = {
    "resnet50": resnet50,
    "resnet101": resnet101,
    "mobilenetv2_x1_0": mobilenetv2_x1_0,
    "mobilenetv2_x1_4": mobilenetv2_x1_4,
    "osnet_x1_0": osnet_x1_0,
    "osnet_x0_75": osnet_x0_75,
    "osnet_x0_5": osnet_x0_5,
    "osnet_x0_25": osnet_x0_25,
    "osnet_ibn_x1_0": osnet_ibn_x1_0,
    "osnet_ain_x1_0": osnet_ain_x1_0,
}
SUPPORTED_ARCHS = tuple(sorted(TRAIN_MODEL_FACTORY.keys()))


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    identity: str


class ReIDFolderDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_size: tuple[int, int],
        train: bool,
        hflip_prob: float,
        color_jitter: float,
        random_erasing: float,
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.train = train
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter
        self.random_erasing = random_erasing
        self.samples, self.label_to_identity = self._collect_samples(root)
        self.labels = [sample.label for sample in self.samples]
        self.index_by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.index_by_label[label].append(index)

    @staticmethod
    def _collect_samples(root: Path) -> tuple[list[Sample], dict[int, str]]:
        if not root.is_dir():
            raise FileNotFoundError(f"Missing ReID split folder: {root}")

        identity_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
        if not identity_dirs:
            raise ValueError(f"No identity folders found under: {root}")

        samples: list[Sample] = []
        label_to_identity: dict[int, str] = {}
        for label, identity_dir in enumerate(identity_dirs):
            label_to_identity[label] = identity_dir.name
            image_paths = sorted(
                path
                for path in identity_dir.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            )
            for image_path in image_paths:
                samples.append(Sample(path=image_path, label=label, identity=identity_dir.name))

        if not samples:
            raise ValueError(f"No crop images found under: {root}")
        return samples, label_to_identity

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_color_jitter(self, image: Image.Image) -> Image.Image:
        if self.color_jitter <= 0.0:
            return image

        jitter = self.color_jitter
        brightness = random.uniform(max(0.0, 1.0 - jitter), 1.0 + jitter)
        contrast = random.uniform(max(0.0, 1.0 - jitter), 1.0 + jitter)
        saturation = random.uniform(max(0.0, 1.0 - jitter), 1.0 + jitter)

        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)
        return image

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize((self.image_size[1], self.image_size[0]), resample=Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - MEAN) / STD
        return tensor

    def _apply_random_erasing(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.train or self.random_erasing <= 0.0:
            return tensor
        if random.random() > self.random_erasing:
            return tensor

        _, height, width = tensor.shape
        area = height * width
        for _ in range(10):
            erase_area = random.uniform(0.02, 0.12) * area
            aspect = random.uniform(0.3, 3.0)
            erase_h = int(round((erase_area * aspect) ** 0.5))
            erase_w = int(round((erase_area / aspect) ** 0.5))
            if erase_h >= height or erase_w >= width or erase_h <= 0 or erase_w <= 0:
                continue
            top = random.randint(0, height - erase_h)
            left = random.randint(0, width - erase_w)
            tensor[:, top:top + erase_h, left:left + erase_w] = 0.0
            break
        return tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.train:
            if random.random() < self.hflip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = self._apply_color_jitter(image)
        tensor = self._to_tensor(image)
        tensor = self._apply_random_erasing(tensor)
        return tensor, sample.label


class RandomIdentitySampler(Sampler[int]):
    def __init__(self, labels: list[int], batch_size: int, instances_per_identity: int) -> None:
        if batch_size % instances_per_identity != 0:
            raise ValueError("batch-size must be divisible by instances-per-identity")
        self.labels = labels
        self.batch_size = batch_size
        self.instances_per_identity = instances_per_identity
        self.identities_per_batch = batch_size // instances_per_identity
        self.index_by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.index_by_label[label].append(index)
        self.unique_labels = sorted(self.index_by_label.keys())
        self.length = max(batch_size, len(labels) - (len(labels) % batch_size))

    def __iter__(self):
        batches: list[int] = []
        while len(batches) < self.length:
            chosen_labels = random.sample(
                self.unique_labels,
                k=min(self.identities_per_batch, len(self.unique_labels)),
            )
            if len(chosen_labels) < self.identities_per_batch:
                chosen_labels += random.choices(
                    self.unique_labels,
                    k=self.identities_per_batch - len(chosen_labels),
                )
            for label in chosen_labels:
                indices = self.index_by_label[label]
                if len(indices) >= self.instances_per_identity:
                    batches.extend(random.sample(indices, self.instances_per_identity))
                else:
                    batches.extend(random.choices(indices, k=self.instances_per_identity))
        return iter(batches[: self.length])

    def __len__(self) -> int:
        return self.length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BoxMOT-compatible ReID model on image-folder crops."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Image-folder ReID dataset root containing train/ and val/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Checkpoint path. Filename must contain the arch name, e.g. blueberry_osnet_x0_25.pt",
    )
    parser.add_argument("--arch", type=str, default="osnet_x0_25", choices=SUPPORTED_ARCHS)
    parser.add_argument("--epochs", type=int, default=80)
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
        help="Crop height used during training. Keep this aligned with BoxMOT inference preprocessing.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Crop width used during training. Keep this aligned with BoxMOT inference preprocessing.",
    )
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--color-jitter", type=float, default=0.1)
    parser.add_argument("--random-erasing", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="Optional local .pt checkpoint used to initialize matching layers.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a previous checkpoint produced by this script.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialize from the backbone's ImageNet weights when supported by boxmot/reid.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=12,
        help="Stop after this many non-improving epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum score improvement required to reset early stopping.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace) -> None:
    if args.arch not in args.output.name:
        raise ValueError(
            f"Output filename must contain the selected arch '{args.arch}' so BoxMOT can infer the backbone: {args.output}"
        )
    if args.output.suffix.lower() != ".pt":
        raise ValueError(f"Output checkpoint must use a .pt suffix: {args.output}")
    if args.instances_per_identity < 2:
        raise ValueError("--instances-per-identity must be at least 2 for triplet loss")


def load_matching_weights(model: nn.Module, weight_path: Path) -> None:
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_dict = model.state_dict()

    new_state_dict = OrderedDict()
    matched_layers = 0
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key in model_dict and model_dict[clean_key].shape == value.shape:
            new_state_dict[clean_key] = value
            matched_layers += 1

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print(f"loaded_matching_tensors={matched_layers} source={weight_path}")


def build_model(
    arch: str,
    num_classes: int,
    use_pretrained: bool,
    init_weights: Path | None,
) -> nn.Module:
    if arch not in TRAIN_MODEL_FACTORY:
        raise KeyError(f"Unsupported training arch '{arch}'. Choices: {SUPPORTED_ARCHS}")

    model = TRAIN_MODEL_FACTORY[arch](
        num_classes=num_classes,
        loss="triplet",
        pretrained=use_pretrained,
    )
    if init_weights is not None:
        load_matching_weights(model, init_weights)
    return model


def pairwise_distance(embeddings: torch.Tensor) -> torch.Tensor:
    squared = embeddings.pow(2).sum(dim=1, keepdim=True)
    distances = squared + squared.t() - 2.0 * embeddings @ embeddings.t()
    return distances.clamp_min_(1e-12).sqrt_()


def hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    distances = pairwise_distance(F.normalize(embeddings, p=2, dim=1))
    labels = labels.view(-1, 1)
    same = labels.eq(labels.t())
    eye = torch.eye(labels.size(0), device=labels.device, dtype=torch.bool)
    same = same & ~eye
    diff = ~labels.eq(labels.t())

    loss_terms: list[torch.Tensor] = []
    for index in range(labels.size(0)):
        positive_mask = same[index]
        negative_mask = diff[index]
        if not positive_mask.any() or not negative_mask.any():
            continue
        hardest_positive = distances[index][positive_mask].max()
        hardest_negative = distances[index][negative_mask].min()
        loss_terms.append(F.relu(margin + hardest_positive - hardest_negative))

    if not loss_terms:
        return embeddings.new_tensor(0.0)
    return torch.stack(loss_terms).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    ce_loss_fn = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        embeddings = model(images)
        logits = model.classifier(embeddings)
        ce_loss = ce_loss_fn(logits, labels)
        total_loss += float(ce_loss.item()) * labels.size(0)
        total_samples += labels.size(0)
        all_embeddings.append(F.normalize(embeddings.detach().cpu(), p=2, dim=1))
        all_labels.append(labels.detach().cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    distances = torch.cdist(embeddings, embeddings, p=2)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.argmin(dim=1)
    top1 = float((labels[nearest] == labels).float().mean().item())

    return {
        "val_loss": total_loss / max(1, total_samples),
        "val_top1": top1,
    }


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    arch: str,
    num_classes: int,
    epoch: int,
    metrics: dict[str, float],
    label_to_identity: dict[int, str],
    image_size: tuple[int, int],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    best_score: float | None = None,
    epochs_without_improvement: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "arch": arch,
        "num_classes": num_classes,
        "epoch": epoch,
        "metrics": metrics,
        "image_size": {"height": image_size[0], "width": image_size[1]},
        "label_to_identity": {str(k): v for k, v in label_to_identity.items()},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if best_score is not None:
        payload["best_score"] = best_score
    if epochs_without_improvement is not None:
        payload["epochs_without_improvement"] = epochs_without_improvement
    torch.save(payload, output_path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    image_size = (args.height, args.width)

    train_dataset = ReIDFolderDataset(
        root=args.data_root / "train",
        image_size=image_size,
        train=True,
        hflip_prob=args.hflip_prob,
        color_jitter=args.color_jitter,
        random_erasing=args.random_erasing,
    )
    val_dataset = ReIDFolderDataset(
        root=args.data_root / "val",
        image_size=image_size,
        train=False,
        hflip_prob=0.0,
        color_jitter=0.0,
        random_erasing=0.0,
    )

    num_classes = len(train_dataset.label_to_identity)
    model = build_model(
        arch=args.arch,
        num_classes=num_classes,
        use_pretrained=bool(args.pretrained and args.init_weights is None and args.resume is None),
        init_weights=args.init_weights,
    )
    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomIdentitySampler(
            labels=train_dataset.labels,
            batch_size=args.batch_size,
            instances_per_identity=args.instances_per_identity,
        ),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_score = -1.0
    best_state = None
    start_epoch = 1
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    last_output = args.output.with_name(f"{args.output.stem}_last{args.output.suffix}")

    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError(f"Resume checkpoint is missing state_dict: {args.resume}")
        if payload.get("arch") and payload["arch"] != args.arch:
            raise ValueError(
                f"Resume checkpoint arch mismatch: expected {args.arch}, got {payload['arch']}"
            )
        model.load_state_dict(payload["state_dict"])
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_score = float(payload.get("best_score", -1.0))
        epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
        best_state = copy.deepcopy(model.state_dict())
        history_path = args.output.with_suffix(".history.json")
        if history_path.is_file():
            with history_path.open("r", encoding="utf-8") as handle:
                loaded_history = json.load(handle)
            if isinstance(loaded_history, list):
                history = loaded_history
        print(
            f"resumed_from={args.resume} start_epoch={start_epoch:03d} "
            f"best_score={best_score:.6f}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_tri = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, embeddings = model(images)
            ce_loss = ce_loss_fn(logits, labels)
            tri_loss = hard_triplet_loss(embeddings, labels, margin=args.margin)
            loss = args.ce_weight * ce_loss + args.tri_weight * tri_loss
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            running_ce += float(ce_loss.item()) * batch_size
            running_tri += float(tri_loss.item()) * batch_size
            seen += batch_size

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        metrics.update(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / max(1, seen),
                "train_ce": running_ce / max(1, seen),
                "train_tri": running_tri / max(1, seen),
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
        history.append(metrics)

        score = metrics["val_top1"] - 0.05 * metrics["val_loss"]
        improved = score > (best_score + args.early_stop_min_delta)

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(metrics["train_loss"], 6),
                    "train_ce": round(metrics["train_ce"], 6),
                    "train_tri": round(metrics["train_tri"], 6),
                    "val_loss": round(metrics["val_loss"], 6),
                    "val_top1": round(metrics["val_top1"], 6),
                    "lr": round(metrics["lr"], 8),
                    "score": round(score, 6),
                    "best_score": round(best_score, 6),
                    "bad_epochs": epochs_without_improvement,
                },
                ensure_ascii=False,
            )
        )

        if improved:
            best_score = score
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                output_path=args.output,
                model=model,
                arch=args.arch,
                num_classes=num_classes,
                epoch=epoch,
                metrics=metrics,
                label_to_identity=train_dataset.label_to_identity,
                image_size=image_size,
                optimizer=optimizer,
                scheduler=scheduler,
                best_score=best_score,
                epochs_without_improvement=epochs_without_improvement,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            output_path=last_output,
            model=model,
            arch=args.arch,
            num_classes=num_classes,
            epoch=epoch,
            metrics=metrics,
            label_to_identity=train_dataset.label_to_identity,
            image_size=image_size,
            optimizer=optimizer,
            scheduler=scheduler,
            best_score=best_score,
            epochs_without_improvement=epochs_without_improvement,
        )

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "epoch": epoch,
                        "patience": args.early_stop_patience,
                        "best_score": round(best_score, 6),
                    },
                    ensure_ascii=False,
                )
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_path = args.output.with_suffix(".history.json")
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)

    summary = {
        "output": str(args.output.resolve()),
        "last_output": str(last_output.resolve()),
        "epochs": args.epochs,
        "arch": args.arch,
        "num_classes": num_classes,
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "history": str(history_path.resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
