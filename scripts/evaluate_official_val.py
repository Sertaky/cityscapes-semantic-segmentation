"""Evaluate the frozen Baseline 05 checkpoint once on official Cityscapes val."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cityscapes_segmentation.analysis.metrics import SegmentationMetrics  # noqa: E402
from cityscapes_segmentation.data.dataset import CityscapesSegmentationDataset  # noqa: E402
from cityscapes_segmentation.data.labels import (  # noqa: E402
    get_ignore_index,
    get_train_id_classes,
    get_train_id_to_name,
)
from cityscapes_segmentation.data.transforms import SegmentationTransform  # noqa: E402
from cityscapes_segmentation.engine import load_checkpoint  # noqa: E402
from cityscapes_segmentation.models import UNet  # noqa: E402


EXPERIMENT = "baseline_05_dilated_bottleneck"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / EXPERIMENT / "best.pt"
DEV_METRICS = PROJECT_ROOT / "outputs" / EXPERIMENT / "best_metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_official_val"
RESOLUTION = (640, 320)  # width, height
EXPECTED_IMAGES = 500
EXPECTED_EPOCH = 30
EXPECTED_PARAMETERS = 31_038_803
QUALITATIVE_INDICES = tuple(np.linspace(0, EXPECTED_IMAGES - 1, 8, dtype=int).tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def official_val_images(dataset_root: Path) -> tuple[Path, ...]:
    image_root = dataset_root / "leftImg8bit" / "val"
    mask_root = dataset_root / "gtFine" / "val"
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError("Official Cityscapes leftImg8bit/val or gtFine/val is missing")

    images = tuple(sorted(image_root.glob("*/*_leftImg8bit.png")))
    masks = tuple(sorted(mask_root.glob("*/*_gtFine_labelIds.png")))
    if len(images) != EXPECTED_IMAGES or len(masks) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_IMAGES} official-val images and labelIds masks; "
            f"found {len(images)} and {len(masks)}"
        )
    expected_masks = {
        mask_root / image.parent.name / f"{image.name.removesuffix('_leftImg8bit.png')}_gtFine_labelIds.png"
        for image in images
    }
    if expected_masks != set(masks):
        missing = sorted(expected_masks - set(masks))
        unexpected = sorted(set(masks) - expected_masks)
        raise RuntimeError(f"Official-val pairing mismatch: missing={missing[:3]}, unexpected={unexpected[:3]}")
    return images


def load_frozen_model(device: torch.device) -> tuple[UNet, object]:
    model = UNet(in_channels=3, num_classes=19, base_channels=64, bottleneck_dilation=2)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("Frozen B05 parameter count changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    metadata = load_checkpoint(CHECKPOINT, model=model, optimizer=optimizer, map_location="cpu")
    config = metadata.config
    checks = {
        "epoch": metadata.epoch == EXPECTED_EPOCH,
        "experiment": config.get("experiment_name") == EXPERIMENT,
        "model": config.get("model") == "UNet",
        "base_channels": int(config.get("base_channels", -1)) == 64,
        "num_classes": int(config.get("num_classes", -1)) == 19,
        "bottleneck_dilation": int(config.get("bottleneck_dilation", -1)) == 2,
        "resolution_width": int(config.get("resolution_width", -1)) == RESOLUTION[0],
        "resolution_height": int(config.get("resolution_height", -1)) == RESOLUTION[1],
        "precision": config.get("precision") == "FP32",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Frozen checkpoint identity check failed: {failed}")
    del optimizer
    model.to(device).eval()
    return model, metadata


def class_rows(result) -> list[dict]:
    matrix = result.confusion_matrix.numpy()
    rows = []
    for train_id in range(19):
        support = int(result.support[train_id])
        predicted = int(matrix[:, train_id].sum())
        tp = int(result.true_positives[train_id])
        fp = int(result.false_positives[train_id])
        fn = int(result.false_negatives[train_id])
        precision = tp / predicted if predicted else math.nan
        recall = tp / support if support else math.nan
        rows.append(
            {
                "train_id": train_id,
                "class_name": result.class_names[train_id],
                "iou": float(result.per_class_iou[train_id]),
                "dice": float(result.per_class_dice[train_id]),
                "precision": precision,
                "recall": recall,
                "support": support,
                "predicted_pixels": predicted,
                "prediction_gt_ratio": predicted / support if support else math.nan,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def colorize(mask: np.ndarray) -> np.ndarray:
    colors = np.zeros((256, 3), dtype=np.uint8)
    for label in get_train_id_classes():
        colors[label.trainId] = label.color
    colors[get_ignore_index()] = (0, 0, 0)
    return colors[mask.astype(np.uint8)]


def save_example(index: int, image_path: Path, target: np.ndarray, prediction: np.ndarray, output_dir: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize(RESOLUTION, Image.Resampling.BILINEAR)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, content, title in zip(
        axes,
        (np.asarray(image), colorize(target), colorize(prediction)),
        ("RGB", "Ground truth", "Frozen B05 prediction"),
        strict=True,
    ):
        axis.imshow(content)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"Official val index {index}: {image_path.stem}")
    figure.savefig(output_dir / f"{index:03d}_{image_path.stem}.png", dpi=140)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    dataset_root = args.dataset_root.expanduser().resolve()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Final official-val evaluation requires a healthy CUDA device")

    images = official_val_images(dataset_root)
    transform = SegmentationTransform(size=RESOLUTION, horizontal_flip_probability=0.0)
    dataset = CityscapesSegmentationDataset(dataset_root, images, transform)
    if len(dataset) != EXPECTED_IMAGES or any(path.parts[-3] != "val" for path, _ in dataset.samples):
        raise RuntimeError("Evaluator resolved data outside the 500-image official validation split")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model, metadata = load_frozen_model(device)
    output_dir = OUTPUT_DIR
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    meter = SegmentationMetrics(num_classes=19, ignore_index=255)
    selected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    seen = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_images, batch_targets in loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            logits = model(batch_images)
            if logits.shape != (batch_images.shape[0], 19, RESOLUTION[1], RESOLUTION[0]):
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Non-finite official-val logits detected")
            predictions = logits.argmax(dim=1)
            meter.update(predictions, batch_targets)
            for offset in range(batch_images.shape[0]):
                index = seen + offset
                if index in QUALITATIVE_INDICES:
                    selected[index] = (
                        batch_targets[offset].cpu().numpy().astype(np.uint8),
                        predictions[offset].cpu().numpy().astype(np.uint8),
                    )
            seen += batch_images.shape[0]
    torch.cuda.synchronize()
    runtime = time.perf_counter() - started
    result = meter.compute()
    if seen != EXPECTED_IMAGES or int(result.confusion_matrix.sum()) <= 0:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES} evaluated images; processed {seen}")

    per_class = class_rows(result)
    numeric_metrics = [result.miou, result.mean_dice, result.pixel_accuracy]
    numeric_metrics.extend(row[key] for row in per_class for key in ("iou", "dice", "precision", "recall"))
    if not all(math.isfinite(value) for value in numeric_metrics):
        raise RuntimeError("Undefined or non-finite official-val metric detected")

    for index in QUALITATIVE_INDICES:
        target, prediction = selected[index]
        save_example(index, dataset.samples[index][0], target, prediction, examples_dir)

    dev = read_json(DEV_METRICS)
    dev_by_id = {int(row["train_id"]): row for row in dev["per_class"]}
    comparison = []
    for row in per_class:
        old = dev_by_id[row["train_id"]]
        comparison.append(
            {
                "train_id": row["train_id"],
                "class_name": row["class_name"],
                "dev_iou": old["iou"],
                "official_val_iou": row["iou"],
                "delta_iou": row["iou"] - old["iou"],
                "dev_precision": old["precision"],
                "official_val_precision": row["precision"],
                "dev_recall": old["recall"],
                "official_val_recall": row["recall"],
            }
        )

    timestamp = datetime.now(timezone.utc).isoformat()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    config = {
        "evaluation": "final_official_cityscapes_validation",
        "dataset_split": "official_val",
        "official_val_images": EXPECTED_IMAGES,
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": metadata.epoch,
        "experiment_identity": metadata.config["experiment_name"],
        "architecture": "UNet(in_channels=3,num_classes=19,base_channels=64,bottleneck_dilation=2)",
        "input_resolution": {"width": RESOLUTION[0], "height": RESOLUTION[1]},
        "preprocessing": {
            "rgb_resize": "bilinear",
            "mask_resize": "nearest",
            "normalization": "ImageNet mean/std",
            "augmentation": "none",
            "test_time_augmentation": False,
        },
        "ignore_index": 255,
        "precision": "FP32",
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "evaluation_timestamp_utc": timestamp,
    }
    summary = {
        **config,
        "evaluation_runtime_seconds": runtime,
        "images_per_second": EXPECTED_IMAGES / runtime,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "miou": result.miou,
        "mean_dice": result.mean_dice,
        "pixel_accuracy": result.pixel_accuracy,
        "development_reference": {
            "split": "313-image city-separated development validation",
            "miou": dev["best_miou"],
            "mean_dice": dev["mean_dice"],
            "pixel_accuracy": dev["pixel_accuracy"],
        },
        "aggregate_delta_official_minus_dev": {
            "miou": result.miou - dev["best_miou"],
            "mean_dice": result.mean_dice - dev["mean_dice"],
            "pixel_accuracy": result.pixel_accuracy - dev["pixel_accuracy"],
        },
        "per_class": per_class,
        "qualitative_indices": list(QUALITATIVE_INDICES),
        "official_test_loaded": False,
        "training_performed": False,
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "official_val_metrics.json", summary)
    write_csv(output_dir / "per_class_metrics.csv", per_class)
    write_csv(output_dir / "dev_vs_official_val.csv", comparison)
    np.savetxt(output_dir / "confusion_matrix.csv", result.confusion_matrix.numpy(), delimiter=",", fmt="%d")
    print(json.dumps({
        "completed": True,
        "images": seen,
        "miou": result.miou,
        "mean_dice": result.mean_dice,
        "pixel_accuracy": result.pixel_accuracy,
        "runtime_seconds": runtime,
        "images_per_second": EXPECTED_IMAGES / runtime,
        "peak_cuda_allocated_mib": peak_allocated / 1024**2,
        "peak_cuda_reserved_mib": peak_reserved / 1024**2,
        "official_test_loaded": False,
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
