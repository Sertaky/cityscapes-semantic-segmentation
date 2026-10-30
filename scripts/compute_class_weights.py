"""Compute candidate Baseline 02 class weights from dev-train masks only."""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cityscapes_segmentation.analysis.class_weights import (  # noqa: E402
    NUM_CLASSES,
    compute_class_weights,
    count_raw_label_mask,
)
from cityscapes_segmentation.data.dataset import CityscapesSegmentationDataset  # noqa: E402
from cityscapes_segmentation.data.labels import get_ignore_index, get_train_id_to_name  # noqa: E402
from cityscapes_segmentation.data.splits import load_dev_split, validate_dev_split  # noqa: E402
from cityscapes_segmentation.data.transforms import make_train_transform  # noqa: E402
from cityscapes_segmentation.models import UNet  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_02_weighted_ce"
CSV_PATH = OUTPUT_DIR / "class_weights.csv"
JSON_PATH = OUTPUT_DIR / "class_weights.json"


def save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def semantic_mask_path(image_path: Path) -> Path:
    suffix = "_leftImg8bit.png"
    stem = image_path.name.removesuffix(suffix)
    return PROJECT_ROOT / "gtFine" / "train" / image_path.parent.name / f"{stem}_gtFine_labelIds.png"


def count_dev_train_pixels(image_paths: tuple[Path, ...]) -> tuple[np.ndarray, int, int]:
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    ignored_pixels = 0
    total_pixels = 0
    for index, image_path in enumerate(image_paths, start=1):
        mask_path = semantic_mask_path(image_path)
        with Image.open(mask_path) as source:
            raw_mask = np.array(source)
        counted = count_raw_label_mask(raw_mask)
        class_counts += counted.class_counts
        ignored_pixels += counted.ignored_pixels
        total_pixels += counted.total_pixels
        if index % 250 == 0 or index == len(image_paths):
            print(json.dumps({"masks_counted": index, "total_masks": len(image_paths)}), flush=True)
    return class_counts, ignored_pixels, total_pixels


def weighted_loss_smoke(image_paths: tuple[Path, ...], weights: np.ndarray) -> dict:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CityscapesSegmentationDataset(
        PROJECT_ROOT,
        image_paths[:2],
        make_train_transform(rng=random.Random(42)),
    )
    images, targets = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    images = images.to(device)
    targets = targets.to(device)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    if class_weights.shape != (NUM_CLASSES,):
        raise RuntimeError(f"Weight tensor shape differs from ({NUM_CLASSES},): {tuple(class_weights.shape)}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=get_ignore_index())
    model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_channels=64).to(device).eval()
    with torch.inference_mode():
        logits = model(images)
        loss = criterion(logits, targets)
        ignored = targets == get_ignore_index()
        changed_logits = logits.clone()
        changed_logits[:, 0][ignored] = 1_000.0
        changed_loss = criterion(changed_logits, targets)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f"Weighted loss is not finite: {float(loss)}")
    if int(ignored.sum()) == 0:
        raise RuntimeError("Real smoke batch contains no ignore pixels; ignore behavior was not exercised")
    ignore_invariant = bool(torch.allclose(loss, changed_loss, rtol=0, atol=1e-7))
    if not ignore_invariant:
        raise RuntimeError("Changing logits at ignored pixels changed the weighted loss")
    return {
        "device": str(device),
        "weight_shape": list(class_weights.shape),
        "weight_dtype": str(class_weights.dtype),
        "weight_device": str(class_weights.device),
        "image_shape": list(images.shape),
        "target_shape": list(targets.shape),
        "logits_shape": list(logits.shape),
        "ignored_pixels_in_batch": int(ignored.sum()),
        "loss": float(loss),
        "finite_loss": True,
        "ignore_index": get_ignore_index(),
        "ignore_invariance_verified": True,
        "forward_only": True,
    }


def main() -> None:
    started = time.perf_counter()
    split = load_dev_split()
    resolved = validate_dev_split(split, PROJECT_ROOT)
    dev_train = resolved.dev_train_images
    if len(dev_train) != 2662:
        raise RuntimeError(f"Expected 2662 dev-train images; found {len(dev_train)}")
    official_train = (PROJECT_ROOT / "leftImg8bit" / "train").resolve()
    if any(not path.resolve().is_relative_to(official_train) for path in dev_train):
        raise RuntimeError("A weighting-source image is outside official Cityscapes train")
    val_cities = set(split.dev_val_cities)
    if any(path.parent.name in val_cities for path in dev_train):
        raise RuntimeError("A dev-validation city entered the weighting source")

    class_counts, ignored_pixels, total_pixels = count_dev_train_pixels(dev_train)
    valid_pixels = int(class_counts.sum())
    if valid_pixels + ignored_pixels != total_pixels:
        raise RuntimeError("Valid and ignored counts do not cover all dev-train pixels")
    result = compute_class_weights(class_counts)
    weights = result.sqrt_inverse_normalized
    if not (
        weights.shape == (19,)
        and np.all(np.isfinite(weights))
        and np.all(weights > 0)
        and math.isclose(float(weights.mean()), 1.0, rel_tol=0, abs_tol=1e-12)
    ):
        raise RuntimeError("Candidate weight sanity checks failed")

    smoke = weighted_loss_smoke(dev_train, weights)
    names = get_train_id_to_name()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for train_id in range(NUM_CLASSES):
        rows.append({
            "train_id": train_id,
            "class_name": names[train_id],
            "pixel_count": int(class_counts[train_id]),
            "valid_pixel_frequency": float(result.frequencies[train_id]),
            "sqrt_inverse_raw_weight": float(result.sqrt_inverse_raw[train_id]),
            "normalized_weight": float(weights[train_id]),
            "relative_to_unweighted": float(weights[train_id]),
            "diagnostic_inverse_normalized_weight": float(result.inverse_normalized[train_id]),
        })
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "source": {
            "split": "dev_train",
            "source_official_split": "train",
            "image_count": len(dev_train),
            "cities": list(split.dev_train_cities),
            "excluded_dev_val_cities": list(split.dev_val_cities),
            "official_val_test_used": False,
        },
        "formula": {
            "frequency": "class_valid_pixels / total_valid_pixels",
            "raw_weight": "1 / sqrt(frequency)",
            "normalization": "raw_weight / mean(raw_weights)",
        },
        "pixels": {
            "total": total_pixels,
            "valid": valid_pixels,
            "ignored": ignored_pixels,
            "ignored_percentage": 100.0 * ignored_pixels / total_pixels,
        },
        "sanity_checks": {
            "weight_count": int(weights.size),
            "all_finite": bool(np.all(np.isfinite(weights))),
            "all_positive": bool(np.all(weights > 0)),
            "no_zero_frequency": bool(np.all(result.frequencies > 0)),
            "mean_normalized_weight": float(weights.mean()),
            "minimum_weight": float(weights.min()),
            "maximum_weight": float(weights.max()),
            "max_min_ratio": float(weights.max() / weights.min()),
        },
        "diagnostic_pure_inverse": {
            "minimum_weight": float(result.inverse_normalized.min()),
            "maximum_weight": float(result.inverse_normalized.max()),
            "max_min_ratio": float(result.inverse_normalized.max() / result.inverse_normalized.min()),
            "candidate_for_training": False,
        },
        "classes": rows,
        "weighted_loss_smoke": smoke,
        "baseline_02_preview": {
            "only_change": "CrossEntropyLoss weight argument",
            "loss": "CrossEntropyLoss(weight=<dev-train sqrt-inverse mean-one weights>, ignore_index=255)",
            "training_started": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    save_json(JSON_PATH, payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
