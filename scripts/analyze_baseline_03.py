"""Residual failure analysis for Baseline 03 on development validation only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from matplotlib.patches import Patch
from PIL import Image
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_baseline_01 import colorize, diagnostic_rows  # noqa: E402
from analyze_baseline_02 import add_matrix, metric_summary  # noqa: E402
from cityscapes_segmentation.data.dataset import CityscapesSegmentationDataset  # noqa: E402
from cityscapes_segmentation.data.labels import (  # noqa: E402
    get_ignore_index,
    get_train_id_classes,
    get_train_id_to_name,
)
from cityscapes_segmentation.data.splits import load_dev_split, validate_dev_split  # noqa: E402
from cityscapes_segmentation.data.transforms import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    SegmentationTransform,
    make_val_transform,
)
from cityscapes_segmentation.engine import load_checkpoint  # noqa: E402
from cityscapes_segmentation.models import UNet  # noqa: E402


NUM_CLASSES = 19
FOCUS_CLASSES = (12, 17, 16, 15)  # rider, motorcycle, train, bus
SECONDARY_CLASSES = (14, 6, 18)  # truck, traffic light, bicycle
EXPERIMENT = "baseline_03_mild_weighted_ce"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / EXPERIMENT
CHECKPOINT = PROJECT_ROOT / "checkpoints" / EXPERIMENT / "best.pt"
OUTPUT_DIR = EXPERIMENT_DIR / "error_analysis"
EXAMPLES_DIR = OUTPUT_DIR / "examples"
COMPARE_DIR = OUTPUT_DIR / "b02_vs_b03"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_model(experiment: str, expected_epoch: int, device: torch.device) -> tuple[UNet, object]:
    model = UNet(in_channels=3, num_classes=19, base_channels=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    metadata = load_checkpoint(
        PROJECT_ROOT / "checkpoints" / experiment / "best.pt",
        model=model,
        optimizer=optimizer,
        map_location=device,
    )
    expected = {"model": "UNet", "in_channels": 3, "num_classes": 19, "base_channels": 64}
    mismatches = {key: (metadata.config.get(key), value) for key, value in expected.items() if metadata.config.get(key) != value}
    if metadata.epoch != expected_epoch or mismatches:
        raise RuntimeError(f"Checkpoint identity mismatch: epoch={metadata.epoch}, config={mismatches}")
    model.eval()
    return model, metadata


def component_stats(mask: np.ndarray, prediction_mask: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Return 4-connected component (area, overlap) pairs for one binary mask."""
    height, width = mask.shape
    remaining = set(np.flatnonzero(mask).tolist())
    components: list[tuple[int, int]] = []
    prediction_flat = prediction_mask.ravel() if prediction_mask is not None else None
    while remaining:
        first = remaining.pop()
        stack = [first]
        area = 0
        overlap = 0
        while stack:
            index = stack.pop()
            area += 1
            if prediction_flat is not None and prediction_flat[index]:
                overlap += 1
            row, column = divmod(index, width)
            if row and index - width in remaining:
                remaining.remove(index - width)
                stack.append(index - width)
            if row + 1 < height and index + width in remaining:
                remaining.remove(index + width)
                stack.append(index + width)
            if column and index - 1 in remaining:
                remaining.remove(index - 1)
                stack.append(index - 1)
            if column + 1 < width and index + 1 in remaining:
                remaining.remove(index + 1)
                stack.append(index + 1)
        components.append((area, overlap))
    return components


def class_boundary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Return an inward heuristic boundary band for one GT class mask."""
    tensor = torch.from_numpy(mask).to(torch.float32)[None, None]
    kernel = 2 * radius + 1
    eroded = 1.0 - functional.max_pool2d(1.0 - tensor, kernel_size=kernel, stride=1, padding=radius)
    return mask & ~(eroded[0, 0].numpy() > 0.5)


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return result


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def confusion_rows(matrices: dict[str, np.ndarray], names: dict[int, str]) -> list[dict]:
    rows = []
    for class_id in FOCUS_CLASSES:
        for model_name, matrix in matrices.items():
            gt_total = int(matrix[class_id].sum())
            fp_total = int(matrix[:, class_id].sum() - matrix[class_id, class_id])
            fn_values = sorted(
                ((other, int(matrix[class_id, other])) for other in range(NUM_CLASSES) if other != class_id),
                key=lambda item: (-item[1], item[0]),
            )[:5]
            fp_values = sorted(
                ((other, int(matrix[other, class_id])) for other in range(NUM_CLASSES) if other != class_id),
                key=lambda item: (-item[1], item[0]),
            )[:5]
            for direction, values, denominator in (
                ("fn_destination", fn_values, gt_total),
                ("fp_source", fp_values, fp_total),
            ):
                for rank, (other, pixels) in enumerate(values, start=1):
                    rows.append({
                        "model": model_name,
                        "train_id": class_id,
                        "class_name": names[class_id],
                        "direction": direction,
                        "rank": rank,
                        "other_train_id": other,
                        "other_class": names[other],
                        "pixels": pixels,
                        "percentage": 100.0 * pixels / denominator if denominator else math.nan,
                    })
    return rows


def evaluate(model, dataset, loader, device, raw_ids):
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    city_matrices = defaultdict(lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64))
    image_rows: list[dict] = []
    components: list[dict] = []
    original_components: list[dict] = []
    boundary = {class_id: {region: [0, 0] for region in ("boundary", "interior")} for class_id in FOCUS_CLASSES}
    offset = 0
    with torch.inference_mode():
        for images, targets in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu().numpy().astype(np.uint8)
            targets_np = targets.numpy().astype(np.uint8)
            for batch_index, (target, prediction) in enumerate(zip(targets_np, predictions, strict=True)):
                dataset_index = offset + batch_index
                image_path, mask_path = dataset.samples[dataset_index]
                city = image_path.parent.name
                image_id = image_path.name.removesuffix("_leftImg8bit.png")
                add_matrix(matrix, target, prediction)
                add_matrix(city_matrices[city], target, prediction)
                with Image.open(mask_path) as source:
                    raw_mask = np.asarray(source)
                for class_id in FOCUS_CLASSES:
                    gt = target == class_id
                    pred = prediction == class_id
                    tp = int((gt & pred).sum())
                    fp = int((~gt & pred & (target != get_ignore_index())).sum())
                    fn = int((gt & ~pred).sum())
                    image_rows.append({
                        "dataset_index": dataset_index,
                        "city": city,
                        "image_id": image_id,
                        "train_id": class_id,
                        "class_name": get_train_id_to_name()[class_id],
                        "gt_pixels": int(gt.sum()),
                        "predicted_pixels": int(pred.sum()),
                        "tp_pixels": tp,
                        "fp_pixels": fp,
                        "fn_pixels": fn,
                        "total_error_pixels": fp + fn,
                    })
                    for component_index, (area, overlap) in enumerate(component_stats(gt, pred), start=1):
                        components.append({
                            "dataset_index": dataset_index,
                            "city": city,
                            "image_id": image_id,
                            "train_id": class_id,
                            "class_name": get_train_id_to_name()[class_id],
                            "component_index": component_index,
                            "area_256x512": area,
                            "overlap_pixels": overlap,
                            "recall_proxy": overlap / area,
                        })
                    for component_index, (area, _) in enumerate(component_stats(raw_mask == raw_ids[class_id]), start=1):
                        original_components.append({
                            "dataset_index": dataset_index,
                            "city": city,
                            "image_id": image_id,
                            "train_id": class_id,
                            "class_name": get_train_id_to_name()[class_id],
                            "component_index": component_index,
                            "original_area_1024x2048": area,
                            "approx_resized_area_div16": area / 16.0,
                        })
                    band = class_boundary(gt)
                    for region, selection in (("boundary", band), ("interior", gt & ~band)):
                        pixels = int(selection.sum())
                        errors = int((selection & (prediction != class_id)).sum())
                        boundary[class_id][region][0] += pixels
                        boundary[class_id][region][1] += errors
            offset += len(targets_np)
    return matrix, city_matrices, image_rows, components, original_components, boundary


def component_tables(components: list[dict], original_components: list[dict], names: dict[int, str]):
    thresholds = {}
    for class_id in FOCUS_CLASSES:
        areas = np.asarray([row["area_256x512"] for row in components if row["train_id"] == class_id], dtype=np.float64)
        if not len(areas):
            raise RuntimeError(f"No resized components for {names[class_id]}")
        thresholds[class_id] = tuple(float(value) for value in np.quantile(areas, (1 / 3, 2 / 3)))
    for row in components:
        low, high = thresholds[row["train_id"]]
        row["small_threshold_max"] = low
        row["medium_threshold_max"] = high
        row["size_bin"] = "small" if row["area_256x512"] <= low else "medium" if row["area_256x512"] <= high else "large"
    summary = []
    for class_id in FOCUS_CLASSES:
        for size_bin in ("small", "medium", "large"):
            selected = [row for row in components if row["train_id"] == class_id and row["size_bin"] == size_bin]
            pixels = sum(row["area_256x512"] for row in selected)
            overlap = sum(row["overlap_pixels"] for row in selected)
            summary.append({
                "train_id": class_id,
                "class_name": names[class_id],
                "size_bin": size_bin,
                "threshold_definition": "per-class observed component-area tertiles at 256x512",
                "small_threshold_max": thresholds[class_id][0],
                "medium_threshold_max": thresholds[class_id][1],
                "component_count": len(selected),
                "gt_pixels": pixels,
                "overlap_pixels": overlap,
                "recall_proxy": overlap / pixels if pixels else math.nan,
            })
    resolution = []
    for class_id in FOCUS_CLASSES:
        selected = [row for row in original_components if row["train_id"] == class_id]
        values = np.asarray([row["approx_resized_area_div16"] for row in selected])
        resolution.append({
            "train_id": class_id,
            "class_name": names[class_id],
            "original_component_count": len(selected),
            "approx_resized_lt_10": int((values < 10).sum()),
            "approx_resized_lt_25": int((values < 25).sum()),
            "approx_resized_lt_50": int((values < 50).sum()),
            "percentage_lt_10": float(100 * (values < 10).mean()),
            "percentage_lt_25": float(100 * (values < 25).mean()),
            "percentage_lt_50": float(100 * (values < 50).mean()),
            "note": "Original 4-connected component area divided by 16; diagnostic approximation, not exact resized topology",
        })
    return summary, resolution, thresholds


def ranked_image_rows(image_rows: list[dict]) -> tuple[list[dict], dict]:
    output = []
    concentration = {}
    for class_id in FOCUS_CLASSES:
        rows = [row for row in image_rows if row["train_id"] == class_id]
        for metric in ("fn_pixels", "fp_pixels", "total_error_pixels"):
            ranked = sorted(rows, key=lambda row: (-row[metric], row["city"], row["image_id"]))
            for rank, row in enumerate(ranked[:10], start=1):
                output.append({"ranking_metric": metric, "rank": rank, **row})
        error_rows = [row for row in rows if row["total_error_pixels"] > 0]
        total = sum(row["total_error_pixels"] for row in error_rows)
        top_ten = sum(sorted((row["total_error_pixels"] for row in error_rows), reverse=True)[:10])
        concentration[class_id] = {
            "images_with_gt": sum(row["gt_pixels"] > 0 for row in rows),
            "images_with_predictions": sum(row["predicted_pixels"] > 0 for row in rows),
            "images_with_error": len(error_rows),
            "top_10_error_share": top_ten / total if total else math.nan,
            "interpretation": "scene_concentrated" if total and top_ten / total >= 0.5 else "broad_or_mixed",
        }
    return output, concentration


def city_rows(city_matrices, names):
    rows = []
    for city in ("jena", "krefeld", "ulm"):
        diagnostics = diagnostic_rows(city_matrices[city], names)
        for class_id in FOCUS_CLASSES:
            row = diagnostics[class_id]
            rows.append({
                "city": city,
                "train_id": class_id,
                "class_name": names[class_id],
                "gt_pixels": row["gt_pixels"],
                "predicted_pixels": row["predicted_pixels"],
                "precision": row["precision"],
                "recall": row["recall"],
                "iou": row["iou"],
            })
    return rows


def boundary_rows(boundary, names):
    rows = []
    for class_id in FOCUS_CLASSES:
        for region in ("boundary", "interior"):
            pixels, errors = boundary[class_id][region]
            rows.append({
                "train_id": class_id,
                "class_name": names[class_id],
                "region": region,
                "inward_boundary_radius_pixels": 2,
                "pixels": pixels,
                "error_pixels": errors,
                "error_rate": errors / pixels if pixels else math.nan,
                "note": "Heuristic GT-class boundary band; not an official metric",
            })
    return rows


def support_rows(diagnostics):
    supports = np.asarray([row["gt_pixels"] for row in diagnostics], dtype=np.float64)
    ious = np.asarray([row["iou"] for row in diagnostics], dtype=np.float64)
    recalls = np.asarray([row["recall"] for row in diagnostics], dtype=np.float64)
    total = supports.sum()
    rows = []
    for row in diagnostics:
        rows.append({
            "train_id": row["train_id"],
            "class_name": row["class_name"],
            "gt_pixels": row["gt_pixels"],
            "gt_frequency": row["gt_pixels"] / total,
            "iou": row["iou"],
            "recall": row["recall"],
        })
    return rows, {"spearman_support_vs_iou": spearman(supports, ious), "spearman_support_vs_recall": spearman(supports, recalls)}


def select_examples(image_rows: list[dict]) -> list[dict]:
    selected = []
    seen = set()
    for class_id in FOCUS_CLASSES:
        rows = [row for row in image_rows if row["train_id"] == class_id and row["gt_pixels"] > 0]
        if not rows:
            continue
        error_sorted = sorted(rows, key=lambda row: (-row["total_error_pixels"], row["dataset_index"]))
        recall_sorted = sorted(rows, key=lambda row: (row["fn_pixels"] / row["gt_pixels"], row["dataset_index"]))
        candidates = [
            ("high_error", error_sorted[0]),
            ("representative", error_sorted[len(error_sorted) // 2]),
            ("good_detection", recall_sorted[0]),
        ]
        for category, row in candidates:
            key = (class_id, row["dataset_index"])
            if key not in seen:
                selected.append({**row, "category": category})
                seen.add(key)
    return selected[:12]


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def overlay(gt: np.ndarray, pred: np.ndarray, class_id: int) -> np.ndarray:
    gt_mask = gt == class_id
    pred_mask = pred == class_id
    result = np.zeros((*gt.shape, 4), dtype=np.uint8)
    result[gt_mask & pred_mask] = (40, 190, 90, 220)
    result[gt_mask & ~pred_mask] = (230, 65, 65, 220)
    result[~gt_mask & pred_mask] = (60, 120, 230, 220)
    return result


def save_examples(model, dataset, selected, device, names):
    palette = np.asarray([label.color for label in get_train_id_classes()], dtype=np.uint8)
    records = []
    with torch.inference_mode():
        for row in selected:
            image, target = dataset[row["dataset_index"]]
            prediction = model(image.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
            target_np = target.numpy()
            figure, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
            axes[0].imshow(denormalize(image)); axes[0].set_title("RGB")
            axes[1].imshow(colorize(target_np, palette)); axes[1].set_title("GT")
            axes[2].imshow(colorize(prediction, palette)); axes[2].set_title("Baseline 03")
            axes[3].imshow(denormalize(image)); axes[3].imshow(overlay(target_np, prediction, row["train_id"]))
            axes[3].set_title(f"{names[row['train_id']]} overlay")
            axes[3].legend(handles=[Patch(color="#28be5a", label="TP"), Patch(color="#e64141", label="FN"), Patch(color="#3c78e6", label="FP")], loc="lower right", fontsize=8)
            for axis in axes: axis.axis("off")
            filename = f"{row['class_name'].replace(' ', '_')}_{row['category']}_{row['city']}_{row['image_id']}.png"
            figure.suptitle(f"{row['class_name']} | {row['category']} | {row['city']} | {row['image_id']}")
            figure.savefig(EXAMPLES_DIR / filename, dpi=140)
            plt.close(figure)
            records.append({"file": filename, **row})
    return records


def save_b02_comparisons(model_b02, model_b03, dataset, selected, device, names):
    palette = np.asarray([label.color for label in get_train_id_classes()], dtype=np.uint8)
    records = []
    with torch.inference_mode():
        for row in selected[:5]:
            image, target = dataset[row["dataset_index"]]
            batch = image.unsqueeze(0).to(device)
            pred02 = model_b02(batch).argmax(1)[0].cpu().numpy()
            pred03 = model_b03(batch).argmax(1)[0].cpu().numpy()
            figure, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
            axes[0].imshow(denormalize(image)); axes[0].set_title("RGB")
            axes[1].imshow(colorize(target.numpy(), palette)); axes[1].set_title("GT")
            axes[2].imshow(colorize(pred02, palette)); axes[2].set_title("Baseline 02")
            axes[3].imshow(colorize(pred03, palette)); axes[3].set_title("Baseline 03")
            for axis in axes: axis.axis("off")
            filename = f"{row['class_name'].replace(' ', '_')}_{row['city']}_{row['image_id']}.png"
            figure.suptitle(f"B02 vs B03 | focus: {names[row['train_id']]}")
            figure.savefig(COMPARE_DIR / filename, dpi=140)
            plt.close(figure)
            records.append({"file": filename, **row})
    return records


def diagnoses(diagnostics, components_summary, boundary_summary, city_summary, concentration, confusions, names):
    output = {}
    for class_id in FOCUS_CLASSES:
        metric = diagnostics[class_id]
        categories = []
        evidence = []
        ratio = metric["prediction_ratio"]
        if ratio < 0.85:
            categories.append("A_underprediction")
            evidence.append(f"prediction/GT={ratio:.3f}")
        elif ratio > 1.15:
            categories.append("B_overprediction")
            evidence.append(f"prediction/GT={ratio:.3f}")
        top_fn = next(row for row in confusions if row["model"] == "baseline_03" and row["train_id"] == class_id and row["direction"] == "fn_destination" and row["rank"] == 1)
        if top_fn["percentage"] >= 20:
            categories.append("C_semantic_confusion")
            evidence.append(f"top FN destination {top_fn['other_class']}={top_fn['percentage']:.1f}% of GT")
        boundary = next(row for row in boundary_summary if row["train_id"] == class_id and row["region"] == "boundary")
        interior = next(row for row in boundary_summary if row["train_id"] == class_id and row["region"] == "interior")
        if boundary["error_rate"] >= interior["error_rate"] + 0.10:
            categories.append("D_boundary_detail_loss")
            evidence.append(f"boundary/interior error={boundary['error_rate']:.3f}/{interior['error_rate']:.3f}")
        small = next(row for row in components_summary if row["train_id"] == class_id and row["size_bin"] == "small")
        large = next(row for row in components_summary if row["train_id"] == class_id and row["size_bin"] == "large")
        if small["recall_proxy"] + 0.10 <= large["recall_proxy"]:
            categories.append("E_small_region_resolution_sensitivity")
            evidence.append(f"small/large component recall={small['recall_proxy']:.3f}/{large['recall_proxy']:.3f}")
        city_values = [row["iou"] for row in city_summary if row["train_id"] == class_id and not math.isnan(row["iou"])]
        if len(city_values) >= 2 and max(city_values) - min(city_values) >= 0.25:
            categories.append("F_city_specific_concentration")
            evidence.append(f"city IoU range={min(city_values):.3f}-{max(city_values):.3f}")
        if not categories:
            categories.append("G_inconclusive")
        output[names[class_id]] = {"categories": categories, "evidence": evidence, **concentration[class_id]}
    return output


def memory_probe(width: int, height: int, batch_size: int, device: torch.device) -> dict:
    split = validate_dev_split(load_dev_split(), PROJECT_ROOT)
    transform = SegmentationTransform(size=(width, height), horizontal_flip_probability=0.0)
    dataset = CityscapesSegmentationDataset(PROJECT_ROOT, split.dev_train_images[:batch_size], transform)
    images, targets = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)))
    model = UNet(in_channels=3, num_classes=19, base_channels=64)
    temporary_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    metadata = load_checkpoint(CHECKPOINT, model=model, optimizer=temporary_optimizer, map_location="cpu")
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    weights = torch.tensor(metadata.config["class_weights"], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=get_ignore_index())
    images, targets = images.to(device), targets.to(device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, targets)
    loss.backward(); optimizer.step(); torch.cuda.synchronize()
    result = {
        "resolution": [width, height], "batch_size": batch_size, "loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss)), "logits_shape": list(logits.shape),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "gpu_total_mib": torch.cuda.get_device_properties(device).total_memory / 1024**2,
    }
    del model, optimizer, images, targets, logits, loss
    torch.cuda.empty_cache()
    write_json(OUTPUT_DIR / f"memory_probe_{width}x{height}_batch{batch_size}.json", result)
    refresh_memory_summary()
    return result


def refresh_memory_summary() -> dict:
    """Summarize completed probes and select the largest comfortable batch-2 resolution."""
    records = [read_json(path) for path in sorted(OUTPUT_DIR.glob("memory_probe_*x*_batch*.json"))]
    if not records:
        raise RuntimeError("No memory-probe results exist")
    for record in records:
        record["reserved_fraction_of_reported_vram"] = record["peak_reserved_mib"] / record["gpu_total_mib"]
    comfortable_batch_two = [
        record for record in records
        if record["batch_size"] == 2
        and record["loss_finite"]
        and record["reserved_fraction_of_reported_vram"] <= 0.80
    ]
    if comfortable_batch_two:
        recommended = max(comfortable_batch_two, key=lambda record: record["resolution"][0] * record["resolution"][1])
        rationale = "Largest tested resolution retaining batch size 2 at no more than 80% of reported VRAM reserved"
    else:
        comfortable_batch_one = [
            record for record in records
            if record["batch_size"] == 1
            and record["loss_finite"]
            and record["reserved_fraction_of_reported_vram"] <= 0.75
        ]
        if not comfortable_batch_one:
            raise RuntimeError("No tested resolution met the documented memory comfort criterion")
        recommended = max(comfortable_batch_one, key=lambda record: record["resolution"][0] * record["resolution"][1])
        rationale = "Largest tested batch-size-1 resolution at no more than 75% of reported VRAM reserved"
    payload = {
        "probes": records,
        "recommended_resolution": recommended["resolution"],
        "recommended_batch_size": recommended["batch_size"],
        "selection_rule": rationale,
        "single_controlled_change": "input resolution only; preserve the Baseline 03 model, loss, optimizer, scheduler, augmentation, seed, epochs, and batch size",
    }
    write_json(OUTPUT_DIR / "memory_probe_summary.json", payload)
    report_path = OUTPUT_DIR / "analysis_summary.md"
    if report_path.exists():
        marker = "\n## Memory feasibility and next experiment\n"
        report = report_path.read_text(encoding="utf-8").split(marker, maxsplit=1)[0].rstrip()
        probe_lines = []
        for record in sorted(records, key=lambda item: (item["resolution"][0], item["batch_size"])):
            probe_lines.append(
                f"- {record['resolution'][0]}x{record['resolution'][1]}, batch {record['batch_size']}: "
                f"{record['peak_allocated_mib']:.1f} MiB allocated, {record['peak_reserved_mib']:.1f} MiB reserved "
                f"({100 * record['reserved_fraction_of_reported_vram']:.1f}% of reported VRAM); finite loss={record['loss_finite']}."
            )
        report += marker + "\n" + "\n".join(probe_lines)
        report += (
            f"\n\nRecommended single Baseline 04 change: increase input resolution to "
            f"{recommended['resolution'][0]}x{recommended['resolution'][1]} while retaining batch size "
            f"{recommended['batch_size']} and all other Baseline 03 settings. No training was started.\n"
        )
        report_path.write_text(report, encoding="utf-8")
    return payload


def analysis_report(summary, diagnostics, confusions, component_summary, resolution_summary, boundary_summary, city_summary, correlation, diagnoses_by_class):
    lines = [
        "# Baseline 03 Residual Failure Analysis",
        "",
        "Development-validation only (jena, krefeld, ulm; 313 images). Official Cityscapes val/test were not loaded.",
        "",
        "## Checkpoint and headline metrics",
        "",
        f"- Epoch: {summary['checkpoint_epoch']}",
        f"- mIoU: {summary['miou']:.6f}",
        f"- Mean Dice: {summary['mean_dice']:.6f}",
        f"- Pixel accuracy: {summary['pixel_accuracy']:.6f}",
        f"- Confusion matrix exact match: {summary['confusion_matrix_exact_match']}",
        "",
        "## Focus classes",
        "",
        "| Class | Precision | Recall | IoU | Dice | Pred/GT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for class_id in FOCUS_CLASSES:
        row = diagnostics[class_id]
        lines.append(f"| {row['class_name']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['iou']:.4f} | {row['dice']:.4f} | {row['prediction_ratio']:.3f}x |")
    lines.extend(["", "## Diagnostic conclusions", ""])
    for class_name, diagnosis in diagnoses_by_class.items():
        lines.append(f"- **{class_name}:** {', '.join(diagnosis['categories'])}. {'; '.join(diagnosis['evidence'])}")
    lines.extend([
        "",
        "## Support relationship",
        "",
        f"- Spearman support vs IoU: {correlation['spearman_support_vs_iou']:.4f}",
        f"- Spearman support vs recall: {correlation['spearman_support_vs_recall']:.4f}",
        "- Correlation is descriptive and does not establish causality.",
        "",
        "## Method limits",
        "",
        "Connected components use 4-connectivity on semantic masks. Size bins are per-class observed tertiles at 256x512. Original-area/16 is a resolution proxy and does not model topology changes from nearest-neighbor resizing. Boundary results use a two-pixel inward band and are not an official Cityscapes metric.",
    ])
    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-probe", action="store_true")
    parser.add_argument("--probe-width", type=int, default=768)
    parser.add_argument("--probe-height", type=int, default=384)
    parser.add_argument("--probe-batch-size", type=int, default=1)
    parser.add_argument("--refresh-memory-summary", action="store_true")
    args = parser.parse_args()
    if args.probe_width % 16 or args.probe_height % 16 or args.probe_width != 2 * args.probe_height:
        raise ValueError("Probe resolution must preserve 2:1 aspect ratio and be divisible by 16")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); EXAMPLES_DIR.mkdir(parents=True, exist_ok=True); COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    if args.refresh_memory_summary:
        print(json.dumps(refresh_memory_summary(), indent=2))
        return
    if args.memory_probe:
        print(json.dumps(memory_probe(args.probe_width, args.probe_height, args.probe_batch_size, device), indent=2))
        return

    started = time.perf_counter()
    names = get_train_id_to_name()
    raw_ids = {label.trainId: label.id for label in get_train_id_classes()}
    expected03 = read_json(EXPERIMENT_DIR / "best_metrics.json")
    expected01 = read_json(PROJECT_ROOT / "outputs/baseline_01_unet_ce/best_metrics.json")
    expected02 = read_json(PROJECT_ROOT / "outputs/baseline_02_weighted_ce/best_metrics.json")
    model03, metadata = checkpoint_model(EXPERIMENT, 30, device)
    if not math.isclose(metadata.best_miou, float(expected03["best_miou"]), abs_tol=1e-12):
        raise RuntimeError("Checkpoint best mIoU differs from saved Baseline 03 metrics")
    split = validate_dev_split(load_dev_split(), PROJECT_ROOT)
    if len(split.dev_val_images) != 313:
        raise RuntimeError(f"Expected 313 dev-validation images; got {len(split.dev_val_images)}")
    official_train = (PROJECT_ROOT / "leftImg8bit/train").resolve()
    if any(not path.resolve().is_relative_to(official_train) for path in split.dev_val_images):
        raise RuntimeError("Development validation contains a path outside official train")
    dataset = CityscapesSegmentationDataset(PROJECT_ROOT, split.dev_val_images, make_val_transform())
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    matrix03, city_matrices, image_rows, components, original_components, boundary = evaluate(model03, dataset, loader, device, raw_ids)
    saved_matrix = np.asarray(expected03["confusion_matrix"], dtype=np.int64)
    if not np.array_equal(matrix03, saved_matrix):
        raise RuntimeError(f"B03 confusion matrix mismatch: absolute difference={np.abs(matrix03 - saved_matrix).sum()}")
    metric03 = metric_summary(matrix03)
    for key, expected_key in (("miou", "best_miou"), ("mean_dice", "mean_dice"), ("pixel_accuracy", "pixel_accuracy")):
        if not math.isclose(metric03[key], float(expected03[expected_key]), rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"Recomputed {key} differs from saved result")
    matrices = {
        "baseline_01": np.asarray(expected01["confusion_matrix"], dtype=np.int64),
        "baseline_02": np.asarray(expected02["confusion_matrix"], dtype=np.int64),
        "baseline_03": matrix03,
    }
    diagnostics_by_model = {name: diagnostic_rows(matrix, names) for name, matrix in matrices.items()}
    focus_rows = []
    for class_id in (*FOCUS_CLASSES, *SECONDARY_CLASSES):
        for model_name in ("baseline_01", "baseline_02", "baseline_03"):
            focus_rows.append({"model": model_name, **diagnostics_by_model[model_name][class_id]})
    confusions = confusion_rows(matrices, names)
    component_summary, resolution_summary, thresholds = component_tables(components, original_components, names)
    ranking_rows, concentration = ranked_image_rows(image_rows)
    cities = city_rows(city_matrices, names)
    boundaries = boundary_rows(boundary, names)
    support, correlations = support_rows(diagnostics_by_model["baseline_03"])
    diagnosis = diagnoses(diagnostics_by_model["baseline_03"], component_summary, boundaries, cities, concentration, confusions, names)
    selected = select_examples(image_rows)
    example_records = save_examples(model03, dataset, selected, device, names)
    model02, _ = checkpoint_model("baseline_02_weighted_ce", 28, device)
    comparison_records = save_b02_comparisons(model02, model03, dataset, selected, device, names)

    write_csv(OUTPUT_DIR / "focus_class_diagnostics.csv", focus_rows)
    write_csv(OUTPUT_DIR / "focus_class_confusions.csv", confusions)
    write_csv(OUTPUT_DIR / "image_failure_ranking.csv", ranking_rows)
    write_csv(OUTPUT_DIR / "component_size_diagnostic.csv", component_summary)
    write_csv(OUTPUT_DIR / "resolution_loss_proxy.csv", resolution_summary)
    write_csv(OUTPUT_DIR / "boundary_diagnostic.csv", boundaries)
    write_csv(OUTPUT_DIR / "city_focus_metrics.csv", cities)
    write_csv(OUTPUT_DIR / "support_vs_performance.csv", support)
    write_json(OUTPUT_DIR / "diagnostic_summary.json", {
        "checkpoint_epoch": metadata.epoch,
        "checkpoint_best_miou": metadata.best_miou,
        "miou": metric03["miou"], "mean_dice": metric03["mean_dice"], "pixel_accuracy": metric03["pixel_accuracy"],
        "confusion_matrix_exact_match": True, "dev_val_images": len(dataset),
        "component_thresholds": {names[key]: value for key, value in thresholds.items()},
        "image_error_concentration": {names[key]: value for key, value in concentration.items()},
        "support_correlations": correlations, "diagnoses": diagnosis,
        "examples": example_records, "b02_vs_b03_examples": comparison_records,
        "runtime_seconds": time.perf_counter() - started,
        "official_val_test_loaded": False,
    })
    report_summary = {"checkpoint_epoch": metadata.epoch, **metric03, "confusion_matrix_exact_match": True}
    analysis_report(report_summary, diagnostics_by_model["baseline_03"], confusions, component_summary, resolution_summary, boundaries, cities, correlations, diagnosis)
    print(json.dumps({
        "complete": True, "dev_val_images": len(dataset), "miou": metric03["miou"],
        "mean_dice": metric03["mean_dice"], "pixel_accuracy": metric03["pixel_accuracy"],
        "runtime_seconds": time.perf_counter() - started, "output_dir": str(OUTPUT_DIR),
    }, indent=2))


if __name__ == "__main__":
    main()
