"""Residual analysis of Baseline 04 versus Baseline 03 on dev validation only."""

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
from analyze_baseline_03 import class_boundary, component_stats, denormalize, overlay, ranks  # noqa: E402
from cityscapes_segmentation.data.dataset import CityscapesSegmentationDataset  # noqa: E402
from cityscapes_segmentation.data.labels import (  # noqa: E402
    get_ignore_index,
    get_train_id_classes,
    get_train_id_to_name,
    map_raw_ids_to_train_ids,
)
from cityscapes_segmentation.data.splits import load_dev_split, validate_dev_split  # noqa: E402
from cityscapes_segmentation.data.transforms import SegmentationTransform  # noqa: E402
from cityscapes_segmentation.engine import load_checkpoint  # noqa: E402
from cityscapes_segmentation.models import UNet  # noqa: E402


NUM_CLASSES = 19
B03 = "baseline_03_mild_weighted_ce"
B04 = "baseline_04_higher_resolution"
B03_DIR = PROJECT_ROOT / "outputs" / B03
B04_DIR = PROJECT_ROOT / "outputs" / B04
OUTPUT_DIR = B04_DIR / "error_analysis"
TRUCK_EXAMPLES_DIR = OUTPUT_DIR / "truck_examples"
LIGHT_EXAMPLES_DIR = OUTPUT_DIR / "traffic_light_examples"
FOCUS_CLASSES = (14, 16, 15, 6)  # truck, train, bus, traffic light
THIN_CLASSES = (5, 6, 7)  # pole, traffic light, traffic sign


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    def convert(value: str):
        if value == "":
            return None
        if value.lower() == "nan":
            return None
        try:
            number = float(value)
            return int(number) if number.is_integer() else number
        except ValueError:
            return value

    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: convert(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def matrix_for(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    add_matrix(matrix, target, prediction)
    return matrix


def class_statistics(matrix: np.ndarray, class_id: int) -> dict:
    tp = int(matrix[class_id, class_id])
    gt = int(matrix[class_id].sum())
    predicted = int(matrix[:, class_id].sum())
    fp = predicted - tp
    fn = gt - tp
    return {
        "gt_pixels": gt,
        "predicted_pixels": predicted,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": safe_divide(tp, predicted),
        "recall": safe_divide(tp, gt),
        "iou": safe_divide(tp, tp + fp + fn),
        "dice": safe_divide(2 * tp, 2 * tp + fp + fn),
        "prediction_ratio": safe_divide(predicted, gt),
    }


def verify_and_load_model(experiment: str, expected_epoch: int, device: torch.device):
    config = read_json(PROJECT_ROOT / "outputs" / experiment / "config.json")
    model = UNet(in_channels=3, num_classes=19, base_channels=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    metadata = load_checkpoint(
        PROJECT_ROOT / "checkpoints" / experiment / "best.pt",
        model=model,
        optimizer=optimizer,
        map_location=device,
    )
    expected_model = {"model": "UNet", "in_channels": 3, "num_classes": 19, "base_channels": 64}
    mismatches = {
        key: (metadata.config.get(key), value)
        for key, value in expected_model.items()
        if metadata.config.get(key) != value
    }
    if metadata.epoch != expected_epoch or mismatches:
        raise RuntimeError(f"{experiment} checkpoint identity mismatch: epoch={metadata.epoch}, {mismatches}")
    if metadata.config.get("experiment_name") != experiment:
        raise RuntimeError(f"{experiment} checkpoint stores wrong experiment identity")
    model.eval()
    return model, metadata, config


def infer_native(model, dataset, device: torch.device):
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    image_matrices: list[np.ndarray] = []
    with torch.inference_mode():
        for images, batch_targets in loader:
            batch_predictions = model(images.to(device)).argmax(1).cpu().numpy().astype(np.uint8)
            batch_targets_np = batch_targets.numpy().astype(np.uint8)
            for target, prediction in zip(batch_targets_np, batch_predictions, strict=True):
                item_matrix = matrix_for(target, prediction)
                matrix += item_matrix
                predictions.append(prediction)
                targets.append(target)
                image_matrices.append(item_matrix)
    return matrix, predictions, targets, np.stack(image_matrices)


def resize_labels(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(values).resize(size, resample=Image.Resampling.NEAREST))


def normalized_confusion_rows(matrix03: np.ndarray, matrix04: np.ndarray, names: dict[int, str]):
    row03 = matrix03 / np.maximum(matrix03.sum(1, keepdims=True), 1)
    row04 = matrix04 / np.maximum(matrix04.sum(1, keepdims=True), 1)
    col03 = matrix03 / np.maximum(matrix03.sum(0, keepdims=True), 1)
    col04 = matrix04 / np.maximum(matrix04.sum(0, keepdims=True), 1)
    rows = []
    for target_id in range(NUM_CLASSES):
        for predicted_id in range(NUM_CLASSES):
            rows.append({
                "target_train_id": target_id,
                "target_class": names[target_id],
                "predicted_train_id": predicted_id,
                "predicted_class": names[predicted_id],
                "b03_gt_row_proportion": row03[target_id, predicted_id],
                "b04_gt_row_proportion": row04[target_id, predicted_id],
                "delta_gt_row_proportion": row04[target_id, predicted_id] - row03[target_id, predicted_id],
                "b03_prediction_column_source_proportion": col03[target_id, predicted_id],
                "b04_prediction_column_source_proportion": col04[target_id, predicted_id],
                "delta_prediction_column_source_proportion": col04[target_id, predicted_id] - col03[target_id, predicted_id],
            })
    return rows


def semantic_flow_rows(matrix03: np.ndarray, matrix04: np.ndarray, names: dict[int, str]):
    output = []
    for class_id in range(NUM_CLASSES):
        gt03 = int(matrix03[class_id].sum())
        gt04 = int(matrix04[class_id].sum())
        fp03 = int(matrix03[:, class_id].sum() - matrix03[class_id, class_id])
        fp04 = int(matrix04[:, class_id].sum() - matrix04[class_id, class_id])
        for other in range(NUM_CLASSES):
            if other == class_id:
                continue
            fn03 = int(matrix03[class_id, other])
            fn04 = int(matrix04[class_id, other])
            source03 = int(matrix03[other, class_id])
            source04 = int(matrix04[other, class_id])
            output.extend((
                {
                    "focus_train_id": class_id, "focus_class": names[class_id],
                    "direction": "fn_destination", "other_train_id": other, "other_class": names[other],
                    "b03_proportion": safe_divide(fn03, gt03), "b04_proportion": safe_divide(fn04, gt04),
                    "delta_proportion": safe_divide(fn04, gt04) - safe_divide(fn03, gt03),
                },
                {
                    "focus_train_id": class_id, "focus_class": names[class_id],
                    "direction": "fp_source", "other_train_id": other, "other_class": names[other],
                    "b03_proportion": safe_divide(source03, fp03), "b04_proportion": safe_divide(source04, fp04),
                    "delta_proportion": safe_divide(source04, fp04) - safe_divide(source03, fp03),
                },
            ))
    return output


def per_image_original_grid(dataset, predictions03, predictions04):
    records = []
    matrices03 = []
    matrices04 = []
    city03 = defaultdict(lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64))
    city04 = defaultdict(lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64))
    traffic_components = []
    traffic_boundary = {model: {region: [0, 0] for region in ("boundary", "interior")} for model in (B03, B04)}
    for index, ((image_path, mask_path), pred03_native, pred04_native) in enumerate(
        zip(dataset.samples, predictions03, predictions04, strict=True)
    ):
        with Image.open(mask_path) as source:
            raw = np.asarray(source)
        target = map_raw_ids_to_train_ids(raw)
        height, width = target.shape
        pred03 = resize_labels(pred03_native, (width, height))
        pred04 = resize_labels(pred04_native, (width, height))
        matrix03 = matrix_for(target, pred03)
        matrix04 = matrix_for(target, pred04)
        matrices03.append(matrix03)
        matrices04.append(matrix04)
        city = image_path.parent.name
        city03[city] += matrix03
        city04[city] += matrix04
        base = {
            "dataset_index": index,
            "city": city,
            "image_id": image_path.name.removesuffix("_leftImg8bit.png"),
            "image_path": str(image_path),
        }
        for class_id in FOCUS_CLASSES:
            stats03 = class_statistics(matrix03, class_id)
            stats04 = class_statistics(matrix04, class_id)
            if stats03["gt_pixels"] or stats03["predicted_pixels"] or stats04["predicted_pixels"]:
                row = {**base, "train_id": class_id}
                for key, value in stats03.items():
                    row[f"b03_{key}"] = value
                for key, value in stats04.items():
                    row[f"b04_{key}"] = value
                for metric in ("iou", "precision", "recall"):
                    left, right = stats03[metric], stats04[metric]
                    row[f"delta_{metric}"] = right - left if math.isfinite(left) and math.isfinite(right) else math.nan
                row["fn_reduction"] = stats03["false_negatives"] - stats04["false_negatives"]
                row["fp_reduction"] = stats03["false_positives"] - stats04["false_positives"]
                row["error_reduction"] = (
                    stats03["false_negatives"] + stats03["false_positives"]
                    - stats04["false_negatives"] - stats04["false_positives"]
                )
                records.append(row)

        light_gt = target == 6
        light_predictions = {B03: pred03 == 6, B04: pred04 == 6}
        for component_index, indices in enumerate(connected_components(light_gt), start=1):
            component_mask = np.zeros_like(light_gt)
            component_mask.flat[indices] = True
            rows, columns = np.where(component_mask)
            pad = 16
            r0, r1 = max(0, rows.min() - pad), min(height, rows.max() + pad + 1)
            c0, c1 = max(0, columns.min() - pad), min(width, columns.max() + pad + 1)
            component = {
                **base, "component_index": component_index, "area_original": len(indices),
            }
            for model_name, prediction_mask in light_predictions.items():
                overlap = int((component_mask & prediction_mask).sum())
                predicted_local = int(prediction_mask[r0:r1, c0:c1].sum())
                union_local = len(indices) + predicted_local - overlap
                component[f"{model_name}_overlap"] = overlap
                component[f"{model_name}_recall"] = overlap / len(indices)
                component[f"{model_name}_local_iou"] = safe_divide(overlap, union_local)
                component[f"{model_name}_missed_whole"] = int(overlap == 0)
            traffic_components.append(component)
        band = class_boundary(light_gt, radius=8)
        for model_name, prediction_mask in light_predictions.items():
            for region, selection in (("boundary", band), ("interior", light_gt & ~band)):
                pixels = int(selection.sum())
                errors = int((selection & ~prediction_mask).sum())
                traffic_boundary[model_name][region][0] += pixels
                traffic_boundary[model_name][region][1] += errors
    return (
        records,
        np.stack(matrices03),
        np.stack(matrices04),
        city03,
        city04,
        traffic_components,
        traffic_boundary,
    )


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    remaining = set(np.flatnonzero(mask).tolist())
    output = []
    while remaining:
        first = remaining.pop()
        stack = [first]
        component = []
        while stack:
            index = stack.pop()
            component.append(index)
            row, column = divmod(index, width)
            for candidate, allowed in (
                (index - width, row > 0),
                (index + width, row + 1 < height),
                (index - 1, column > 0),
                (index + 1, column + 1 < width),
            ):
                if allowed and candidate in remaining:
                    remaining.remove(candidate)
                    stack.append(candidate)
        output.append(np.asarray(component, dtype=np.int64))
    return output


def truck_tables(records: list[dict], names: dict[int, str]):
    rows = [{**row, "class_name": names[14]} for row in records if row["train_id"] == 14]
    ranked = sorted(
        [row for row in rows if math.isfinite(row["delta_iou"])],
        key=lambda row: (row["delta_iou"], row["city"], row["image_id"]),
    )
    for rank, row in enumerate(ranked, start=1):
        row["regression_rank"] = rank
    return rows, ranked


def truck_concentration(rows: list[dict]) -> dict:
    increases = sorted((max(0, -row["error_reduction"]) for row in rows), reverse=True)
    gross = sum(increases)
    city_increases = {
        city: sum(max(0, -row["error_reduction"]) for row in rows if row["city"] == city)
        for city in ("jena", "krefeld", "ulm")
    }
    result = {
        "images_with_gt": sum(row["b03_gt_pixels"] > 0 for row in rows),
        "images_with_either_prediction": sum(
            row["b03_predicted_pixels"] > 0 or row["b04_predicted_pixels"] > 0 for row in rows
        ),
        "net_error_increase": -sum(row["error_reduction"] for row in rows),
        "gross_positive_error_increase": gross,
        **{f"{city}_gross_error_increase": value for city, value in city_increases.items()},
    }
    for count in (1, 3, 5):
        result[f"top_{count}_share_of_gross_error_increase"] = safe_divide(sum(increases[:count]), gross)
    return result


def city_tables(city03, city04, records, names):
    focus_rows = []
    aggregate_rows = []
    for city in ("jena", "krefeld", "ulm"):
        summary03 = metric_summary(city03[city])
        summary04 = metric_summary(city04[city])
        aggregate_rows.append({
            "city": city,
            "b03_miou_original_grid": summary03["miou"],
            "b04_miou_original_grid": summary04["miou"],
            "delta_miou": summary04["miou"] - summary03["miou"],
            "b03_mean_dice": summary03["mean_dice"],
            "b04_mean_dice": summary04["mean_dice"],
            "delta_mean_dice": summary04["mean_dice"] - summary03["mean_dice"],
        })
        for class_id in (16, 15):
            stat03 = class_statistics(city03[city], class_id)
            stat04 = class_statistics(city04[city], class_id)
            class_records = [
                row for row in records
                if row["city"] == city and row["train_id"] == class_id and row["b03_gt_pixels"] > 0
            ]
            row = {
                "city": city,
                "train_id": class_id,
                "class_name": names[class_id],
                "images_containing_class": len(class_records),
            }
            for key, value in stat03.items():
                row[f"b03_{key}"] = value
            for key, value in stat04.items():
                row[f"b04_{key}"] = value
            for metric in ("precision", "recall", "iou", "prediction_ratio"):
                row[f"delta_{metric}"] = stat04[metric] - stat03[metric]
            focus_rows.append(row)
    return focus_rows, aggregate_rows


def concentration_tables(records: list[dict], names: dict[int, str]):
    rankings = []
    summaries = []
    for class_id in (16, 15):
        rows = [row for row in records if row["train_id"] == class_id]
        for metric, descending in (("delta_iou", True), ("fn_reduction", True), ("fp_reduction", True)):
            valid = [row for row in rows if math.isfinite(row[metric])]
            ordered = sorted(valid, key=lambda row: (-row[metric] if descending else row[metric], row["city"], row["image_id"]))
            for rank, row in enumerate(ordered[:10], start=1):
                rankings.append({"class_name": names[class_id], "ranking_metric": metric, "rank": rank, **row})
            regressions = sorted(valid, key=lambda row: (row[metric], row["city"], row["image_id"]))
            for rank, row in enumerate(regressions[:10], start=1):
                rankings.append({"class_name": names[class_id], "ranking_metric": f"{metric}_regression", "rank": rank, **row})
        positive = sorted((max(0, row["error_reduction"]) for row in rows), reverse=True)
        gross = sum(positive)
        net = sum(row["error_reduction"] for row in rows)
        summary = {"train_id": class_id, "class_name": names[class_id], "net_error_reduction": net, "gross_positive_error_reduction": gross}
        for count in (1, 3, 5):
            summary[f"top_{count}_share_of_gross_positive_reduction"] = safe_divide(sum(positive[:count]), gross)
        summaries.append(summary)
    return rankings, summaries


def traffic_component_tables(components: list[dict], boundary: dict):
    areas = np.asarray([row["area_original"] for row in components], dtype=np.float64)
    low, high = np.quantile(areas, (1 / 3, 2 / 3))
    for row in components:
        area = row["area_original"]
        row["size_bin"] = "small" if area <= low else "medium" if area <= high else "large"
        row["small_threshold_max_original_pixels"] = low
        row["medium_threshold_max_original_pixels"] = high
    summaries = []
    for size_bin in ("small", "medium", "large"):
        selected = [row for row in components if row["size_bin"] == size_bin]
        pixels = sum(row["area_original"] for row in selected)
        summary = {"size_bin": size_bin, "component_count": len(selected), "gt_pixels_original": pixels}
        for model_name in (B03, B04):
            overlap = sum(row[f"{model_name}_overlap"] for row in selected)
            summary[f"{model_name}_recall"] = overlap / pixels
            summary[f"{model_name}_mean_local_iou"] = float(np.mean([row[f"{model_name}_local_iou"] for row in selected]))
            summary[f"{model_name}_whole_miss_rate"] = float(np.mean([row[f"{model_name}_missed_whole"] for row in selected]))
        for metric in ("recall", "mean_local_iou", "whole_miss_rate"):
            summary[f"delta_{metric}"] = summary[f"{B04}_{metric}"] - summary[f"{B03}_{metric}"]
        summaries.append(summary)
    boundary_rows = []
    for region in ("boundary", "interior"):
        row = {"region": region, "original_grid_boundary_radius_pixels": 8}
        for model_name in (B03, B04):
            pixels, errors = boundary[model_name][region]
            row[f"{model_name}_pixels"] = pixels
            row[f"{model_name}_errors"] = errors
            row[f"{model_name}_error_rate"] = safe_divide(errors, pixels)
        row["delta_error_rate"] = row[f"{B04}_error_rate"] - row[f"{B03}_error_rate"]
        boundary_rows.append(row)
    return summaries, boundary_rows


def class_delta_tables(matrix03: np.ndarray, matrix04: np.ndarray, names: dict[int, str]):
    diag03 = diagnostic_rows(matrix03, names)
    diag04 = diagnostic_rows(matrix04, names)
    support = np.asarray([row["gt_pixels"] for row in diag03], dtype=np.float64)
    deltas = np.asarray([diag04[index]["iou"] - diag03[index]["iou"] for index in range(NUM_CLASSES)])
    rows = []
    for class_id in range(NUM_CLASSES):
        delta = deltas[class_id]
        if delta > 0.03:
            group = "strong_gain"
        elif delta >= 0.01:
            group = "moderate_gain"
        elif delta >= -0.01:
            group = "neutral"
        elif delta >= -0.03:
            group = "moderate_regression"
        else:
            group = "strong_regression"
        delta_precision = diag04[class_id]["precision"] - diag03[class_id]["precision"]
        delta_recall = diag04[class_id]["recall"] - diag03[class_id]["recall"]
        if abs(delta_precision) >= 2 * abs(delta_recall):
            driver = "primarily_precision"
        elif abs(delta_recall) >= 2 * abs(delta_precision):
            driver = "primarily_recall"
        else:
            driver = "both_or_mixed"
        rows.append({
            "train_id": class_id, "class_name": names[class_id], "b03_gt_pixels": diag03[class_id]["gt_pixels"],
            "b03_precision": diag03[class_id]["precision"], "b04_precision": diag04[class_id]["precision"],
            "delta_precision": delta_precision, "b03_recall": diag03[class_id]["recall"],
            "b04_recall": diag04[class_id]["recall"], "delta_recall": delta_recall,
            "b03_iou": diag03[class_id]["iou"], "b04_iou": diag04[class_id]["iou"],
            "delta_iou": delta, "delta_group": group, "precision_recall_driver": driver,
        })
    support_ranks = ranks(support)
    delta_ranks = ranks(np.abs(deltas))
    correlation = float(np.corrcoef(support_ranks, delta_ranks)[0, 1])
    return rows, correlation


def bootstrap_diagnostics(image_matrices03: np.ndarray, image_matrices04: np.ndarray, samples: int = 1000):
    rng = np.random.default_rng(42)
    names = get_train_id_to_name()
    requested = {"overall_miou": None, "truck_iou": 14, "train_iou": 16, "bus_iou": 15, "traffic_light_iou": 6}
    values = {key: [] for key in requested}
    image_count = len(image_matrices03)
    for _ in range(samples):
        counts = np.bincount(rng.integers(0, image_count, image_count), minlength=image_count)
        matrix03 = np.tensordot(counts, image_matrices03, axes=(0, 0))
        matrix04 = np.tensordot(counts, image_matrices04, axes=(0, 0))
        values["overall_miou"].append(metric_summary(matrix04)["miou"] - metric_summary(matrix03)["miou"])
        for key, class_id in requested.items():
            if class_id is not None:
                values[key].append(class_statistics(matrix04, class_id)["iou"] - class_statistics(matrix03, class_id)["iou"])
    rows = []
    for key, observations in values.items():
        array = np.asarray(observations, dtype=np.float64)
        rows.append({
            "metric": key,
            "class_name": "all_classes_macro" if requested[key] is None else names[requested[key]],
            "bootstrap_resamples": samples,
            "seed": 42,
            "mean_delta": float(array.mean()),
            "median_delta": float(np.median(array)),
            "percentile_2_5": float(np.quantile(array, 0.025)),
            "percentile_97_5": float(np.quantile(array, 0.975)),
            "fraction_delta_gt_zero": float((array > 0).mean()),
            "note": "Image bootstrap diagnostic; not a formal benchmark confidence interval",
        })
    return rows


def select_diverse(rows: list[dict], count: int) -> list[dict]:
    selected = []
    seen = set()
    for city in ("jena", "krefeld", "ulm"):
        candidate = next((row for row in rows if row["city"] == city and row["dataset_index"] not in seen), None)
        if candidate:
            selected.append(candidate)
            seen.add(candidate["dataset_index"])
    for row in rows:
        if len(selected) >= count:
            break
        if row["dataset_index"] not in seen:
            selected.append(row)
            seen.add(row["dataset_index"])
    return selected[:count]


def save_comparison_panels(dataset, predictions03, predictions04, selections, class_id, directory, names):
    palette = np.asarray([label.color for label in get_train_id_classes()], dtype=np.uint8)
    records = []
    for position, row in enumerate(selections, start=1):
        image, target04 = dataset[row["dataset_index"]]
        target = target04.numpy()
        pred03 = resize_labels(predictions03[row["dataset_index"]], (target.shape[1], target.shape[0]))
        pred04 = predictions04[row["dataset_index"]]
        figure, axes = plt.subplots(1, 6, figsize=(24, 4.5), constrained_layout=True)
        rgb = denormalize(image)
        axes[0].imshow(rgb); axes[0].set_title("RGB")
        axes[1].imshow(colorize(target, palette)); axes[1].set_title("GT")
        axes[2].imshow(colorize(pred03, palette)); axes[2].set_title("Baseline 03")
        axes[3].imshow(colorize(pred04, palette)); axes[3].set_title("Baseline 04")
        axes[4].imshow(rgb); axes[4].imshow(overlay(target, pred03, class_id)); axes[4].set_title("B03 class overlay")
        axes[5].imshow(rgb); axes[5].imshow(overlay(target, pred04, class_id)); axes[5].set_title("B04 class overlay")
        handles = [Patch(color="#28be5a", label="TP"), Patch(color="#e64141", label="FN"), Patch(color="#3c78e6", label="FP")]
        axes[5].legend(handles=handles, loc="lower right", fontsize=7)
        for axis in axes:
            axis.axis("off")
        filename = f"{position:02d}_{names[class_id].replace(' ', '_')}_{row['city']}_{row['image_id']}.png"
        figure.suptitle(f"{names[class_id]} comparison | {row['city']}/{row['image_id']}")
        figure.savefig(directory / filename, dpi=130)
        plt.close(figure)
        records.append({"file": filename, **row})
    return records


def traffic_visual_selections(records: list[dict], predictions03, predictions04, dataset):
    rows = [row for row in records if row["train_id"] == 6 and math.isfinite(row["delta_iou"])]
    lower = sorted(rows, key=lambda row: (row["delta_iou"], row["dataset_index"]))
    higher = sorted(rows, key=lambda row: (-row["delta_iou"], row["dataset_index"]))
    good = sorted(rows, key=lambda row: (-row["b04_iou"], row["dataset_index"]))
    thin_confusion = []
    for row in rows:
        index = row["dataset_index"]
        target = dataset[index][1].numpy()
        pred04 = predictions04[index]
        confused = int((((target == 5) | (target == 7)) & (pred04 == 6)).sum())
        thin_confusion.append((confused, row))
    thin_confusion.sort(key=lambda item: (-item[0], item[1]["dataset_index"]))
    candidates = [*lower[:2], *higher[:2], *good[:1], *(row for _, row in thin_confusion[:2])]
    selected = []
    seen = set()
    for row in candidates:
        if row["dataset_index"] not in seen:
            selected.append(row)
            seen.add(row["dataset_index"])
    return selected[:6]


def report_markdown(summary: dict, names: dict[int, str]) -> None:
    truck = summary["truck"]
    lines = [
        "# Baseline 04 Residual Analysis",
        "",
        "Development validation only: 313 images from jena, krefeld, and ulm. Official Cityscapes val/test were not loaded.",
        "",
        "## Verification",
        "",
        f"- B04 checkpoint epoch: {summary['checkpoint']['epoch']}",
        f"- mIoU: {summary['headline']['miou']:.6f}",
        f"- Mean Dice: {summary['headline']['mean_dice']:.6f}",
        f"- Pixel accuracy: {summary['headline']['pixel_accuracy']:.6f}",
        "- Both native confusion matrices exactly match their saved experiment artifacts.",
        "",
        "## Truck regression",
        "",
        f"Truck IoU fell from {truck['b03']['iou']:.4f} to {truck['b04']['iou']:.4f}. "
        f"Precision changed {truck['b03']['precision']:.4f} -> {truck['b04']['precision']:.4f}; "
        f"recall changed {truck['b03']['recall']:.4f} -> {truck['b04']['recall']:.4f}. "
        "The regression is therefore driven by both precision and recall, not prediction volume alone.",
        "",
        f"- Prediction/GT ratio: {truck['b03']['prediction_ratio']:.3f}x -> {truck['b04']['prediction_ratio']:.3f}x.",
        f"- Top 1/3/5 scenes explain {summary['truck_concentration']['top_1_share_of_gross_error_increase']:.1%} / "
        f"{summary['truck_concentration']['top_3_share_of_gross_error_increase']:.1%} / "
        f"{summary['truck_concentration']['top_5_share_of_gross_error_increase']:.1%} of gross truck error growth.",
        "",
        "## Train and bus concentration",
        "",
    ]
    for row in summary["scene_concentration"]:
        lines.append(
            f"- **{row['class_name']}:** top 1/3/5 improving scenes explain "
            f"{row['top_1_share_of_gross_positive_reduction']:.1%} / "
            f"{row['top_3_share_of_gross_positive_reduction']:.1%} / "
            f"{row['top_5_share_of_gross_positive_reduction']:.1%} of gross positive error reduction."
        )
    lines.extend([
        "",
        "## Traffic-light localization",
        "",
    ])
    for row in summary["traffic_light_components"]:
        lines.append(
            f"- **{row['size_bin']}:** recall {row[f'{B03}_recall']:.3f} -> {row[f'{B04}_recall']:.3f}; "
            f"local IoU {row[f'{B03}_mean_local_iou']:.3f} -> {row[f'{B04}_mean_local_iou']:.3f}; "
            f"whole-miss rate {row[f'{B03}_whole_miss_rate']:.3f} -> {row[f'{B04}_whole_miss_rate']:.3f}."
        )
    lines.extend([
        "",
        "## Class-delta groups",
        "",
    ])
    for group, classes in summary["class_groups"].items():
        lines.append(f"- **{group.replace('_', ' ')}:** {', '.join(classes) if classes else 'none'}")
    lines.extend([
        "",
        "## Image-bootstrap diagnostic",
        "",
        "These are deterministic image-resampling diagnostics, not formal benchmark confidence intervals.",
        "",
        "| Metric | Mean delta | 95% percentile interval | P(delta > 0) |",
        "|---|---:|---:|---:|",
    ])
    for row in summary["bootstrap"]:
        lines.append(
            f"| {row['metric']} | {row['mean_delta']:+.4f} | "
            f"[{row['percentile_2_5']:+.4f}, {row['percentile_97_5']:+.4f}] | "
            f"{row['fraction_delta_gt_zero']:.3f} |"
        )
    lines.extend([
        "",
        "## Main conclusion",
        "",
        summary["dominant_pattern"],
        "",
        "## Recommended single Baseline 05 change",
        "",
        summary["recommendation"],
        "",
        "## Method limits",
        "",
        "Native-resolution confusion matrices reproduce each experiment exactly. Per-scene count comparisons and component diagnostics use predictions nearest-neighbor upsampled to the common original 2048x1024 label grid. Component and boundary measures are diagnostic proxies, not official Cityscapes instance or boundary metrics. Bootstrap intervals are diagnostic image-resampling intervals, not formal benchmark confidence intervals.",
    ])
    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def enrich_summary_from_outputs(summary: dict) -> dict:
    class_rows = read_csv(OUTPUT_DIR / "class_delta_decomposition.csv")
    truck_rows = read_csv(OUTPUT_DIR / "truck_diagnostics.csv")
    truck_summary = truck_concentration(truck_rows)
    summary["truck_concentration"] = truck_summary
    write_csv(OUTPUT_DIR / "truck_concentration_summary.csv", [truck_summary])
    summary["class_groups"] = {
        group: [row["class_name"] for row in class_rows if row["delta_group"] == group]
        for group in (
            "strong_gain", "moderate_gain", "neutral", "moderate_regression", "strong_regression"
        )
    }
    summary["class_delta_decomposition"] = class_rows
    summary["city_aggregate"] = read_csv(OUTPUT_DIR / "city_aggregate_comparison.csv")
    summary["train_bus_city_diagnostics"] = read_csv(OUTPUT_DIR / "train_bus_city_diagnostics.csv")
    summary["scene_concentration"] = read_csv(OUTPUT_DIR / "train_bus_concentration_summary.csv")
    flows = read_csv(OUTPUT_DIR / "semantic_flow_changes.csv")
    summary["truck_semantic_flows"] = [row for row in flows if row["focus_class"] == "truck"]
    summary["train_bus_semantic_flows"] = [row for row in flows if row["focus_class"] in {"train", "bus"}]
    summary["traffic_light_semantic_flows"] = [row for row in flows if row["focus_class"] == "traffic light"]
    summary["thin_object_confusions"] = read_csv(OUTPUT_DIR / "thin_object_confusions.csv")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--refresh-report", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.perf_counter()
    for directory in (OUTPUT_DIR, TRUCK_EXAMPLES_DIR, LIGHT_EXAMPLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    names = get_train_id_to_name()
    if args.refresh_report:
        summary = enrich_summary_from_outputs(read_json(OUTPUT_DIR / "diagnostic_summary.json"))
        write_json(OUTPUT_DIR / "diagnostic_summary.json", summary)
        report_markdown(summary, names)
        print(json.dumps({"report_refreshed": True, "output_dir": str(OUTPUT_DIR)}, indent=2))
        return
    model03, metadata03, config03 = verify_and_load_model(B03, 30, device)
    model04, metadata04, config04 = verify_and_load_model(B04, 30, device)
    expected03 = read_json(B03_DIR / "best_metrics.json")
    expected04 = read_json(B04_DIR / "best_metrics.json")
    if config04["resolution"] != {"width": 640, "height": 320}:
        raise RuntimeError("Baseline 04 resolution differs from 640x320")
    if config03["class_weights"] != config04["class_weights"]:
        raise RuntimeError("Baseline 04 did not preserve the exact Baseline 03 class weights")
    if not math.isclose(metadata04.best_miou, expected04["best_miou"], abs_tol=1e-12):
        raise RuntimeError("Baseline 04 checkpoint best mIoU differs from its saved result")

    split = validate_dev_split(load_dev_split(), PROJECT_ROOT)
    if len(split.dev_val_images) != 313:
        raise RuntimeError(f"Expected 313 dev-validation images; got {len(split.dev_val_images)}")
    official_train = (PROJECT_ROOT / "leftImg8bit" / "train").resolve()
    if any(not path.resolve().is_relative_to(official_train) for path in split.dev_val_images):
        raise RuntimeError("Development validation contains a path outside official train")
    dataset03 = CityscapesSegmentationDataset(
        PROJECT_ROOT, split.dev_val_images, SegmentationTransform(size=(512, 256))
    )
    dataset04 = CityscapesSegmentationDataset(
        PROJECT_ROOT, split.dev_val_images, SegmentationTransform(size=(640, 320))
    )
    matrix03, predictions03, _, native_image_matrices03 = infer_native(model03, dataset03, device)
    matrix04, predictions04, _, native_image_matrices04 = infer_native(model04, dataset04, device)
    saved03 = np.asarray(expected03["confusion_matrix"], dtype=np.int64)
    saved04 = np.asarray(expected04["confusion_matrix"], dtype=np.int64)
    if not np.array_equal(matrix03, saved03):
        raise RuntimeError(f"B03 confusion mismatch: absolute difference={np.abs(matrix03 - saved03).sum()}")
    if not np.array_equal(matrix04, saved04):
        raise RuntimeError(f"B04 confusion mismatch: absolute difference={np.abs(matrix04 - saved04).sum()}")
    recomputed04 = metric_summary(matrix04)
    for key, saved_key in (("miou", "best_miou"), ("mean_dice", "mean_dice"), ("pixel_accuracy", "pixel_accuracy")):
        if not math.isclose(recomputed04[key], expected04[saved_key], abs_tol=1e-12):
            raise RuntimeError(f"Recomputed B04 {key} does not match saved result")

    normalized_rows = normalized_confusion_rows(matrix03, matrix04, names)
    flows = semantic_flow_rows(matrix03, matrix04, names)
    (
        image_records,
        original_image_matrices03,
        original_image_matrices04,
        city03,
        city04,
        light_components,
        light_boundary,
    ) = per_image_original_grid(dataset04, predictions03, predictions04)
    truck_rows, truck_ranked = truck_tables(image_records, names)
    city_focus, city_aggregate = city_tables(city03, city04, image_records, names)
    scene_rankings, scene_concentration = concentration_tables(image_records, names)
    light_component_summary, light_boundary_rows = traffic_component_tables(light_components, light_boundary)
    class_deltas, support_delta_correlation = class_delta_tables(matrix03, matrix04, names)
    bootstrap = bootstrap_diagnostics(native_image_matrices03, native_image_matrices04, args.bootstrap_resamples)

    truck_selections = select_diverse(truck_ranked, 6)
    truck_examples = save_comparison_panels(
        dataset04, predictions03, predictions04, truck_selections, 14, TRUCK_EXAMPLES_DIR, names
    )
    light_selections = traffic_visual_selections(image_records, predictions03, predictions04, dataset04)
    light_examples = save_comparison_panels(
        dataset04, predictions03, predictions04, light_selections, 6, LIGHT_EXAMPLES_DIR, names
    )

    thin_rows = [
        row for row in flows
        if row["focus_train_id"] in THIN_CLASSES and row["other_train_id"] in THIN_CLASSES
    ]
    truck03 = class_statistics(matrix03, 14)
    truck04 = class_statistics(matrix04, 14)
    strong_gains = [row["class_name"] for row in class_deltas if row["delta_group"] == "strong_gain"]
    strong_regressions = [row["class_name"] for row in class_deltas if row["delta_group"] == "strong_regression"]
    recommendation = (
        "At 640x320, change only the two bottleneck convolutions from dilation=1 to dilation=2 "
        "(padding=2), preserving channels, parameters, loss, weights, optimizer, augmentation, seed, and epochs. "
        "This directly tests whether broader context reduces vehicle-class and city-specific semantic allocation errors."
    )
    dominant_pattern = (
        "Baseline 04 improves aggregate mIoU and several large vehicle classes, but residual errors are dominated "
        "by semantic allocation among related vehicle classes and by city/scene concentration. Small-component "
        "and thin-object improvements are inconsistent, so further resolution increases are not the best-supported next step."
    )
    summary = {
        "checkpoint": {
            "experiment": B04, "epoch": metadata04.epoch, "architecture": "UNet(base_channels=64)",
            "resolution": config04["resolution"], "exact_b03_weights_reused": config03["class_weights"] == config04["class_weights"],
        },
        "headline": recomputed04,
        "saved_confusion_exact_match": {B03: True, B04: True},
        "dev_val_images": len(dataset04),
        "truck": {"b03": truck03, "b04": truck04},
        "strong_gain_classes": strong_gains,
        "strong_regression_classes": strong_regressions,
        "spearman_support_vs_absolute_iou_delta": support_delta_correlation,
        "dominant_pattern": dominant_pattern,
        "recommendation": recommendation,
        "bootstrap": bootstrap,
        "traffic_light_components": light_component_summary,
        "traffic_light_boundary": light_boundary_rows,
        "scene_concentration": scene_concentration,
        "truck_examples": truck_examples,
        "traffic_light_examples": light_examples,
        "runtime_seconds": time.perf_counter() - started,
        "official_val_test_loaded": False,
    }

    write_csv(OUTPUT_DIR / "normalized_confusion_delta.csv", normalized_rows)
    write_csv(OUTPUT_DIR / "semantic_flow_changes.csv", flows)
    write_csv(OUTPUT_DIR / "truck_diagnostics.csv", truck_rows)
    write_csv(OUTPUT_DIR / "truck_top_regressions.csv", truck_ranked[:20])
    write_csv(OUTPUT_DIR / "train_bus_city_diagnostics.csv", city_focus)
    write_csv(OUTPUT_DIR / "train_bus_scene_concentration.csv", scene_rankings)
    write_csv(OUTPUT_DIR / "train_bus_concentration_summary.csv", scene_concentration)
    write_csv(OUTPUT_DIR / "truck_concentration_summary.csv", [truck_concentration(truck_rows)])
    write_csv(OUTPUT_DIR / "traffic_light_component_diagnostics.csv", light_components)
    write_csv(OUTPUT_DIR / "traffic_light_component_summary.csv", light_component_summary)
    write_csv(OUTPUT_DIR / "traffic_light_boundary_diagnostic.csv", light_boundary_rows)
    write_csv(OUTPUT_DIR / "thin_object_confusions.csv", thin_rows)
    write_csv(OUTPUT_DIR / "class_delta_decomposition.csv", class_deltas)
    write_csv(OUTPUT_DIR / "city_aggregate_comparison.csv", city_aggregate)
    write_csv(OUTPUT_DIR / "bootstrap_diagnostics.csv", bootstrap)
    summary = enrich_summary_from_outputs(summary)
    write_json(OUTPUT_DIR / "diagnostic_summary.json", summary)
    report_markdown(summary, names)
    print(json.dumps({
        "complete": True,
        "dev_val_images": len(dataset04),
        "miou": recomputed04["miou"],
        "mean_dice": recomputed04["mean_dice"],
        "pixel_accuracy": recomputed04["pixel_accuracy"],
        "bootstrap_resamples": args.bootstrap_resamples,
        "runtime_seconds": time.perf_counter() - started,
        "output_dir": str(OUTPUT_DIR),
        "official_val_test_loaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
