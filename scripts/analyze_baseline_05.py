"""Residual analysis of Baseline 05 versus Baseline 04 on dev validation only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import analyze_baseline_04 as a04  # noqa: E402
from analyze_baseline_01 import diagnostic_rows  # noqa: E402
from analyze_baseline_02 import metric_summary  # noqa: E402
from analyze_baseline_03 import class_boundary, ranks  # noqa: E402
from cityscapes_segmentation.data.dataset import CityscapesSegmentationDataset  # noqa: E402
from cityscapes_segmentation.data.labels import (  # noqa: E402
    get_train_id_to_name,
    map_raw_ids_to_train_ids,
)
from cityscapes_segmentation.data.splits import load_dev_split, validate_dev_split  # noqa: E402
from cityscapes_segmentation.data.transforms import SegmentationTransform  # noqa: E402
from cityscapes_segmentation.engine import load_checkpoint  # noqa: E402
from cityscapes_segmentation.models import UNet  # noqa: E402


B03 = "baseline_03_mild_weighted_ce"
B04 = "baseline_04_higher_resolution"
B05 = "baseline_05_dilated_bottleneck"
B03_DIR = PROJECT_ROOT / "outputs" / B03
B04_DIR = PROJECT_ROOT / "outputs" / B04
B05_DIR = PROJECT_ROOT / "outputs" / B05
OUTPUT_DIR = B05_DIR / "error_analysis"
MOTORCYCLE = 17
BUS = 15
TRAIN = 16
NUM_CLASSES = 19


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(a04.json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_model(experiment: str, dilation: int, device: torch.device):
    config = read_json(PROJECT_ROOT / "outputs" / experiment / "config.json")
    model = UNet(in_channels=3, num_classes=19, base_channels=64, bottleneck_dilation=dilation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    metadata = load_checkpoint(
        PROJECT_ROOT / "checkpoints" / experiment / "best.pt",
        model=model,
        optimizer=optimizer,
        map_location=device,
    )
    expected = {
        "experiment_name": experiment,
        "model": "UNet",
        "in_channels": 3,
        "num_classes": 19,
        "base_channels": 64,
        "resolution": {"width": 640, "height": 320},
    }
    mismatches = {key: (metadata.config.get(key), value) for key, value in expected.items() if metadata.config.get(key) != value}
    if mismatches or metadata.epoch != 30 or int(metadata.config.get("bottleneck_dilation", 1)) != dilation:
        raise RuntimeError(f"Checkpoint identity mismatch for {experiment}: epoch={metadata.epoch}, {mismatches}")
    if config != metadata.config:
        raise RuntimeError(f"Saved config and checkpoint config differ for {experiment}")
    model.eval()
    return model, metadata, config


def infer(model, dataset, device: torch.device):
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    predictions = []
    targets = []
    image_matrices = []
    with torch.inference_mode():
        for images, batch_targets in DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0):
            batch_predictions = model(images.to(device)).argmax(1).cpu().numpy().astype(np.uint8)
            for target, prediction in zip(batch_targets.numpy().astype(np.uint8), batch_predictions, strict=True):
                item = a04.matrix_for(target, prediction)
                matrix += item
                targets.append(target)
                predictions.append(prediction)
                image_matrices.append(item)
    return matrix, predictions, targets, np.stack(image_matrices)


def top_confusions(matrix: np.ndarray, class_id: int, names: dict[int, str], direction: str) -> list[dict]:
    if direction == "fn_destination":
        denominator = int(matrix[class_id].sum())
        values = ((other, int(matrix[class_id, other])) for other in range(NUM_CLASSES) if other != class_id)
    elif direction == "fp_source":
        denominator = int(matrix[:, class_id].sum() - matrix[class_id, class_id])
        values = ((other, int(matrix[other, class_id])) for other in range(NUM_CLASSES) if other != class_id)
    else:
        raise ValueError(direction)
    return [
        {
            "train_id": other,
            "class_name": names[other],
            "pixels": pixels,
            "percentage": 100 * pixels / denominator if denominator else math.nan,
        }
        for other, pixels in sorted(values, key=lambda item: (-item[1], item[0]))[:8]
    ]


def original_grid_analysis(dataset, predictions04, predictions05, names):
    motorcycle_rows = []
    components = []
    boundary = {model: {region: [0, 0] for region in ("boundary", "interior")} for model in (B04, B05)}
    city_matrices = {
        B04: defaultdict(lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)),
        B05: defaultdict(lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)),
    }
    image_positive = defaultdict(lambda: defaultdict(int))

    for index, ((image_path, mask_path), native04, native05) in enumerate(
        zip(dataset.samples, predictions04, predictions05, strict=True)
    ):
        with Image.open(mask_path) as source:
            target = map_raw_ids_to_train_ids(np.asarray(source))
        height, width = target.shape
        prediction04 = a04.resize_labels(native04, (width, height))
        prediction05 = a04.resize_labels(native05, (width, height))
        matrix04 = a04.matrix_for(target, prediction04)
        matrix05 = a04.matrix_for(target, prediction05)
        city = image_path.parent.name
        city_matrices[B04][city] += matrix04
        city_matrices[B05][city] += matrix05
        for class_id in range(NUM_CLASSES):
            image_positive[(city, class_id)]["images"] += int(np.any(target == class_id))

        old = a04.class_statistics(matrix04, MOTORCYCLE)
        new = a04.class_statistics(matrix05, MOTORCYCLE)
        if old["gt_pixels"] or old["predicted_pixels"] or new["predicted_pixels"]:
            delta_iou = new["iou"] - old["iou"] if math.isfinite(old["iou"]) and math.isfinite(new["iou"]) else math.nan
            motorcycle_rows.append({
                "dataset_index": index,
                "city": city,
                "image_id": image_path.name.removesuffix("_leftImg8bit.png"),
                "gt_pixels": old["gt_pixels"],
                "b04_tp": old["true_positives"],
                "b04_fp": old["false_positives"],
                "b04_fn": old["false_negatives"],
                "b04_iou": old["iou"],
                "b05_tp": new["true_positives"],
                "b05_fp": new["false_positives"],
                "b05_fn": new["false_negatives"],
                "b05_iou": new["iou"],
                "delta_iou": delta_iou,
                "error_increase": new["false_positives"] + new["false_negatives"] - old["false_positives"] - old["false_negatives"],
            })

        motorcycle_mask = target == MOTORCYCLE
        prediction_masks = {B04: prediction04 == MOTORCYCLE, B05: prediction05 == MOTORCYCLE}
        for component_index, indices in enumerate(a04.connected_components(motorcycle_mask), start=1):
            component_mask = np.zeros_like(motorcycle_mask)
            component_mask.flat[indices] = True
            rows, columns = np.where(component_mask)
            pad = 16
            r0, r1 = max(0, rows.min() - pad), min(height, rows.max() + pad + 1)
            c0, c1 = max(0, columns.min() - pad), min(width, columns.max() + pad + 1)
            record = {
                "dataset_index": index,
                "city": city,
                "image_id": image_path.name.removesuffix("_leftImg8bit.png"),
                "component_index": component_index,
                "area_original": len(indices),
            }
            for model_name, prediction_mask in prediction_masks.items():
                overlap = int((component_mask & prediction_mask).sum())
                predicted_local = int(prediction_mask[r0:r1, c0:c1].sum())
                union_local = len(indices) + predicted_local - overlap
                record[f"{model_name}_overlap"] = overlap
                record[f"{model_name}_recall"] = overlap / len(indices)
                record[f"{model_name}_local_iou"] = overlap / union_local if union_local else math.nan
                record[f"{model_name}_whole_miss"] = int(overlap == 0)
            components.append(record)

        band = class_boundary(motorcycle_mask, radius=8)
        for model_name, prediction_mask in prediction_masks.items():
            for region, selection in (("boundary", band), ("interior", motorcycle_mask & ~band)):
                boundary[model_name][region][0] += int(selection.sum())
                boundary[model_name][region][1] += int((selection & ~prediction_mask).sum())

    return motorcycle_rows, components, boundary, city_matrices, image_positive


def motorcycle_scene_tables(rows: list[dict]):
    valid = [row for row in rows if math.isfinite(row["delta_iou"])]
    regressions = sorted(valid, key=lambda row: (row["delta_iou"], row["city"], row["image_id"]))
    improvements = sorted(valid, key=lambda row: (-row["delta_iou"], row["city"], row["image_id"]))
    ranked = []
    for label, ordered in (("regression", regressions), ("improvement", improvements)):
        for rank, row in enumerate(ordered[:20], start=1):
            ranked.append({"ranking": label, "rank": rank, **row})

    increases = sorted((max(0, row["error_increase"]) for row in rows), reverse=True)
    gross = sum(increases)
    summary = {
        "images_with_gt": sum(row["gt_pixels"] > 0 for row in rows),
        "images_with_error_increase": sum(row["error_increase"] > 0 for row in rows),
        "images_with_error_reduction": sum(row["error_increase"] < 0 for row in rows),
        "net_error_increase": sum(row["error_increase"] for row in rows),
        "gross_error_increase": gross,
    }
    for count in (1, 3, 5):
        summary[f"top_{count}_share_of_gross_regression"] = sum(increases[:count]) / gross if gross else math.nan
    return ranked, summary


def component_tables(components: list[dict], boundary: dict):
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
        summary = {
            "size_bin": size_bin,
            "component_count": len(selected),
            "gt_pixels_original": pixels,
            "small_threshold_max_original_pixels": low,
            "medium_threshold_max_original_pixels": high,
        }
        for model_name in (B04, B05):
            overlap = sum(row[f"{model_name}_overlap"] for row in selected)
            summary[f"{model_name}_recall_proxy"] = overlap / pixels
            summary[f"{model_name}_mean_local_iou"] = float(np.mean([row[f"{model_name}_local_iou"] for row in selected]))
            summary[f"{model_name}_whole_miss_rate"] = float(np.mean([row[f"{model_name}_whole_miss"] for row in selected]))
        for metric in ("recall_proxy", "mean_local_iou", "whole_miss_rate"):
            summary[f"delta_{metric}"] = summary[f"{B05}_{metric}"] - summary[f"{B04}_{metric}"]
        summaries.append(summary)

    boundary_rows = []
    for region in ("boundary", "interior"):
        row = {"region": region, "original_grid_boundary_radius_pixels": 8}
        for model_name in (B04, B05):
            pixels, errors = boundary[model_name][region]
            row[f"{model_name}_pixels"] = pixels
            row[f"{model_name}_errors"] = errors
            row[f"{model_name}_error_rate"] = errors / pixels if pixels else math.nan
        row["delta_error_rate"] = row[f"{B05}_error_rate"] - row[f"{B04}_error_rate"]
        boundary_rows.append(row)
    return summaries, boundary_rows


def city_diagnostics(city_matrices, image_positive, names):
    rows = []
    for city in ("jena", "krefeld", "ulm"):
        for class_id in (BUS, TRAIN):
            row = {
                "city": city,
                "train_id": class_id,
                "class_name": names[class_id],
                "images_containing_class": image_positive[(city, class_id)]["images"],
            }
            for model_name in (B04, B05):
                stats = a04.class_statistics(city_matrices[model_name][city], class_id)
                for key in ("gt_pixels", "predicted_pixels", "precision", "recall", "iou", "prediction_ratio"):
                    row[f"{model_name}_{key}"] = stats[key]
            for metric in ("precision", "recall", "iou", "prediction_ratio"):
                row[f"delta_{metric}"] = row[f"{B05}_{metric}"] - row[f"{B04}_{metric}"]
            rows.append(row)
    return rows


def semantic_flow(matrix04, matrix05, names):
    pairs = (
        (TRAIN, BUS), (BUS, TRAIN), (TRAIN, 2), (BUS, 14), (BUS, 13), (TRAIN, 14),
        (MOTORCYCLE, 12), (MOTORCYCLE, 18), (MOTORCYCLE, 11), (MOTORCYCLE, 13),
        (MOTORCYCLE, 8), (MOTORCYCLE, 0),
    )
    rows = []
    for source, destination in pairs:
        row = {
            "source_train_id": source,
            "source_class": names[source],
            "destination_train_id": destination,
            "destination_class": names[destination],
        }
        for model_name, matrix in ((B04, matrix04), (B05, matrix05)):
            support = int(matrix[source].sum())
            pixels = int(matrix[source, destination])
            row[f"{model_name}_pixels"] = pixels
            row[f"{model_name}_gt_row_proportion"] = pixels / support if support else math.nan
        row["delta_gt_row_proportion"] = row[f"{B05}_gt_row_proportion"] - row[f"{B04}_gt_row_proportion"]
        rows.append(row)
    return rows


def ulm_deltas(city_matrices, names):
    diag04 = diagnostic_rows(city_matrices[B04]["ulm"], names)
    diag05 = diagnostic_rows(city_matrices[B05]["ulm"], names)
    rows = []
    for class_id in range(NUM_CLASSES):
        rows.append({
            "train_id": class_id,
            "class_name": names[class_id],
            "gt_pixels_original": diag04[class_id]["gt_pixels"],
            "b04_iou": diag04[class_id]["iou"],
            "b05_iou": diag05[class_id]["iou"],
            "delta_iou": diag05[class_id]["iou"] - diag04[class_id]["iou"],
        })
    return sorted(rows, key=lambda row: (-row["delta_iou"], row["train_id"]))


def support_delta_rows(matrix04, matrix05, names):
    diag04 = diagnostic_rows(matrix04, names)
    diag05 = diagnostic_rows(matrix05, names)
    support = np.asarray([row["gt_pixels"] for row in diag04], dtype=np.float64)
    deltas = np.asarray([diag05[index]["iou"] - diag04[index]["iou"] for index in range(NUM_CLASSES)])
    support_ranks = ranks(support)
    absolute_delta_ranks = ranks(np.abs(deltas))
    correlation = float(np.corrcoef(support_ranks, absolute_delta_ranks)[0, 1])
    rows = [
        {
            "train_id": class_id,
            "class_name": names[class_id],
            "support": int(support[class_id]),
            "support_share": support[class_id] / support.sum(),
            "b04_iou": diag04[class_id]["iou"],
            "b05_iou": diag05[class_id]["iou"],
            "delta_iou": deltas[class_id],
            "absolute_delta_iou": abs(deltas[class_id]),
            "spearman_support_vs_absolute_delta": correlation,
        }
        for class_id in range(NUM_CLASSES)
    ]
    return rows, correlation


def error_taxonomy(matrix05, city_matrices, names):
    diagnostics = diagnostic_rows(matrix05, names)
    total = matrix05.sum()
    labels = {
        "A": "underprediction",
        "B": "overprediction",
        "C": "semantic confusion",
        "D": "boundary/detail",
        "E": "sparse support instability",
        "F": "city/scene sensitivity",
        "G": "reasonably balanced/inconclusive",
    }
    rows = []
    detail_classes = {5, 6, 7, 12, 17, 18}
    for class_id, diagnostic in enumerate(diagnostics):
        if diagnostic["iou"] >= 0.45:
            continue
        categories = []
        if diagnostic["prediction_ratio"] < 0.85:
            categories.append("A")
        elif diagnostic["prediction_ratio"] > 1.15:
            categories.append("B")
        if top_confusions(matrix05, class_id, names, "fn_destination")[0]["percentage"] >= 5:
            categories.append("C")
        if class_id in detail_classes:
            categories.append("D")
        if diagnostic["gt_pixels"] / total < 0.005:
            categories.append("E")
        city_ious = [a04.class_statistics(city_matrices[B05][city], class_id)["iou"] for city in ("jena", "krefeld", "ulm")]
        defined = [value for value in city_ious if math.isfinite(value)]
        if defined and max(defined) - min(defined) > 0.15:
            categories.append("F")
        if not categories:
            categories.append("G")
        rows.append({
            "train_id": class_id,
            "class_name": names[class_id],
            "b05_iou": diagnostic["iou"],
            "prediction_gt_ratio": diagnostic["prediction_ratio"],
            "support_share": diagnostic["gt_pixels"] / total,
            "category_codes": ",".join(categories),
            "categories": "; ".join(labels[code] for code in categories),
        })
    return rows


def three_way_trends(names):
    metrics = {
        B03: read_json(B03_DIR / "best_metrics.json")["per_class"],
        B04: read_json(B04_DIR / "best_metrics.json")["per_class"],
        B05: read_json(B05_DIR / "best_metrics.json")["per_class"],
    }
    rows = []
    for class_id in (14, 16, 15, 17, 12, 6):
        row = {"train_id": class_id, "class_name": names[class_id]}
        for experiment, prefix in ((B03, "b03"), (B04, "b04"), (B05, "b05")):
            item = metrics[experiment][class_id]
            for metric in ("iou", "precision", "recall", "prediction_ratio"):
                row[f"{prefix}_{metric}"] = item[metric]
        rows.append(row)
    return rows


def bootstrap_diagnostics(image04, image05, dataset, samples=1000):
    rng = np.random.default_rng(42)
    requested = {"overall_miou": None, "motorcycle_iou": MOTORCYCLE, "train_iou": TRAIN, "bus_iou": BUS}
    values = {key: [] for key in (*requested, "ulm_city_miou")}
    ulm_indices = np.asarray([index for index, sample in enumerate(dataset.samples) if sample[0].parent.name == "ulm"])
    for _ in range(samples):
        counts = np.bincount(rng.integers(0, len(image04), len(image04)), minlength=len(image04))
        matrix04 = np.tensordot(counts, image04, axes=(0, 0))
        matrix05 = np.tensordot(counts, image05, axes=(0, 0))
        values["overall_miou"].append(metric_summary(matrix05)["miou"] - metric_summary(matrix04)["miou"])
        for key, class_id in requested.items():
            if class_id is not None:
                values[key].append(a04.class_statistics(matrix05, class_id)["iou"] - a04.class_statistics(matrix04, class_id)["iou"])
        sampled_ulm = ulm_indices[rng.integers(0, len(ulm_indices), len(ulm_indices))]
        ulm04 = image04[sampled_ulm].sum(axis=0)
        ulm05 = image05[sampled_ulm].sum(axis=0)
        values["ulm_city_miou"].append(metric_summary(ulm05)["miou"] - metric_summary(ulm04)["miou"])

    names = get_train_id_to_name()
    rows = []
    for key, observations in values.items():
        array = np.asarray(observations, dtype=np.float64)
        class_id = requested.get(key)
        rows.append({
            "metric": key,
            "class_name": "all_classes_macro" if key == "overall_miou" else "ulm_all_classes" if key == "ulm_city_miou" else names[class_id],
            "bootstrap_resamples": samples,
            "seed": 42,
            "mean_delta": float(array.mean()),
            "median_delta": float(np.median(array)),
            "percentile_2_5": float(np.quantile(array, 0.025)),
            "percentile_97_5": float(np.quantile(array, 0.975)),
            "fraction_delta_gt_zero": float((array > 0).mean()),
            "note": "Image-bootstrap diagnostic; not a formal benchmark confidence interval",
        })
    return rows


def convergence_diagnostics():
    with (B05_DIR / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    last = history[-5:]
    rows = []
    for offset, row in enumerate(last):
        previous = history[len(history) - 5 + offset - 1] if len(history) - 5 + offset > 0 else None
        rows.append({
            "epoch": int(row["epoch"]),
            "learning_rate": row["learning_rate"],
            "train_loss": row["train_loss"],
            "val_loss": row["val_loss"],
            "val_miou": row["val_miou"],
            "val_mean_dice": row["val_mean_dice"],
            "delta_miou_from_previous": row["val_miou"] - previous["val_miou"] if previous else math.nan,
            "delta_val_loss_from_previous": row["val_loss"] - previous["val_loss"] if previous else math.nan,
        })
    epochs = np.asarray([row["epoch"] for row in last])
    summary = {
        "window": "epochs_26_to_30",
        "miou_net_change": last[-1]["val_miou"] - last[0]["val_miou"],
        "mean_dice_net_change": last[-1]["val_mean_dice"] - last[0]["val_mean_dice"],
        "train_loss_net_change": last[-1]["train_loss"] - last[0]["train_loss"],
        "val_loss_net_change": last[-1]["val_loss"] - last[0]["val_loss"],
        "miou_linear_slope_per_epoch": float(np.polyfit(epochs, [row["val_miou"] for row in last], 1)[0]),
        "best_epoch_is_final_epoch": int(history[-1]["is_best"]) == 1,
        "final_lr": history[-1]["learning_rate"],
        "classification": "still_improving_modestly_but_noisy_at_schedule_endpoint",
    }
    return rows, summary


def report_markdown(summary):
    motorcycle = summary["motorcycle"]
    scene = summary["motorcycle_scene_concentration"]
    component = summary["motorcycle_components"]
    boundary = summary["motorcycle_boundary"]
    lines = [
        "# Baseline 05 Residual Failure Analysis",
        "",
        "Development validation only: 313 images from jena, krefeld, and ulm. Official Cityscapes val/test were not loaded.",
        "",
        "## Verification",
        "",
        f"- Checkpoint epoch: {summary['checkpoint']['epoch']}",
        f"- Bottleneck dilation: {summary['checkpoint']['bottleneck_dilation']}",
        f"- mIoU: {summary['headline']['miou']:.6f}",
        f"- Mean Dice: {summary['headline']['mean_dice']:.6f}",
        f"- Pixel accuracy: {summary['headline']['pixel_accuracy']:.6f}",
        "- Recomputed B04 and B05 confusion matrices exactly match their saved best-checkpoint results.",
        "",
        "## Motorcycle regression",
        "",
        f"IoU changed {motorcycle['b04']['iou']:.4f} -> {motorcycle['b05']['iou']:.4f}; "
        f"precision {motorcycle['b04']['precision']:.4f} -> {motorcycle['b05']['precision']:.4f}; "
        f"recall {motorcycle['b04']['recall']:.4f} -> {motorcycle['b05']['recall']:.4f}; "
        f"prediction/GT {motorcycle['b04']['prediction_ratio']:.3f}x -> {motorcycle['b05']['prediction_ratio']:.3f}x. "
        "The regression is primarily recall-driven underprediction.",
        "",
        f"The top 1/3/5 scenes account for {scene['top_1_share_of_gross_regression']:.1%} / "
        f"{scene['top_3_share_of_gross_regression']:.1%} / {scene['top_5_share_of_gross_regression']:.1%} of gross motorcycle error growth.",
        "",
        "## Motorcycle component and boundary diagnostics",
        "",
        "| Size | B04 recall | B05 recall | B04 local IoU | B05 local IoU | B04 miss | B05 miss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in component:
        lines.append(
            f"| {row['size_bin']} | {row[f'{B04}_recall_proxy']:.3f} | {row[f'{B05}_recall_proxy']:.3f} | "
            f"{row[f'{B04}_mean_local_iou']:.3f} | {row[f'{B05}_mean_local_iou']:.3f} | "
            f"{row[f'{B04}_whole_miss_rate']:.3f} | {row[f'{B05}_whole_miss_rate']:.3f} |"
        )
    lines.extend(["", "Boundary heuristic:"])
    for row in boundary:
        lines.append(
            f"- {row['region']}: error {row[f'{B04}_error_rate']:.3f} -> {row[f'{B05}_error_rate']:.3f} "
            f"(delta {row['delta_error_rate']:+.3f})."
        )
    lines.extend([
        "",
        "## Support stability",
        "",
        f"Spearman support versus absolute B05-B04 IoU delta: {summary['support_vs_delta_spearman']:.4f}. "
        "This is descriptive and does not establish causality.",
        "",
        "## Convergence",
        "",
        f"Across epochs 26-30, mIoU changed by {summary['convergence']['miou_net_change']:+.4f}, "
        f"validation loss by {summary['convergence']['val_loss_net_change']:+.4f}, and the final epoch was best. "
        "The curve still improved modestly but remained noisy at the endpoint of the locked cosine schedule.",
        "",
        "## Stopping decision",
        "",
        summary["stopping_assessment"],
        "",
        "## Exact next step",
        "",
        summary["recommended_next_step"],
        "",
        "## Method limits",
        "",
        "Component-size and boundary results are diagnostic proxies on original-resolution masks, not official Cityscapes instance-size or boundary metrics. Bootstrap intervals are deterministic image-resampling diagnostics, not formal benchmark confidence intervals.",
    ])
    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = get_train_id_to_name()

    model04, metadata04, config04 = load_model(B04, 1, device)
    model05, metadata05, config05 = load_model(B05, 2, device)
    if config04["class_weights"] != config05["class_weights"]:
        raise RuntimeError("Baseline 05 did not preserve the exact Baseline 04 class weights")
    if config05["resolution"] != {"width": 640, "height": 320}:
        raise RuntimeError("Baseline 05 resolution differs from 640x320")

    split = validate_dev_split(load_dev_split(), PROJECT_ROOT)
    if len(split.dev_val_images) != 313:
        raise RuntimeError(f"Expected 313 dev-validation images; got {len(split.dev_val_images)}")
    official_train = (PROJECT_ROOT / "leftImg8bit" / "train").resolve()
    if any(not path.resolve().is_relative_to(official_train) for path in split.dev_val_images):
        raise RuntimeError("Development validation contains a path outside official train")
    dataset = CityscapesSegmentationDataset(
        PROJECT_ROOT,
        split.dev_val_images,
        SegmentationTransform(size=(640, 320)),
    )
    matrix04, predictions04, _, image_matrices04 = infer(model04, dataset, device)
    matrix05, predictions05, _, image_matrices05 = infer(model05, dataset, device)
    metrics04 = read_json(B04_DIR / "best_metrics.json")
    metrics05 = read_json(B05_DIR / "best_metrics.json")
    if not np.array_equal(matrix04, np.asarray(metrics04["confusion_matrix"], dtype=np.int64)):
        raise RuntimeError("Recomputed Baseline 04 confusion matrix differs from saved result")
    if not np.array_equal(matrix05, np.asarray(metrics05["confusion_matrix"], dtype=np.int64)):
        raise RuntimeError("Recomputed Baseline 05 confusion matrix differs from saved result")
    headline = metric_summary(matrix05)
    for key, saved_key in (("miou", "best_miou"), ("mean_dice", "mean_dice"), ("pixel_accuracy", "pixel_accuracy")):
        if not math.isclose(headline[key], metrics05[saved_key], abs_tol=1e-12):
            raise RuntimeError(f"Recomputed Baseline 05 {key} differs from saved result")

    motorcycle04 = a04.class_statistics(matrix04, MOTORCYCLE)
    motorcycle05 = a04.class_statistics(matrix05, MOTORCYCLE)
    motorcycle_diagnostics = []
    for model_name, values in ((B04, motorcycle04), (B05, motorcycle05)):
        motorcycle_diagnostics.append({"experiment": model_name, **values})
    write_csv(OUTPUT_DIR / "motorcycle_diagnostics.csv", motorcycle_diagnostics)

    rows, components, boundary, city_matrices, image_positive = original_grid_analysis(
        dataset, predictions04, predictions05, names
    )
    scene_rankings, scene_summary = motorcycle_scene_tables(rows)
    component_summary, boundary_rows = component_tables(components, boundary)
    bus_train_city = city_diagnostics(city_matrices, image_positive, names)
    flow_rows = semantic_flow(matrix04, matrix05, names)
    ulm_rows = ulm_deltas(city_matrices, names)
    support_rows, support_correlation = support_delta_rows(matrix04, matrix05, names)
    taxonomy = error_taxonomy(matrix05, city_matrices, names)
    trends = three_way_trends(names)
    bootstrap_rows = bootstrap_diagnostics(image_matrices04, image_matrices05, dataset, args.bootstrap_resamples)
    convergence_rows, convergence_summary = convergence_diagnostics()

    write_csv(OUTPUT_DIR / "motorcycle_scene_concentration.csv", scene_rankings)
    write_csv(OUTPUT_DIR / "motorcycle_component_analysis.csv", component_summary)
    write_csv(OUTPUT_DIR / "motorcycle_boundary_analysis.csv", boundary_rows)
    write_csv(OUTPUT_DIR / "bus_city_diagnostics.csv", [row for row in bus_train_city if row["train_id"] == BUS])
    write_csv(OUTPUT_DIR / "train_city_diagnostics.csv", [row for row in bus_train_city if row["train_id"] == TRAIN])
    write_csv(OUTPUT_DIR / "train_bus_semantic_flow.csv", flow_rows)
    write_csv(OUTPUT_DIR / "ulm_class_deltas.csv", ulm_rows)
    write_csv(OUTPUT_DIR / "support_vs_delta.csv", support_rows)
    write_csv(OUTPUT_DIR / "error_taxonomy.csv", taxonomy)
    write_csv(OUTPUT_DIR / "three_way_class_trends.csv", trends)
    write_csv(OUTPUT_DIR / "bootstrap_diagnostics.csv", bootstrap_rows)
    write_csv(OUTPUT_DIR / "convergence_diagnostics.csv", convergence_rows)

    confusions = {
        "baseline_04": {
            "fn_destinations": top_confusions(matrix04, MOTORCYCLE, names, "fn_destination"),
            "fp_sources": top_confusions(matrix04, MOTORCYCLE, names, "fp_source"),
        },
        "baseline_05": {
            "fn_destinations": top_confusions(matrix05, MOTORCYCLE, names, "fn_destination"),
            "fp_sources": top_confusions(matrix05, MOTORCYCLE, names, "fp_source"),
        },
    }
    write_json(OUTPUT_DIR / "motorcycle_confusions.json", confusions)

    stopping_assessment = (
        "The project has a clear current best model after five controlled experiments. Aggregate B05 improvement is supported, "
        "but the remaining train/bus/motorcycle errors are sparse, city/scene-sensitive, and do not identify one clean mechanism. "
        "The official validation set is still untouched. Further development tuning now has a material risk of becoming "
        "hyperparameter fishing, especially because longer training would be a post-hoc change to the locked schedule."
    )
    recommended_next_step = (
        "Stop development experimentation, freeze the Baseline 05 epoch-30 checkpoint and its preprocessing/configuration, "
        "then perform the single final held-out evaluation on the untouched 500-image official Cityscapes validation split."
    )
    summary = {
        "checkpoint": {
            "epoch": metadata05.epoch,
            "best_miou": metadata05.best_miou,
            "bottleneck_dilation": config05["bottleneck_dilation"],
            "resolution": config05["resolution"],
            "exact_b04_class_weights": config04["class_weights"] == config05["class_weights"],
            "architecture": "UNet(in_channels=3,num_classes=19,base_channels=64,bottleneck_dilation=2)",
        },
        "headline": headline,
        "saved_confusion_exact_match": {B04: True, B05: True},
        "motorcycle": {"b04": motorcycle04, "b05": motorcycle05, "confusions": confusions},
        "motorcycle_scene_concentration": scene_summary,
        "motorcycle_components": component_summary,
        "motorcycle_boundary": boundary_rows,
        "bus_train_city": bus_train_city,
        "train_bus_semantic_flow": flow_rows,
        "ulm_top_positive": ulm_rows[:5],
        "ulm_top_negative": list(reversed(ulm_rows[-5:])),
        "support_vs_delta_spearman": support_correlation,
        "error_taxonomy": taxonomy,
        "three_way_trends": trends,
        "bootstrap": bootstrap_rows,
        "convergence": convergence_summary,
        "stopping_decision": "B_stop_experimentation_and_proceed_to_final_held_out_evaluation",
        "stopping_assessment": stopping_assessment,
        "baseline_06_justified": False,
        "recommended_next_step": recommended_next_step,
        "official_val_test_loaded": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(OUTPUT_DIR / "diagnostic_summary.json", summary)
    report_markdown(summary)
    print(json.dumps({
        "complete": True,
        "dev_val_images": len(dataset),
        "miou": headline["miou"],
        "mean_dice": headline["mean_dice"],
        "pixel_accuracy": headline["pixel_accuracy"],
        "support_vs_delta_spearman": support_correlation,
        "baseline_06_justified": False,
        "official_val_test_loaded": False,
        "runtime_seconds": time.perf_counter() - started,
        "output_dir": str(OUTPUT_DIR),
    }, indent=2))


if __name__ == "__main__":
    main()
