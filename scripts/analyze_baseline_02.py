"""Focused Baseline 02 error analysis on development validation only."""

from __future__ import annotations

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
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_baseline_01 import colorize, diagnostic_rows, top_destinations  # noqa: E402
from run_baseline_01 import save_json  # noqa: E402
from cityscapes_segmentation.data.dataset import CityscapesSegmentationDataset  # noqa: E402
from cityscapes_segmentation.data.labels import (  # noqa: E402
    get_ignore_index,
    get_train_id_classes,
    get_train_id_to_name,
)
from cityscapes_segmentation.data.splits import load_dev_split, validate_dev_split  # noqa: E402
from cityscapes_segmentation.data.transforms import IMAGENET_MEAN, IMAGENET_STD, make_val_transform  # noqa: E402
from cityscapes_segmentation.engine import load_checkpoint  # noqa: E402
from cityscapes_segmentation.models import UNet  # noqa: E402


NUM_CLASSES = 19
TRAFFIC_LIGHT = 6
COMMON_CLASSES = (0, 1, 2, 8, 10, 11, 13)
REGRESSED_CLASSES = (1, 2, 10, 8, 13, 0)
RARE_PREDICTIONS = (6, 12, 17, 14, 15, 16)
BOUNDARY_CLASSES = (1, 2, 10, 6)
BASELINE_01_DIR = PROJECT_ROOT / "outputs" / "baseline_01_unet_ce"
BASELINE_02_DIR = PROJECT_ROOT / "outputs" / "baseline_02_weighted_ce"
OUTPUT_DIR = BASELINE_02_DIR / "error_analysis"
TRAFFIC_DIR = OUTPUT_DIR / "traffic_light"


def checkpoint_model(experiment_dir: Path, expected_epoch: int, expected_metrics: dict, device: torch.device):
    model = UNet(in_channels=3, num_classes=19, base_channels=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    metadata = load_checkpoint(
        PROJECT_ROOT / "checkpoints" / experiment_dir.name / "best.pt",
        model=model,
        optimizer=optimizer,
        map_location=device,
    )
    expected_config = {"model": "UNet", "in_channels": 3, "num_classes": 19, "base_channels": 64}
    mismatches = {
        key: (metadata.config.get(key), value)
        for key, value in expected_config.items()
        if metadata.config.get(key) != value
    }
    if metadata.epoch != expected_epoch or mismatches:
        raise RuntimeError(f"Checkpoint identity mismatch for {experiment_dir.name}: epoch={metadata.epoch}, config={mismatches}")
    if not math.isclose(metadata.best_miou, float(expected_metrics["best_miou"]), abs_tol=1e-12):
        raise RuntimeError(f"Checkpoint mIoU metadata differs for {experiment_dir.name}")
    model.eval()
    return model, metadata


def add_matrix(matrix: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> None:
    valid = target != get_ignore_index()
    encoded = target[valid].astype(np.int64) * NUM_CLASSES + prediction[valid].astype(np.int64)
    matrix += np.bincount(encoded, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def metric_summary(matrix: np.ndarray) -> dict:
    diagnostics = diagnostic_rows(matrix, get_train_id_to_name())
    ious = np.asarray([row["iou"] for row in diagnostics], dtype=np.float64)
    dices = np.asarray([row["dice"] for row in diagnostics], dtype=np.float64)
    return {
        "miou": float(np.nanmean(ious)),
        "mean_dice": float(np.nanmean(dices)),
        "pixel_accuracy": float(np.trace(matrix) / matrix.sum()),
        "diagnostics": diagnostics,
    }


def boundary_band(target: np.ndarray, radius: int = 2) -> np.ndarray:
    """Return a heuristic band around valid ground-truth label transitions."""
    valid = target != get_ignore_index()
    edge = np.zeros_like(valid)
    vertical = valid[:-1] & valid[1:] & (target[:-1] != target[1:])
    horizontal = valid[:, :-1] & valid[:, 1:] & (target[:, :-1] != target[:, 1:])
    edge[:-1] |= vertical
    edge[1:] |= vertical
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    tensor = torch.from_numpy(edge).to(torch.float32)[None, None]
    size = 2 * radius + 1
    return functional.max_pool2d(tensor, kernel_size=size, stride=1, padding=radius)[0, 0].numpy().astype(bool)


def evaluate_models(models, dataset, loader, device):
    matrices = {name: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for name in models}
    city_matrices = {
        name: defaultdict(lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)) for name in models
    }
    image_rows = []
    boundary_counts = {
        name: {class_id: {region: [0, 0] for region in ("boundary", "interior")} for class_id in BOUNDARY_CLASSES}
        for name in models
    }
    offset = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            predictions = {
                name: model(images).argmax(dim=1).cpu().numpy().astype(np.uint8)
                for name, model in models.items()
            }
            targets_np = targets.numpy().astype(np.uint8)
            for batch_index in range(len(targets_np)):
                dataset_index = offset + batch_index
                image_path = dataset.samples[dataset_index][0]
                city = image_path.parent.name
                target = targets_np[batch_index]
                band = boundary_band(target)
                for name in models:
                    prediction = predictions[name][batch_index]
                    add_matrix(matrices[name], target, prediction)
                    add_matrix(city_matrices[name][city], target, prediction)
                    for class_id in BOUNDARY_CLASSES:
                        class_mask = target == class_id
                        for region, region_mask in (("boundary", band), ("interior", ~band)):
                            selected = class_mask & region_mask
                            pixels = int(selected.sum())
                            errors = int((selected & (prediction != class_id)).sum())
                            boundary_counts[name][class_id][region][0] += pixels
                            boundary_counts[name][class_id][region][1] += errors

                prediction = predictions["baseline_02"][batch_index]
                gt = target == TRAFFIC_LIGHT
                pred = prediction == TRAFFIC_LIGHT
                image_rows.append({
                    "dataset_index": dataset_index,
                    "city": city,
                    "image_id": image_path.name.removesuffix("_leftImg8bit.png"),
                    "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                    "gt_pixels": int(gt.sum()),
                    "predicted_pixels": int(pred.sum()),
                    "tp_pixels": int((gt & pred).sum()),
                    "fp_pixels": int((~gt & pred & (target != get_ignore_index())).sum()),
                    "fn_pixels": int((gt & ~pred).sum()),
                })
            offset += len(targets_np)
    return matrices, city_matrices, image_rows, boundary_counts


def verify_matrices(matrices: dict[str, np.ndarray], saved: dict[str, dict]) -> dict[str, dict]:
    summaries = {}
    for name, matrix in matrices.items():
        expected = np.asarray(saved[name]["confusion_matrix"], dtype=np.int64)
        if not np.array_equal(matrix, expected):
            raise RuntimeError(f"{name} independently accumulated confusion matrix differs by {np.abs(matrix - expected).sum()} pixels")
        summary = metric_summary(matrix)
        for key, saved_key in (("miou", "best_miou"), ("mean_dice", "mean_dice"), ("pixel_accuracy", "pixel_accuracy")):
            if not math.isclose(summary[key], float(saved[name][saved_key]), rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f"{name} recomputed {key} differs from saved metric")
        summaries[name] = summary
    return summaries


def write_traffic_tables(matrices, image_rows, names):
    matrix_02 = matrices["baseline_02"]
    matrix_01 = matrices["baseline_01"]
    fp_total = int(matrix_02[:, TRAFFIC_LIGHT].sum() - matrix_02[TRAFFIC_LIGHT, TRAFFIC_LIGHT])
    with (OUTPUT_DIR / "traffic_light_fp_sources.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("rank", "target_train_id", "target_class", "fp_pixels", "percentage_of_all_traffic_light_fp")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        values = sorted(
            ((class_id, int(matrix_02[class_id, TRAFFIC_LIGHT])) for class_id in range(NUM_CLASSES) if class_id != TRAFFIC_LIGHT),
            key=lambda item: (-item[1], item[0]),
        )
        for rank, (class_id, pixels) in enumerate(values, start=1):
            writer.writerow({
                "rank": rank, "target_train_id": class_id, "target_class": names[class_id],
                "fp_pixels": pixels, "percentage_of_all_traffic_light_fp": 100 * pixels / fp_total,
            })

    with (OUTPUT_DIR / "traffic_light_fn_destinations.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("model", "rank", "predicted_train_id", "predicted_class", "pixels", "percentage_of_traffic_light_gt")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model_name, matrix in (("baseline_01", matrix_01), ("baseline_02", matrix_02)):
            support = int(matrix[TRAFFIC_LIGHT].sum())
            values = sorted(
                ((class_id, int(matrix[TRAFFIC_LIGHT, class_id])) for class_id in range(NUM_CLASSES) if class_id != TRAFFIC_LIGHT),
                key=lambda item: (-item[1], item[0]),
            )
            for rank, (class_id, pixels) in enumerate(values, start=1):
                writer.writerow({
                    "model": model_name, "rank": rank, "predicted_train_id": class_id,
                    "predicted_class": names[class_id], "pixels": pixels,
                    "percentage_of_traffic_light_gt": 100 * pixels / support,
                })

    ranked = sorted(image_rows, key=lambda row: (-row["fp_pixels"], row["city"], row["image_id"]))
    with (OUTPUT_DIR / "traffic_light_image_ranking.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("rank", "dataset_index", "city", "image_id", "fp_pixels", "gt_pixels", "predicted_pixels", "tp_pixels", "fn_pixels")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            writer.writerow({"rank": rank, **{key: row[key] for key in fields if key != "rank"}})
    return ranked


def write_regression_tables(matrices, summaries, names):
    old = summaries["baseline_01"]["diagnostics"]
    new = summaries["baseline_02"]["diagnostics"]
    fields = (
        "train_id", "class_name", "b01_precision", "b02_precision", "delta_precision",
        "b01_recall", "b02_recall", "delta_recall", "b01_iou", "b02_iou", "delta_iou",
        "gt_pixels", "b01_predicted_pixels", "b02_predicted_pixels", "b01_prediction_gt_ratio",
        "b02_prediction_gt_ratio", "delta_false_positives", "delta_false_negatives", "primary_error_change",
    )
    with (OUTPUT_DIR / "per_class_regression_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for class_id in REGRESSED_CLASSES:
            before, after = old[class_id], new[class_id]
            delta_fp = after["false_positives"] - before["false_positives"]
            delta_fn = after["false_negatives"] - before["false_negatives"]
            if delta_fp > 0 and delta_fn > 0:
                cause = "both_fp_and_fn_increased"
            elif delta_fp > 0:
                cause = "false_positive_increase"
            elif delta_fn > 0:
                cause = "false_negative_increase"
            else:
                cause = "neither_count_increased"
            writer.writerow({
                "train_id": class_id, "class_name": names[class_id],
                "b01_precision": before["precision"], "b02_precision": after["precision"],
                "delta_precision": after["precision"] - before["precision"],
                "b01_recall": before["recall"], "b02_recall": after["recall"],
                "delta_recall": after["recall"] - before["recall"],
                "b01_iou": before["iou"], "b02_iou": after["iou"], "delta_iou": after["iou"] - before["iou"],
                "gt_pixels": before["gt_pixels"], "b01_predicted_pixels": before["predicted_pixels"],
                "b02_predicted_pixels": after["predicted_pixels"],
                "b01_prediction_gt_ratio": before["prediction_ratio"], "b02_prediction_gt_ratio": after["prediction_ratio"],
                "delta_false_positives": delta_fp, "delta_false_negatives": delta_fn, "primary_error_change": cause,
            })

    with (OUTPUT_DIR / "common_confusion_changes.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "gt_train_id", "gt_class", "predicted_train_id", "predicted_class", "b01_pixels", "b02_pixels",
            "delta_pixels", "b01_percentage_of_gt", "b02_percentage_of_gt", "delta_percentage_points",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for gt_id in REGRESSED_CLASSES:
            support = int(matrices["baseline_01"][gt_id].sum())
            rows = []
            for pred_id in range(NUM_CLASSES):
                if pred_id == gt_id:
                    continue
                b01 = int(matrices["baseline_01"][gt_id, pred_id])
                b02 = int(matrices["baseline_02"][gt_id, pred_id])
                rows.append({
                    "gt_train_id": gt_id, "gt_class": names[gt_id], "predicted_train_id": pred_id,
                    "predicted_class": names[pred_id], "b01_pixels": b01, "b02_pixels": b02,
                    "delta_pixels": b02 - b01, "b01_percentage_of_gt": 100 * b01 / support,
                    "b02_percentage_of_gt": 100 * b02 / support,
                    "delta_percentage_points": 100 * (b02 - b01) / support,
                })
            for row in sorted(rows, key=lambda item: (-item["delta_pixels"], item["predicted_train_id"])):
                writer.writerow(row)


def write_rare_stealing_matrix(matrices, names):
    with (OUTPUT_DIR / "rare_class_fp_source_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["true_train_id", "true_class"]
        for predicted_id in RARE_PREDICTIONS:
            slug = names[predicted_id].replace(" ", "_")
            fields.extend((f"b01_to_{slug}", f"b02_to_{slug}", f"delta_to_{slug}"))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for true_id in COMMON_CLASSES:
            row = {"true_train_id": true_id, "true_class": names[true_id]}
            for predicted_id in RARE_PREDICTIONS:
                slug = names[predicted_id].replace(" ", "_")
                b01 = int(matrices["baseline_01"][true_id, predicted_id])
                b02 = int(matrices["baseline_02"][true_id, predicted_id])
                row.update({f"b01_to_{slug}": b01, f"b02_to_{slug}": b02, f"delta_to_{slug}": b02 - b01})
            writer.writerow(row)


def write_city_comparison(city_matrices, names):
    rows = []
    for city in ("jena", "krefeld", "ulm"):
        summaries = {name: metric_summary(city_matrices[name][city]) for name in ("baseline_01", "baseline_02")}
        row = {"city": city}
        for metric_name in ("miou",):
            row[f"b01_{metric_name}"] = summaries["baseline_01"][metric_name]
            row[f"b02_{metric_name}"] = summaries["baseline_02"][metric_name]
            row[f"delta_{metric_name}"] = summaries["baseline_02"][metric_name] - summaries["baseline_01"][metric_name]
        for class_id, label in ((6, "traffic_light"), (1, "sidewalk"), (2, "building"), (10, "sky")):
            before = summaries["baseline_01"]["diagnostics"][class_id]
            after = summaries["baseline_02"]["diagnostics"][class_id]
            for metric_name in (("precision",) if class_id != 6 else ("precision", "recall", "iou")):
                row[f"b01_{label}_{metric_name}"] = before[metric_name]
                row[f"b02_{label}_{metric_name}"] = after[metric_name]
                row[f"delta_{label}_{metric_name}"] = after[metric_name] - before[metric_name]
            if class_id != 6:
                row[f"b01_{label}_iou"] = before["iou"]
                row[f"b02_{label}_iou"] = after["iou"]
                row[f"delta_{label}_iou"] = after["iou"] - before["iou"]
        rows.append(row)
    with (OUTPUT_DIR / "city_level_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_boundary_diagnostic(boundary_counts, names):
    rows = []
    for class_id in BOUNDARY_CLASSES:
        for model_name in ("baseline_01", "baseline_02"):
            for region in ("boundary", "interior"):
                pixels, errors = boundary_counts[model_name][class_id][region]
                rows.append({
                    "model": model_name, "train_id": class_id, "class_name": names[class_id], "region": region,
                    "band_radius_pixels": 2, "pixels": pixels, "error_pixels": errors,
                    "error_rate": errors / pixels if pixels else math.nan,
                })
    with (OUTPUT_DIR / "boundary_diagnostic.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def select_traffic_examples(ranked):
    selected = []
    categories = defaultdict(list)
    categories["high_fp"] = ranked[:3]
    categories["correct_detection"] = sorted(
        (row for row in ranked if row["tp_pixels"] > 0), key=lambda row: (-row["tp_pixels"], row["fp_pixels"], row["dataset_index"])
    )[:2]
    categories["poor_gt_recovery"] = sorted(
        (row for row in ranked if row["gt_pixels"] > 0), key=lambda row: (-row["fn_pixels"], row["dataset_index"])
    )[:3]
    for category, rows in categories.items():
        for row in rows:
            existing = next((item for item in selected if item["dataset_index"] == row["dataset_index"]), None)
            if existing:
                existing["categories"].append(category)
            else:
                selected.append({**row, "categories": [category]})
    return sorted(selected, key=lambda row: row["dataset_index"])


def save_traffic_examples(model, dataset, selected, device, names):
    palette = np.asarray([label.color for label in get_train_id_classes()], dtype=np.uint8)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    outputs = []
    with torch.inference_mode():
        for row in selected:
            image, target = dataset[row["dataset_index"]]
            prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()
            target_np = target.numpy()
            rgb = ((image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            gt = target_np == TRAFFIC_LIGHT
            pred = prediction == TRAFFIC_LIGHT
            overlay = np.zeros((*gt.shape, 4), dtype=np.uint8)
            overlay[gt & ~pred] = (230, 70, 70, 190)
            overlay[pred & ~gt] = (70, 130, 230, 190)
            overlay[gt & pred] = (70, 210, 100, 210)
            figure, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(rgb); axes[0].set_title("RGB")
            axes[1].imshow(colorize(target_np, palette)); axes[1].set_title("Ground truth")
            axes[2].imshow(colorize(prediction, palette)); axes[2].set_title("Baseline 02 prediction")
            axes[3].imshow(rgb); axes[3].imshow(overlay); axes[3].set_title("Traffic-light overlay")
            for axis in axes:
                axis.axis("off")
            axes[3].legend(handles=(
                Patch(color=(0.90, 0.27, 0.27), label="FN / GT only"),
                Patch(color=(0.27, 0.51, 0.90), label="FP / prediction only"),
                Patch(color=(0.27, 0.82, 0.39), label="TP / overlap"),
            ), loc="lower center", fontsize=7)
            title = f"{row['city']}/{row['image_id']} | {', '.join(row['categories'])} | TP {row['tp_pixels']} FP {row['fp_pixels']} FN {row['fn_pixels']}"
            figure.suptitle(title, fontsize=10)
            figure.tight_layout()
            filename = f"{row['dataset_index']:03d}_{row['image_id']}.png"
            figure.savefig(TRAFFIC_DIR / filename, dpi=150)
            plt.close(figure)
            outputs.append({"file": filename, **row})
    return outputs


def milder_weights(names):
    config = json.loads((BASELINE_02_DIR / "config.json").read_text(encoding="utf-8"))
    frequencies = np.asarray(config["class_frequencies"], dtype=np.float64)
    raw = frequencies ** -0.25
    weights = raw / raw.mean()
    with (OUTPUT_DIR / "milder_weight_diagnostic.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("train_id", "class_name", "frequency", "normalized_frequency_power_minus_0_25_weight"))
        for class_id in range(NUM_CLASSES):
            writer.writerow((class_id, names[class_id], frequencies[class_id], weights[class_id]))
    return {
        "exponent": 0.25,
        "minimum": float(weights.min()), "maximum": float(weights.max()),
        "max_min_ratio": float(weights.max() / weights.min()),
        "weights": {names[class_id]: float(weights[class_id]) for class_id in range(NUM_CLASSES)},
    }


def plot_fp_sources(matrix, names):
    values = sorted(
        ((class_id, int(matrix[class_id, TRAFFIC_LIGHT])) for class_id in range(NUM_CLASSES) if class_id != TRAFFIC_LIGHT),
        key=lambda item: (-item[1], item[0]),
    )[:10]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar([names[class_id] for class_id, _ in values], [pixels for _, pixels in values])
    axis.set(xlabel="True class", ylabel="Pixels predicted as traffic light", title="Baseline 02 traffic-light false-positive sources")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "traffic_light_fp_sources.png", dpi=150)
    plt.close(figure)


def _make_report_legacy(metadata, summaries, matrices, ranked, city_rows, boundary_rows, milder, examples, runtime, names):
    traffic = summaries["baseline_02"]["diagnostics"][TRAFFIC_LIGHT]
    fp_sources = sorted(
        ((class_id, int(matrices["baseline_02"][class_id, TRAFFIC_LIGHT])) for class_id in range(NUM_CLASSES) if class_id != TRAFFIC_LIGHT),
        key=lambda item: (-item[1], item[0]),
    )
    fn_destinations = top_destinations(matrices["baseline_02"], TRAFFIC_LIGHT, names, count=5)
    total_fp = traffic["false_positives"]
    active_images = sum(row["fp_pixels"] > 0 for row in ranked)
    top10_share = sum(row["fp_pixels"] for row in ranked[:10]) / total_fp
    city_fp = {city: sum(row["fp_pixels"] for row in ranked if row["city"] == city) for city in ("jena", "krefeld", "ulm")}
    common_before = summaries["baseline_01"]["diagnostics"]
    common_after = summaries["baseline_02"]["diagnostics"]

    lines = [
        "# Baseline 02 Focused Dev-Validation Error Analysis", "",
        "## Identity and reproduction", "",
        f"- Checkpoint epoch: {metadata.epoch}",
        "- Model: `UNet(in_channels=3, num_classes=19, base_channels=64)`",
        "- Data: 313 dev-validation images from jena, krefeld, and ulm only",
        "- Transform: deterministic 512×256 Baseline validation transform",
        f"- Recomputed mIoU: {summaries['baseline_02']['miou']:.9f}",
        f"- Recomputed mean Dice: {summaries['baseline_02']['mean_dice']:.9f}",
        f"- Recomputed pixel accuracy: {summaries['baseline_02']['pixel_accuracy']:.9f}",
        "- The independently accumulated Baseline 01 and Baseline 02 confusion matrices exactly match their saved matrices.",
        f"- Runtime: {runtime:.2f} seconds", "",
        "## Traffic-light diagnosis", "",
        f"- TP: {traffic['true_positives']:,}; FP: {traffic['false_positives']:,}; FN: {traffic['false_negatives']:,}",
        f"- Precision: {traffic['precision']:.4f}; recall: {traffic['recall']:.4f}; IoU: {traffic['iou']:.4f}; predicted/GT: {traffic['prediction_ratio']:.3f}×",
        "- Top FP source classes: " + ", ".join(f"{names[class_id]} {pixels:,} ({100*pixels/total_fp:.2f}%)" for class_id, pixels in fp_sources[:6]) + ".",
        "- Top FN destinations: " + ", ".join(f"{name} {pixels:,} ({percentage:.2f}% of GT)" for name, pixels, percentage in fn_destinations) + ".",
        f"- FP-bearing images: {active_images}/313; top-10 images contain {100*top10_share:.2f}% of all traffic-light FP pixels.",
        "- City FP totals: " + ", ".join(f"{city} {pixels:,}" for city, pixels in city_fp.items()) + ".", "",
        "**Traffic-light category: A + C + D.** The dominant change is false-positive overprediction. Errors also involve boundary/detail regions and semantic confusion, especially building, vegetation, pole, and traffic sign. Scene concentration is reported quantitatively rather than assumed.", "",
        "## Common-class regression", "",
        "| Class | Δ precision | Δ recall | Δ IoU | B01 pred/GT | B02 pred/GT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for class_id in REGRESSED_CLASSES:
        before, after = common_before[class_id], common_after[class_id]
        lines.append(
            f"| {names[class_id]} | {after['precision']-before['precision']:+.4f} | "
            f"{after['recall']-before['recall']:+.4f} | {after['iou']-before['iou']:+.4f} | "
            f"{before['prediction_ratio']:.3f} | {after['prediction_ratio']:.3f} |"
        )
    lines.extend([
        "", "- **Sidewalk: A + C + D.** Precision fell much more than recall; increased terrain/wall/fence assignments and boundary error indicate overprediction/semantic redistribution around transitions.",
        "- **Building: B + C + D.** Recall fell while precision rose; the regression is driven primarily by false negatives into wall, pole, traffic sign, vegetation, and weighted rare vehicle classes.",
        "- **Sky: A + C + D.** Recall improved but precision fell; the net regression is false-positive overprediction plus boundary/semantic leakage, not under-detection.",
        "", "## City comparison", "",
        "| City | Δ mIoU | Δ traffic-light IoU | Δ sidewalk IoU | Δ building IoU | Δ sky IoU |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in city_rows:
        lines.append(
            f"| {row['city']} | {row['delta_miou']:+.4f} | {row['delta_traffic_light_iou']:+.4f} | "
            f"{row['delta_sidewalk_iou']:+.4f} | {row['delta_building_iou']:+.4f} | {row['delta_sky_iou']:+.4f} |"
        )
    lines.extend(["", "## Boundary heuristic", "", "A two-pixel band around ground-truth class transitions was used. This is diagnostic, not an official Cityscapes boundary metric.", ""])
    for class_id in BOUNDARY_CLASSES:
        parts = []
        for model_name in ("baseline_01", "baseline_02"):
            values = {row["region"]: row["error_rate"] for row in boundary_rows if row["train_id"] == class_id and row["model"] == model_name}
            parts.append(f"{model_name}: boundary {values['boundary']:.3f}, interior {values['interior']:.3f}")
        lines.append(f"- **{names[class_id]}:** " + "; ".join(parts) + ".")
    lines.extend([
        "", "## Milder weighting diagnostic", "",
        f"For `weight ∝ frequency^-0.25`, the normalized range is {milder['minimum']:.4f}–{milder['maximum']:.4f} with a {milder['max_min_ratio']:.3f}× max/min ratio.",
        "", "## Recommended next controlled experiment", "",
        "Test **normalized `frequency^-0.25` class weights** as the single change from Baseline 02. Baseline 02 corrected rare-class underprediction but overshot traffic light and shifted pixels away from common classes. A milder exponent directly tests weight strength while preserving the successful weighting mechanism. It is preferable to higher resolution because the dominant measured change is prediction-frequency redistribution; preferable to CE + Dice because that adds a different loss mechanism; and preferable to arbitrary clipping because the exponent supplies a smooth, reproducible reduction for every class.",
        "", "No training was started.", "", "## Qualitative examples", "",
    ])
    lines.extend(f"- `{item['file']}`: {', '.join(item['categories'])}" for item in examples)
    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"active_fp_images": active_images, "top10_fp_share": top10_share, "city_fp_pixels": city_fp}


def make_report(metadata, summaries, matrices, ranked, city_rows, boundary_rows, milder, examples, runtime, names):
    """Write the evidence summary using only ASCII punctuation for portability."""
    traffic = summaries["baseline_02"]["diagnostics"][TRAFFIC_LIGHT]
    fp_sources = sorted(
        (
            (class_id, int(matrices["baseline_02"][class_id, TRAFFIC_LIGHT]))
            for class_id in range(NUM_CLASSES)
            if class_id != TRAFFIC_LIGHT
        ),
        key=lambda item: (-item[1], item[0]),
    )
    fn_destinations = top_destinations(matrices["baseline_02"], TRAFFIC_LIGHT, names, count=5)
    total_fp = traffic["false_positives"]
    active_images = sum(row["fp_pixels"] > 0 for row in ranked)
    top10_share = sum(row["fp_pixels"] for row in ranked[:10]) / total_fp
    city_fp = {
        city: sum(row["fp_pixels"] for row in ranked if row["city"] == city)
        for city in ("jena", "krefeld", "ulm")
    }
    common_before = summaries["baseline_01"]["diagnostics"]
    common_after = summaries["baseline_02"]["diagnostics"]

    lines = [
        "# Baseline 02 Focused Dev-Validation Error Analysis", "",
        "## Identity and reproduction", "",
        f"- Checkpoint epoch: {metadata.epoch}",
        "- Model: `UNet(in_channels=3, num_classes=19, base_channels=64)`",
        "- Data: 313 dev-validation images from jena, krefeld, and ulm only",
        "- Transform: deterministic 512x256 Baseline validation transform",
        f"- Recomputed mIoU: {summaries['baseline_02']['miou']:.9f}",
        f"- Recomputed mean Dice: {summaries['baseline_02']['mean_dice']:.9f}",
        f"- Recomputed pixel accuracy: {summaries['baseline_02']['pixel_accuracy']:.9f}",
        "- Both independently accumulated confusion matrices exactly match their saved matrices.",
        f"- Runtime: {runtime:.2f} seconds", "",
        "## Traffic-light diagnosis", "",
        f"- TP: {traffic['true_positives']:,}; FP: {traffic['false_positives']:,}; FN: {traffic['false_negatives']:,}",
        f"- Precision: {traffic['precision']:.4f}; recall: {traffic['recall']:.4f}; "
        f"IoU: {traffic['iou']:.4f}; predicted/GT: {traffic['prediction_ratio']:.3f}x",
        "- Top FP source classes: "
        + ", ".join(
            f"{names[class_id]} {pixels:,} ({100 * pixels / total_fp:.2f}%)"
            for class_id, pixels in fp_sources[:6]
        ) + ".",
        "- Top FN destinations: "
        + ", ".join(
            f"{name} {pixels:,} ({percentage:.2f}% of GT)"
            for name, pixels, percentage in fn_destinations
        ) + ".",
        f"- FP-bearing images: {active_images}/313; top-10 images contain "
        f"{100 * top10_share:.2f}% of all traffic-light FP pixels.",
        "- City FP totals: " + ", ".join(f"{city} {pixels:,}" for city, pixels in city_fp.items()) + ".", "",
        "**Traffic-light category: A + D.** The precision collapse is broad false-positive "
        "overprediction with semantic confusion, especially building, vegetation, pole, and traffic sign. "
        "Ground-truth traffic-light boundary and interior miss rates both improved, so this heuristic does "
        "not support boundary loss as the cause of the regression.", "",
        "## Common-class regression", "",
        "| Class | Delta precision | Delta recall | Delta IoU | B01 pred/GT | B02 pred/GT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for class_id in REGRESSED_CLASSES:
        before, after = common_before[class_id], common_after[class_id]
        lines.append(
            f"| {names[class_id]} | {after['precision'] - before['precision']:+.4f} | "
            f"{after['recall'] - before['recall']:+.4f} | "
            f"{after['iou'] - before['iou']:+.4f} | "
            f"{before['prediction_ratio']:.3f} | {after['prediction_ratio']:.3f} |"
        )
    lines.extend([
        "", "- **Sidewalk: A + D.** Precision fell much more than recall. Boundary error was effectively "
        "unchanged while interior error rose slightly, supporting overprediction and semantic redistribution "
        "(especially terrain and wall), not a boundary-driven regression.",
        "- **Building: B + C + D.** Recall fell while precision rose. False negatives increased sharply, "
        "with both boundary and interior error growth and redistribution into pole, wall, sky, traffic sign, "
        "traffic light, train, and other classes.",
        "- **Sky: A + D.** Recall and both target-region error rates improved, but precision fell sharply. "
        "The IoU regression is false-positive overprediction and semantic leakage into sky, not under-detection "
        "or a supported boundary regression.",
        "", "## City comparison", "",
        "| City | Delta mIoU | Delta traffic-light IoU | Delta sidewalk IoU | Delta building IoU | Delta sky IoU |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in city_rows:
        lines.append(
            f"| {row['city']} | {row['delta_miou']:+.4f} | "
            f"{row['delta_traffic_light_iou']:+.4f} | {row['delta_sidewalk_iou']:+.4f} | "
            f"{row['delta_building_iou']:+.4f} | {row['delta_sky_iou']:+.4f} |"
        )
    lines.extend([
        "", "Sidewalk, building, and sky regress in all three cities, so their direction is systematic "
        "rather than dominated by one city. Traffic-light precision also falls in every city, although its "
        "IoU improves in krefeld and declines in jena and ulm.",
        "", "## Boundary heuristic", "",
        "A two-pixel band around ground-truth class transitions was used. This is diagnostic, not an official "
        "Cityscapes boundary metric.", "",
    ])
    for class_id in BOUNDARY_CLASSES:
        parts = []
        for model_name in ("baseline_01", "baseline_02"):
            values = {
                row["region"]: row["error_rate"]
                for row in boundary_rows
                if row["train_id"] == class_id and row["model"] == model_name
            }
            parts.append(
                f"{model_name}: boundary {values['boundary']:.3f}, interior {values['interior']:.3f}"
            )
        lines.append(f"- **{names[class_id]}:** " + "; ".join(parts) + ".")
    lines.extend([
        "", "## Milder weighting diagnostic", "",
        f"For `weight proportional to frequency^-0.25`, the normalized range is "
        f"{milder['minimum']:.4f}-{milder['maximum']:.4f} with a "
        f"{milder['max_min_ratio']:.3f}x max/min ratio.",
        "", "## Recommended next controlled experiment", "",
        "Test **normalized `frequency^-0.25` class weights** as the single change from Baseline 02. "
        "Baseline 02 corrected rare-class underprediction but overshot traffic light and redirected common "
        "pixels into several heavily weighted rare classes. A milder exponent directly tests weight strength "
        "while preserving the successful weighting mechanism. It is preferable to higher resolution because "
        "the dominant measured change is prediction-frequency redistribution; preferable to CE + Dice because "
        "that adds a different loss mechanism; and preferable to arbitrary clipping because the exponent gives "
        "a smooth, reproducible reduction for every class.",
        "", "No training was started.", "", "## Qualitative examples", "",
    ])
    lines.extend(f"- `{item['file']}`: {', '.join(item['categories'])}" for item in examples)
    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "active_fp_images": active_images,
        "top10_fp_share": top10_share,
        "city_fp_pixels": city_fp,
    }


def main() -> None:
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline 02 focused analysis requires CUDA")
    saved = {
        "baseline_01": json.loads((BASELINE_01_DIR / "best_metrics.json").read_text(encoding="utf-8")),
        "baseline_02": json.loads((BASELINE_02_DIR / "best_metrics.json").read_text(encoding="utf-8")),
    }
    split = load_dev_split()
    resolved = validate_dev_split(split, PROJECT_ROOT)
    if len(resolved.dev_val_images) != 313 or set(split.dev_val_cities) != {"jena", "krefeld", "ulm"}:
        raise RuntimeError("Dev-validation identity differs from the locked split")
    official_train = (PROJECT_ROOT / "leftImg8bit" / "train").resolve()
    if any(not path.resolve().is_relative_to(official_train) for path in resolved.dev_val_images):
        raise RuntimeError("A dev-validation image is outside official train")

    device = torch.device("cuda")
    model_01, _ = checkpoint_model(BASELINE_01_DIR, 26, saved["baseline_01"], device)
    model_02, metadata_02 = checkpoint_model(BASELINE_02_DIR, 28, saved["baseline_02"], device)
    dataset = CityscapesSegmentationDataset(PROJECT_ROOT, resolved.dev_val_images, make_val_transform())
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    matrices, city_matrices, image_rows, boundary_counts = evaluate_models(
        {"baseline_01": model_01, "baseline_02": model_02}, dataset, loader, device
    )
    summaries = verify_matrices(matrices, saved)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAFFIC_DIR.mkdir(parents=True, exist_ok=True)
    names = get_train_id_to_name()
    ranked = write_traffic_tables(matrices, image_rows, names)
    write_regression_tables(matrices, summaries, names)
    write_rare_stealing_matrix(matrices, names)
    city_rows = write_city_comparison(city_matrices, names)
    boundary_rows = write_boundary_diagnostic(boundary_counts, names)
    milder = milder_weights(names)
    selected = select_traffic_examples(ranked)
    examples = save_traffic_examples(model_02, dataset, selected, device, names)
    plot_fp_sources(matrices["baseline_02"], names)
    runtime = time.perf_counter() - started
    concentration = make_report(metadata_02, summaries, matrices, ranked, city_rows, boundary_rows, milder, examples, runtime, names)

    traffic = summaries["baseline_02"]["diagnostics"][TRAFFIC_LIGHT]
    payload = {
        "checkpoint_epoch": metadata_02.epoch,
        "metrics": {key: summaries["baseline_02"][key] for key in ("miou", "mean_dice", "pixel_accuracy")},
        "saved_confusion_matrix_exact_match": True,
        "traffic_light": traffic,
        "traffic_light_concentration": concentration,
        "milder_weight_diagnostic": milder,
        "selected_examples": examples,
        "official_val_test_loaded": False,
        "runtime_seconds": runtime,
        "recommended_single_experiment": "normalized frequency^-0.25 class weights",
        "training_started": False,
    }
    save_json(OUTPUT_DIR / "analysis_metrics.json", payload)
    print(json.dumps({
        "complete": True, "epoch": metadata_02.epoch,
        "miou": summaries["baseline_02"]["miou"], "traffic_light": traffic,
        "top10_fp_share": concentration["top10_fp_share"], "runtime_seconds": runtime,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
