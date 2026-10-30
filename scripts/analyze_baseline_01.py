"""Reproducible dev-validation failure analysis for locked Baseline 01."""

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
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from torch import nn
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
from cityscapes_segmentation.data.splits import load_dev_split, validate_dev_split  # noqa: E402
from cityscapes_segmentation.data.transforms import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    make_val_transform,
)
from cityscapes_segmentation.engine import load_checkpoint  # noqa: E402
from cityscapes_segmentation.models import UNet  # noqa: E402


EXPERIMENT = "baseline_01_unet_ce"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / EXPERIMENT / "best.pt"
BASELINE_METRICS = PROJECT_ROOT / "outputs" / EXPERIMENT / "best_metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / EXPERIMENT / "error_analysis"
EXAMPLES_DIR = OUTPUT_DIR / "examples"
WEAK_CLASSES = (12, 17, 16, 14, 15)  # rider, motorcycle, train, truck, bus
RARE_CLASSES = (12, 17, 16, 14, 15, 6)
COMPARE_COMMON = (0, 2, 8, 10, 13)
COMPARE_SMALL = (12, 17, 6, 5, 18)
NUM_CLASSES = 19


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def _format_metric(value: float) -> str:
    return "NaN" if math.isnan(value) else f"{value:.6f}"


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_checkpoint(model: nn.Module, expected: dict, device: torch.device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    metadata = load_checkpoint(
        CHECKPOINT,
        model=model,
        optimizer=optimizer,
        map_location=device,
    )
    required_config = {
        "model": "UNet",
        "in_channels": 3,
        "num_classes": 19,
        "base_channels": 64,
        "val_transform": "deterministic_baseline",
    }
    mismatches = {
        key: (metadata.config.get(key), value)
        for key, value in required_config.items()
        if metadata.config.get(key) != value
    }
    if metadata.epoch != 26:
        raise RuntimeError(f"Expected best checkpoint epoch 26; got {metadata.epoch}")
    if not math.isclose(metadata.best_miou, float(expected["best_miou"]), abs_tol=1e-12):
        raise RuntimeError("Stored checkpoint best mIoU differs from the Baseline 01 result")
    if mismatches:
        raise RuntimeError(f"Checkpoint model/config mismatch: {mismatches}")
    return metadata


def select_examples(occurrences: dict[int, list[int]], limit: int = 8) -> list[int]:
    """Select occurrence quantiles without consulting predictions or errors."""
    selected: list[int] = []
    for fraction in (0.5, 0.25, 0.75):
        for class_id in WEAK_CLASSES:
            choices = occurrences[class_id]
            if not choices:
                continue
            position = min(len(choices) - 1, int(fraction * len(choices)))
            index = choices[position]
            if index not in selected:
                selected.append(index)
            if len(selected) == limit:
                return sorted(selected)
    return sorted(selected)


def infer(model, loader, device):
    metrics = SegmentationMetrics()
    occurrences: dict[int, list[int]] = defaultdict(list)
    sample_index = 0
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1)
            metrics.update(predictions, targets)
            for target in targets:
                counts = torch.bincount(target[target != get_ignore_index()], minlength=NUM_CLASSES)
                for class_id in WEAK_CLASSES:
                    if int(counts[class_id]) > 0:
                        occurrences[class_id].append(sample_index)
                sample_index += 1
    return metrics.compute(), occurrences


def validate_metrics(result, expected: dict) -> None:
    checks = {
        "mIoU": (result.miou, float(expected["best_miou"])),
        "mean Dice": (result.mean_dice, float(expected["mean_dice"])),
        "pixel accuracy": (result.pixel_accuracy, float(expected["pixel_accuracy"])),
    }
    failures = {
        name: values
        for name, values in checks.items()
        if not math.isclose(values[0], values[1], rel_tol=1e-9, abs_tol=1e-10)
    }
    expected_matrix = torch.tensor(expected["confusion_matrix"], dtype=torch.int64)
    if not torch.equal(result.confusion_matrix, expected_matrix):
        difference = int((result.confusion_matrix - expected_matrix).abs().sum())
        failures["confusion_matrix_absolute_difference"] = difference
    if failures:
        raise RuntimeError(f"Independent evaluation does not reproduce Baseline 01: {failures}")


def diagnostic_rows(matrix: np.ndarray, names: dict[int, str]) -> list[dict]:
    total = int(matrix.sum())
    rows = []
    for class_id in range(NUM_CLASSES):
        gt = int(matrix[class_id].sum())
        predicted = int(matrix[:, class_id].sum())
        tp = int(matrix[class_id, class_id])
        fp = predicted - tp
        fn = gt - tp
        precision = _safe_divide(tp, predicted)
        recall = _safe_divide(tp, gt)
        iou = _safe_divide(tp, tp + fp + fn)
        dice = _safe_divide(2 * tp, 2 * tp + fp + fn)
        rows.append({
            "train_id": class_id,
            "class_name": names[class_id],
            "gt_pixels": gt,
            "predicted_pixels": predicted,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "dice": dice,
            "gt_pixel_share": _safe_divide(gt, total),
            "predicted_pixel_share": _safe_divide(predicted, total),
            "prediction_ratio": _safe_divide(predicted, gt),
        })
    return rows


def confusion_rows(matrix: np.ndarray, names: dict[int, str]) -> list[dict]:
    rows = []
    for gt_id in range(NUM_CLASSES):
        support = int(matrix[gt_id].sum())
        destinations = [
            (pred_id, int(matrix[gt_id, pred_id]))
            for pred_id in range(NUM_CLASSES)
            if pred_id != gt_id
        ]
        destinations.sort(key=lambda item: (-item[1], item[0]))
        for rank, (pred_id, count) in enumerate(destinations[:5], start=1):
            rows.append({
                "gt_train_id": gt_id,
                "gt_class": names[gt_id],
                "rank": rank,
                "predicted_train_id": pred_id,
                "predicted_class": names[pred_id],
                "pixels": count,
                "percentage_of_gt": 100.0 * _safe_divide(count, support),
            })
    return rows


def write_csvs(diagnostics: list[dict], confusions: list[dict], matrix: np.ndarray, names: dict[int, str]) -> None:
    with (OUTPUT_DIR / "per_class_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=diagnostics[0].keys())
        writer.writeheader()
        writer.writerows(diagnostics)
    with (OUTPUT_DIR / "confusion_destinations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=confusions[0].keys())
        writer.writeheader()
        writer.writerows(confusions)
    with (OUTPUT_DIR / "rare_class_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "train_id", "class_name", "gt_share", "iou", "precision", "recall",
            "prediction_ratio", "top_3_confusions",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for class_id in RARE_CLASSES:
            row = diagnostics[class_id]
            destinations = sorted(
                ((pred, int(matrix[class_id, pred])) for pred in range(NUM_CLASSES) if pred != class_id),
                key=lambda item: (-item[1], item[0]),
            )[:3]
            top = "; ".join(
                f"{names[pred]}: {100 * _safe_divide(count, row['gt_pixels']):.2f}%"
                for pred, count in destinations
            )
            writer.writerow({
                "train_id": class_id,
                "class_name": names[class_id],
                "gt_share": row["gt_pixel_share"],
                "iou": row["iou"],
                "precision": row["precision"],
                "recall": row["recall"],
                "prediction_ratio": row["prediction_ratio"],
                "top_3_confusions": top,
            })


def plot_confusion(matrix: np.ndarray, names: dict[int, str]) -> None:
    labels = [names[index] for index in range(NUM_CLASSES)]
    figure, axis = plt.subplots(figsize=(14, 12))
    positive = np.ma.masked_where(matrix == 0, matrix)
    image = axis.imshow(positive, cmap="viridis", norm=LogNorm(vmin=1, vmax=max(1, int(matrix.max()))))
    figure.colorbar(image, ax=axis, label="Pixel count (log scale)")
    axis.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES), xticklabels=labels, yticklabels=labels)
    axis.tick_params(axis="x", rotation=60, labelsize=8)
    axis.tick_params(axis="y", labelsize=8)
    axis.set(xlabel="Predicted class", ylabel="Ground-truth class", title="Baseline 01 confusion matrix — counts")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix_counts.png", dpi=180)
    plt.close(figure)

    supports = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, supports, out=np.zeros_like(matrix, dtype=np.float64), where=supports > 0)
    figure, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(normalized, cmap="magma", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, label="Share of ground-truth row")
    axis.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES), xticklabels=labels, yticklabels=labels)
    axis.tick_params(axis="x", rotation=60, labelsize=8)
    axis.tick_params(axis="y", labelsize=8)
    axis.set(xlabel="Predicted class", ylabel="Ground-truth class", title="Baseline 01 confusion matrix — row normalized")
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            value = normalized[row, column]
            if value >= 0.01 or row == column:
                axis.text(column, row, f"{100 * value:.0f}", ha="center", va="center", fontsize=4.5,
                          color="white" if value < 0.65 else "black")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix_normalized.png", dpi=180)
    plt.close(figure)


def colorize(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < NUM_CLASSES)
    output[valid] = palette[mask[valid]]
    return output


def save_examples(model, dataset, selected: list[int], device, names: dict[int, str]) -> list[dict]:
    palette = np.asarray([label.color for label in get_train_id_classes()], dtype=np.uint8)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    details = []
    model.eval()
    with torch.inference_mode():
        for index in selected:
            image, target = dataset[index]
            prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu()
            rgb = ((image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            target_np = target.numpy()
            prediction_np = prediction.numpy()
            path = dataset.samples[index][0]
            present = [class_id for class_id in WEAK_CLASSES if np.any(target_np == class_id)]
            filename = f"{index:03d}_{path.stem.removesuffix('_leftImg8bit')}.png"
            figure, axes = plt.subplots(1, 3, figsize=(16, 5))
            axes[0].imshow(rgb)
            axes[0].set_title("RGB")
            axes[1].imshow(colorize(target_np, palette))
            axes[1].set_title("Ground truth")
            axes[2].imshow(colorize(prediction_np, palette))
            axes[2].set_title("Prediction")
            for axis in axes:
                axis.axis("off")
            figure.suptitle(f"{path.parent.name}/{path.name} | weak GT: {', '.join(names[c] for c in present)}", fontsize=10)
            figure.tight_layout()
            figure.savefig(EXAMPLES_DIR / filename, dpi=150)
            plt.close(figure)
            details.append({"dataset_index": index, "image": str(path.relative_to(PROJECT_ROOT)),
                            "weak_classes_present": [names[class_id] for class_id in present], "file": filename})

            for class_id in (12, 17, 16):
                if class_id not in present:
                    continue
                gt = target_np == class_id
                pred = prediction_np == class_id
                overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
                overlay[gt & ~pred] = (230, 70, 70)
                overlay[pred & ~gt] = (70, 130, 230)
                overlay[gt & pred] = (70, 210, 100)
                figure, axis = plt.subplots(figsize=(9, 4.5))
                axis.imshow(rgb)
                axis.imshow(overlay, alpha=np.where(gt | pred, 0.75, 0.0))
                axis.axis("off")
                axis.set_title(f"{names[class_id]} overlay — {path.name}", fontsize=9)
                axis.legend(handles=(Patch(color=(0.90, 0.27, 0.27), label="GT only"),
                                     Patch(color=(0.27, 0.51, 0.90), label="Prediction only"),
                                     Patch(color=(0.27, 0.82, 0.39), label="Overlap")), loc="lower center", ncol=3)
                figure.tight_layout()
                overlay_name = filename.removesuffix(".png") + f"_{names[class_id]}_overlay.png"
                figure.savefig(EXAMPLES_DIR / overlay_name, dpi=150)
                plt.close(figure)
    return details


def top_destinations(matrix: np.ndarray, class_id: int, names: dict[int, str], count: int = 3) -> list[tuple[str, int, float]]:
    support = int(matrix[class_id].sum())
    values = sorted(
        ((pred, int(matrix[class_id, pred])) for pred in range(NUM_CLASSES) if pred != class_id),
        key=lambda item: (-item[1], item[0]),
    )[:count]
    return [(names[pred], pixels, 100 * _safe_divide(pixels, support)) for pred, pixels in values]


def make_report(metadata, result, diagnostics, matrix, names, examples, runtime: float) -> None:
    strongest = sorted(diagnostics, key=lambda row: row["iou"], reverse=True)[:5]
    weakest = sorted(diagnostics, key=lambda row: row["iou"])[:5]
    under = sorted(
        (row for row in diagnostics if row["prediction_ratio"] < 1.0),
        key=lambda row: row["prediction_ratio"],
    )[:5]
    over = sorted(
        (row for row in diagnostics if row["prediction_ratio"] > 1.0),
        key=lambda row: row["prediction_ratio"],
        reverse=True,
    )[:5]
    rider_targets = (11, 18, 17, 0, 8)

    lines = [
        "# Baseline 01 Dev-Validation Failure Analysis",
        "",
        "## Checkpoint and evaluation identity",
        "",
        f"- Checkpoint: `{CHECKPOINT.relative_to(PROJECT_ROOT)}`",
        f"- Epoch: {metadata.epoch}",
        f"- Stored best mIoU: {metadata.best_miou:.9f}",
        "- Model: `UNet(in_channels=3, num_classes=19, base_channels=64)`",
        "- Evaluation set: all 313 development-validation images from jena, krefeld, and ulm",
        "- Transform: deterministic Baseline 01 validation transform at 512×256",
        "- Official Cityscapes val/test loaded: no",
        f"- Analysis runtime: {runtime:.2f} seconds",
        "",
        "## Recomputed headline metrics",
        "",
        f"- mIoU: {result.miou:.9f}",
        f"- Mean Dice: {result.mean_dice:.9f}",
        f"- Pixel accuracy: {result.pixel_accuracy:.9f}",
        "",
        "The independently accumulated confusion matrix exactly matches the saved Baseline 01 matrix.",
        "",
        "## Strongest and weakest classes",
        "",
        "| Group | Class | IoU | Precision | Recall |",
        "|---|---|---:|---:|---:|",
    ]
    for group, rows in (("Strong", strongest), ("Weak", weakest)):
        lines.extend(f"| {group} | {row['class_name']} | {row['iou']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |" for row in rows)

    lines.extend([
        "",
        "## Rare-class diagnostics",
        "",
        "| Class | GT share | IoU | Precision | Recall | Pred/GT | Top three incorrect destinations |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for class_id in RARE_CLASSES:
        row = diagnostics[class_id]
        destinations = ", ".join(f"{name} {percentage:.2f}%" for name, _, percentage in top_destinations(matrix, class_id, names))
        lines.append(
            f"| {names[class_id]} | {100 * row['gt_pixel_share']:.3f}% | {row['iou']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['prediction_ratio']:.3f} | {destinations} |"
        )

    lines.extend(["", "## Major confusion patterns", ""])
    for class_id in (12, 17, 16, 14, 15, 6, 3, 4, 5):
        destinations = ", ".join(
            f"{name}: {pixels:,} px ({percentage:.2f}%)"
            for name, pixels, percentage in top_destinations(matrix, class_id, names, count=5)
        )
        lines.append(f"- **{names[class_id]}:** {destinations}.")

    rider_support = diagnostics[12]["gt_pixels"]
    lines.extend(["", "## Rider hypothesis check", ""])
    for target_id in rider_targets:
        pixels = int(matrix[12, target_id])
        lines.append(f"- rider → {names[target_id]}: {pixels:,} pixels ({100 * _safe_divide(pixels, rider_support):.3f}%)")
    other = top_destinations(matrix, 12, names, count=8)
    lines.append("- Ranked major incorrect destinations: " + ", ".join(f"{name} {percentage:.2f}%" for name, _, percentage in other) + ".")

    lines.extend([
        "",
        "## Prediction-frequency bias",
        "",
        "Most under-predicted: " + ", ".join(f"{row['class_name']} ({row['prediction_ratio']:.3f}×)" for row in under) + ".",
        "",
        "Most over-predicted: " + ", ".join(f"{row['class_name']} ({row['prediction_ratio']:.3f}×)" for row in over) + ".",
        "",
        "A predicted/GT ratio measures aggregate frequency, not segmentation quality or calibration.",
        "",
        "## Cautious diagnosis",
        "",
        "**Classification: C — evidence is consistent with both class imbalance and spatial/detail loss.**",
        "",
        "The rare classes occupy very small fractions of valid pixels and several are strongly under-predicted, which is consistent with an optimization imbalance toward common classes. Thin and small classes also lose pixels to visually or spatially adjacent categories such as person, building, car, and bicycle. That pattern is consistent with detail loss at 512×256, but pixel-level confusion cannot establish object-size or resolution causality by itself.",
        "",
        "For context, the common-class mean IoU (road, building, vegetation, sky, car) is "
        f"{np.mean([diagnostics[c]['iou'] for c in COMPARE_COMMON]):.4f}; the selected rare/thin-class mean IoU "
        f"(rider, motorcycle, traffic light, pole, bicycle) is {np.mean([diagnostics[c]['iou'] for c in COMPARE_SMALL]):.4f}.",
        "",
        "## Recommended Baseline 02 controlled experiment",
        "",
        "Test **weighted CrossEntropyLoss** as the single change, while keeping the architecture, resolution, split, optimizer, scheduler, augmentation, and epoch count fixed. The severe under-prediction and very low recall of rare classes make loss reweighting the most direct first test of the class-imbalance hypothesis. A higher-resolution experiment remains a separate later test if reweighting does not recover thin-class recall.",
        "",
        "## Qualitative examples",
        "",
        "Examples were selected deterministically by occurrence quantiles among images containing weak classes, without ranking by error severity.",
        "",
    ])
    lines.extend(f"- `{item['file']}`: {', '.join(item['weak_classes_present'])}" for item in examples)
    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline 01 analysis requires the verified CUDA environment")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    expected = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))
    split = load_dev_split()
    resolved = validate_dev_split(split, PROJECT_ROOT)
    if len(resolved.dev_val_images) != 313 or set(split.dev_val_cities) != {"jena", "krefeld", "ulm"}:
        raise RuntimeError("Development validation identity differs from Baseline 01")
    official_train = (PROJECT_ROOT / "leftImg8bit" / "train").resolve()
    if any(not path.resolve().is_relative_to(official_train) for path in resolved.dev_val_images):
        raise RuntimeError("Development validation path escapes official train")

    dataset = CityscapesSegmentationDataset(PROJECT_ROOT, resolved.dev_val_images, make_val_transform())
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    device = torch.device("cuda")
    model = UNet(in_channels=3, num_classes=19, base_channels=64).to(device)
    metadata = verify_checkpoint(model, expected, device)
    result, occurrences = infer(model, loader, device)
    validate_metrics(result, expected)

    names = get_train_id_to_name()
    matrix = result.confusion_matrix.numpy()
    diagnostics = diagnostic_rows(matrix, names)
    confusions = confusion_rows(matrix, names)
    write_csvs(diagnostics, confusions, matrix, names)
    plot_confusion(matrix, names)
    selected = select_examples(occurrences)
    examples = save_examples(model, dataset, selected, device, names)
    runtime = time.perf_counter() - started
    make_report(metadata, result, diagnostics, matrix, names, examples, runtime)

    payload = {
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": metadata.epoch,
        "stored_best_miou": metadata.best_miou,
        "dev_val_images": len(dataset),
        "dev_val_cities": list(split.dev_val_cities),
        "miou": result.miou,
        "mean_dice": result.mean_dice,
        "pixel_accuracy": result.pixel_accuracy,
        "confusion_matrix_matches_baseline": True,
        "official_val_test_loaded": False,
        "selected_examples": examples,
        "runtime_seconds": runtime,
    }
    _atomic_json(OUTPUT_DIR / "analysis_metrics.json", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
