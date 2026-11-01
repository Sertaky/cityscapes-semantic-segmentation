"""Run locked Baseline 04: Baseline 03 with resolution 640x320 only."""

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
import torch.nn.functional as functional
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

import run_baseline_03 as b03
from cityscapes_segmentation.data.labels import get_train_id_classes
from cityscapes_segmentation.data.transforms import SegmentationTransform


baseline = b03.baseline
EXPERIMENT = "baseline_04_higher_resolution"
OUTPUT_DIR = baseline.PROJECT_ROOT / "outputs" / EXPERIMENT
CHECKPOINT_DIR = baseline.PROJECT_ROOT / "checkpoints" / EXPERIMENT
BASELINE_03_DIR = baseline.PROJECT_ROOT / "outputs" / b03.EXPERIMENT
RESOLUTION = (640, 320)  # PIL width, height.
FOCUS_CLASSES = (12, 17, 16, 15)
DETAIL_CLASSES = (12, 17, 16, 15, 6, 18, 5, 7)
EXPECTED_WEIGHTS = np.asarray([
    0.3869842578274517, 0.6079823926680971, 0.43724451196116954,
    1.084465648466904, 0.9878559473645069, 0.9029786920036802,
    1.4004418972643393, 1.1023838899848257, 0.4777135228092011,
    0.921796426669748, 0.6742800565735951, 0.9028496133230156,
    1.5772224654114801, 0.584177119034281, 1.3242343757123156,
    1.3748846809554693, 1.3674338294241042, 1.7080506639523498,
    1.1770200085934652,
], dtype=np.float64)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def locked_baseline_03_weights(image_paths: tuple[Path, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, never recompute, the exact reviewed Baseline 03 weights."""
    if len(image_paths) != 2662:
        raise RuntimeError(f"Expected 2662 dev-train paths; got {len(image_paths)}")
    config = read_json(BASELINE_03_DIR / "config.json")
    weights = np.asarray(config["class_weights"], dtype=np.float64)
    counts = np.asarray(config["class_pixel_counts"], dtype=np.int64)
    frequencies = np.asarray(config["class_frequencies"], dtype=np.float64)
    if config["weight_source"] != "dev_train_only" or config["weighting_exponent"] != 0.25:
        raise RuntimeError("Baseline 03 config does not identify the locked dev-train exponent-0.25 weights")
    if not np.array_equal(weights, EXPECTED_WEIGHTS):
        raise RuntimeError("Saved Baseline 03 weights differ from the locked 19-value vector")
    if not (
        weights.shape == (19,)
        and np.all(np.isfinite(weights))
        and np.all(weights > 0)
        and math.isclose(float(weights.mean()), 1.0, abs_tol=1e-12)
    ):
        raise RuntimeError("Baseline 03 weights failed shape/finite/positive/mean checks")
    return counts, frequencies, weights.copy()


def make_train_transform():
    return SegmentationTransform(size=RESOLUTION, horizontal_flip_probability=0.5)


def make_val_transform():
    return SegmentationTransform(size=RESOLUTION)


def add_matrix(
    matrix: np.ndarray,
    target: torch.Tensor | np.ndarray,
    prediction: torch.Tensor | np.ndarray,
) -> None:
    """Accumulate rows=target, columns=prediction while excluding ignore pixels."""
    target = torch.as_tensor(target, dtype=torch.int64).reshape(-1)
    prediction = torch.as_tensor(prediction, dtype=torch.int64).reshape(-1)
    valid = target != baseline.get_ignore_index()
    encoded = target[valid] * 19 + prediction[valid]
    matrix += torch.bincount(encoded, minlength=19 * 19).reshape(19, 19).cpu().numpy()


def configuration(split, train_transform, counts, frequencies, weights) -> dict:
    config = b03.configuration(split, train_transform, counts, frequencies, weights)
    config.update({
        "experiment_name": EXPERIMENT,
        "resolution": {"width": RESOLUTION[0], "height": RESOLUTION[1]},
        "resolution_width": RESOLUTION[0],
        "resolution_height": RESOLUTION[1],
        "train_count": 2662,
        "val_count": 313,
        "weight_source": "baseline_03_dev_train_only_locked",
        "weighting_exponent": 0.25,
        "class_weights": [float(value) for value in weights],
        "train_transform": {
            "name": "baseline_at_640x320",
            "size": list(RESOLUTION),
            "image_interpolation": "bilinear",
            "mask_interpolation": "nearest",
            "horizontal_flip_probability": train_transform.horizontal_flip_probability,
        },
        "val_transform": {"name": "deterministic_baseline_at_640x320", "size": list(RESOLUTION)},
        "only_change_from_baseline_03": "Input resolution 512x256 -> 640x320",
    })
    config.pop("only_change_from_baseline_02", None)
    return config


def prepare_fresh(config: dict) -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError("Baseline 04 output directory already exists; inspect before running")
    if CHECKPOINT_DIR.exists():
        raise FileExistsError("Baseline 04 checkpoint directory already exists; inspect before running")
    OUTPUT_DIR.mkdir(parents=True)
    CHECKPOINT_DIR.mkdir(parents=True)
    baseline.save_json(OUTPUT_DIR / "config.json", config)
    with (OUTPUT_DIR / "history.csv").open("x", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=baseline.HISTORY_FIELDS).writeheader()
        stream.flush()


def plot_histories(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    specifications = (
        ("loss_curve.png", "Weighted CE loss", (("Train", "train_loss"), ("Dev validation", "val_loss"))),
        ("miou_curve.png", "Dev-validation mIoU", (("mIoU", "val_miou"),)),
        ("dice_curve.png", "Dev-validation mean Dice", (("Mean Dice", "val_mean_dice"),)),
        ("lr_curve.png", "Learning rate", (("AdamW LR", "learning_rate"),)),
    )
    for filename, ylabel, series in specifications:
        figure, axis = plt.subplots(figsize=(7, 4))
        for label, key in series:
            axis.plot(epochs, [float(row[key]) for row in rows], label=label)
        axis.set(xlabel="Epoch", ylabel=ylabel, xlim=(1, 30))
        axis.grid(alpha=0.3)
        if len(series) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close(figure)
    prior = baseline.read_history(BASELINE_03_DIR / "history.csv")
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, [float(row["val_miou"]) for row in prior], label="Baseline 03: 512x256")
    axis.plot(epochs, [float(row["val_miou"]) for row in rows], label="Baseline 04: 640x320")
    axis.set(xlabel="Epoch", ylabel="Dev-validation mIoU", xlim=(1, 30))
    axis.grid(alpha=0.3); axis.legend(); figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "miou_comparison_b03_b04.png", dpi=150)
    plt.close(figure)


def top_confusions(matrix: np.ndarray, class_id: int, names: dict[int, str], direction: str) -> list[dict]:
    if direction == "fn_destination":
        denominator = int(matrix[class_id].sum())
        values = ((other, int(matrix[class_id, other])) for other in range(19) if other != class_id)
    elif direction == "fp_source":
        denominator = int(matrix[:, class_id].sum() - matrix[class_id, class_id])
        values = ((other, int(matrix[other, class_id])) for other in range(19) if other != class_id)
    else:
        raise ValueError(direction)
    return [
        {"class": names[other], "pixels": pixels, "percentage": 100 * pixels / denominator if denominator else math.nan}
        for other, pixels in sorted(values, key=lambda item: (-item[1], item[0]))[:5]
    ]


def write_comparison(best: dict, names: dict[int, str]) -> dict:
    metrics03 = read_json(BASELINE_03_DIR / "best_metrics.json")
    old = metrics03["per_class"]
    new = best["per_class"]
    rows = []
    for class_id in range(19):
        row = {"train_id": class_id, "class_name": names[class_id]}
        for metric in ("iou", "precision", "recall", "prediction_ratio"):
            row.update({
                f"b03_{metric}": old[class_id][metric],
                f"b04_{metric}": new[class_id][metric],
                f"delta_{metric}": new[class_id][metric] - old[class_id][metric],
            })
        rows.append(row)
    write_csv(OUTPUT_DIR / "comparison_vs_b03.csv", rows)
    old_matrix = np.asarray(metrics03["confusion_matrix"], dtype=np.int64)
    new_matrix = np.asarray(best["confusion_matrix"], dtype=np.int64)
    confusions = {
        names[class_id]: {
            model: {
                direction: top_confusions(matrix, class_id, names, direction)
                for direction in ("fn_destination", "fp_source")
            }
            for model, matrix in (("baseline_03", old_matrix), ("baseline_04", new_matrix))
        }
        for class_id in FOCUS_CLASSES
    }
    comparison = {
        "baseline_03": {
            "best_epoch": metrics03["best_epoch"], "miou": metrics03["best_miou"],
            "mean_dice": metrics03["mean_dice"], "pixel_accuracy": metrics03["pixel_accuracy"],
        },
        "baseline_04": {
            "best_epoch": best["best_epoch"], "miou": best["best_miou"],
            "mean_dice": best["mean_dice"], "pixel_accuracy": best["pixel_accuracy"],
        },
        "deltas": {
            "miou": best["best_miou"] - metrics03["best_miou"],
            "mean_dice": best["mean_dice"] - metrics03["mean_dice"],
            "pixel_accuracy": best["pixel_accuracy"] - metrics03["pixel_accuracy"],
        },
        "focus_confusions": confusions,
        "official_val_test_loaded": False,
    }
    baseline.save_json(OUTPUT_DIR / "comparison_summary.json", comparison)
    return comparison


def install_locked_overrides() -> None:
    b03.install_locked_overrides()
    baseline.EXPERIMENT = EXPERIMENT
    baseline.OUTPUT_DIR = OUTPUT_DIR
    baseline.CHECKPOINT_DIR = CHECKPOINT_DIR
    baseline.recompute_verified_weights = locked_baseline_03_weights
    baseline.configuration = configuration
    baseline.prepare_fresh = prepare_fresh
    baseline.make_train_transform = make_train_transform
    baseline.make_val_transform = make_val_transform
    baseline.write_comparison = write_comparison
    baseline.plot_histories = plot_histories


def split_and_datasets():
    split = baseline.load_dev_split()
    resolved = baseline.validate_dev_split(split, baseline.PROJECT_ROOT)
    if len(resolved.dev_train_images) != 2662 or len(resolved.dev_val_images) != 313:
        raise RuntimeError("Development split differs from locked 2662/313")
    train = baseline.CityscapesSegmentationDataset(
        baseline.PROJECT_ROOT, resolved.dev_train_images, make_train_transform()
    )
    val = baseline.CityscapesSegmentationDataset(
        baseline.PROJECT_ROOT, resolved.dev_val_images, make_val_transform()
    )
    return resolved, train, val


def smoke_check() -> dict:
    baseline.seed_everything(42)
    resolved, train_dataset, val_dataset = split_and_datasets()
    _, _, weights = locked_baseline_03_weights(resolved.dev_train_images)
    loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)
    images, targets = next(iter(loader))
    expected_images, expected_targets = (2, 3, 320, 640), (2, 320, 640)
    if tuple(images.shape) != expected_images or tuple(targets.shape) != expected_targets:
        raise RuntimeError(f"Smoke batch shape mismatch: {tuple(images.shape)}, {tuple(targets.shape)}")
    device = torch.device("cuda")
    model = baseline.UNet(in_channels=3, num_classes=19, base_channels=64).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=baseline.get_ignore_index())
    images, targets = images.to(device), targets.to(device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, targets)
    loss.backward()
    gradients_exist = any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    optimizer.step(); torch.cuda.synchronize()
    result = {
        "train_dataset_length": len(train_dataset), "val_dataset_length": len(val_dataset),
        "images_shape": list(images.shape), "targets_shape": list(targets.shape),
        "logits_shape": list(logits.shape), "loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss)), "gradients_exist": gradients_exist,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "weights_exact_baseline_03": bool(np.array_equal(weights, EXPECTED_WEIGHTS)),
        "weight_count": len(weights), "weight_mean": float(weights.mean()),
    }
    if not result["loss_finite"] or not gradients_exist or tuple(logits.shape) != (2, 19, 320, 640):
        raise RuntimeError(f"Smoke training step failed: {result}")
    return result


def class_boundary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    tensor = torch.from_numpy(mask).to(torch.float32)[None, None]
    kernel = 2 * radius + 1
    eroded = 1.0 - functional.max_pool2d(1.0 - tensor, kernel_size=kernel, stride=1, padding=radius)
    return mask & ~(eroded[0, 0].numpy() > 0.5)


def component_coverages(mask: np.ndarray, predictions: dict[str, np.ndarray]) -> list[dict]:
    """Measure both models on identical 4-connected original-resolution components."""
    height, width = mask.shape
    remaining = set(np.flatnonzero(mask).tolist())
    flattened = {name: prediction.ravel() for name, prediction in predictions.items()}
    components = []
    while remaining:
        first = remaining.pop(); stack = [first]; area = 0
        overlaps = {name: 0 for name in predictions}
        while stack:
            index = stack.pop(); area += 1
            for name, prediction in flattened.items():
                overlaps[name] += int(prediction[index])
            row, column = divmod(index, width)
            for candidate, allowed in (
                (index - width, row > 0), (index + width, row + 1 < height),
                (index - 1, column > 0), (index + 1, column + 1 < width),
            ):
                if allowed and candidate in remaining:
                    remaining.remove(candidate); stack.append(candidate)
        components.append({"area_original": area, **{f"overlap_{name}": value for name, value in overlaps.items()}})
    return components


def load_model(experiment: str, epoch: int, device: torch.device):
    model = baseline.UNet(in_channels=3, num_classes=19, base_channels=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    metadata = baseline.load_checkpoint(
        baseline.PROJECT_ROOT / "checkpoints" / experiment / "best.pt",
        model=model, optimizer=optimizer, map_location=device,
    )
    if metadata.epoch != epoch:
        raise RuntimeError(f"Expected {experiment} epoch {epoch}; got {metadata.epoch}")
    model.eval()
    return model


def resolution_diagnostics() -> dict:
    """Compare B03 and B04 detail behavior without changing either checkpoint."""
    names = baseline.get_train_id_to_name()
    metrics03 = read_json(BASELINE_03_DIR / "best_metrics.json")
    metrics04 = read_json(OUTPUT_DIR / "best_metrics.json")
    device = torch.device("cuda")
    model04 = load_model(EXPERIMENT, int(metrics04["best_epoch"]), device)
    model03 = load_model(b03.EXPERIMENT, int(metrics03["best_epoch"]), device)
    resolved = baseline.validate_dev_split(baseline.load_dev_split(), baseline.PROJECT_ROOT)
    dataset04 = baseline.CityscapesSegmentationDataset(
        baseline.PROJECT_ROOT, resolved.dev_val_images, SegmentationTransform(size=RESOLUTION)
    )
    dataset03 = baseline.CityscapesSegmentationDataset(
        baseline.PROJECT_ROOT, resolved.dev_val_images, SegmentationTransform(size=(512, 256))
    )
    city_matrices = defaultdict(lambda: np.zeros((19, 19), dtype=np.int64))
    matrix04 = np.zeros((19, 19), dtype=np.int64)
    boundary04 = {class_id: {region: [0, 0] for region in ("boundary", "interior")} for class_id in FOCUS_CLASSES}
    with torch.inference_mode():
        loader04 = DataLoader(dataset04, batch_size=2, shuffle=False, num_workers=0)
        offset = 0
        for images, targets in loader04:
            predictions = model04(images.to(device)).argmax(1).cpu().numpy().astype(np.uint8)
            targets_np = targets.numpy().astype(np.uint8)
            for batch_index, (target, prediction) in enumerate(zip(targets_np, predictions, strict=True)):
                city = dataset04.samples[offset + batch_index][0].parent.name
                add_matrix(matrix04, target, prediction)
                add_matrix(city_matrices[city], target, prediction)
                for class_id in FOCUS_CLASSES:
                    gt = target == class_id; band = class_boundary(gt)
                    for region, selection in (("boundary", band), ("interior", gt & ~band)):
                        boundary04[class_id][region][0] += int(selection.sum())
                        boundary04[class_id][region][1] += int((selection & (prediction != class_id)).sum())
            offset += len(targets_np)
    expected_matrix = np.asarray(metrics04["confusion_matrix"], dtype=np.int64)
    if not np.array_equal(matrix04, expected_matrix):
        raise RuntimeError("B04 diagnostic confusion matrix differs from clean best evaluation")

    b03_boundary = {
        (row["class_name"], row["region"]): row
        for row in csv.DictReader((BASELINE_03_DIR / "error_analysis/boundary_diagnostic.csv").open(encoding="utf-8"))
    }
    boundary_rows = []
    for class_id in FOCUS_CLASSES:
        for region in ("boundary", "interior"):
            pixels, errors = boundary04[class_id][region]
            old_rate = float(b03_boundary[(names[class_id], region)]["error_rate"])
            new_rate = errors / pixels if pixels else math.nan
            boundary_rows.append({
                "train_id": class_id, "class_name": names[class_id], "region": region,
                "b03_error_rate": old_rate, "b04_error_rate": new_rate,
                "delta_error_rate": new_rate - old_rate,
                "b04_pixels": pixels, "b04_error_pixels": errors,
                "note": "Same two-pixel inward boundary heuristic at each experiment tensor resolution",
            })
    write_csv(OUTPUT_DIR / "boundary_comparison_vs_b03.csv", boundary_rows)

    old_city = {
        (row["city"], row["class_name"]): row
        for row in csv.DictReader((BASELINE_03_DIR / "error_analysis/city_focus_metrics.csv").open(encoding="utf-8"))
    }
    city_rows = []
    for city in ("jena", "krefeld", "ulm"):
        diagnostics = baseline.diagnostic_rows(city_matrices[city], names)
        for class_id in FOCUS_CLASSES:
            current = diagnostics[class_id]; prior = old_city[(city, names[class_id])]
            city_rows.append({
                "city": city, "train_id": class_id, "class_name": names[class_id],
                "b03_precision": prior["precision"], "b04_precision": current["precision"],
                "delta_precision": current["precision"] - float(prior["precision"]),
                "b03_recall": prior["recall"], "b04_recall": current["recall"],
                "delta_recall": current["recall"] - float(prior["recall"]),
                "b03_iou": prior["iou"], "b04_iou": current["iou"],
                "delta_iou": current["iou"] - float(prior["iou"]),
            })
    write_csv(OUTPUT_DIR / "city_focus_comparison_vs_b03.csv", city_rows)

    raw_ids = {label.trainId: label.id for label in get_train_id_classes()}
    component_rows = []
    with torch.inference_mode():
        for index, ((image03, _), (image04, _)) in enumerate(zip(dataset03, dataset04, strict=True)):
            mask_path = dataset03.samples[index][1]
            with Image.open(mask_path) as source:
                raw = np.asarray(source)
            present = [class_id for class_id in FOCUS_CLASSES if np.any(raw == raw_ids[class_id])]
            if not present:
                continue
            prediction03 = model03(image03.unsqueeze(0).to(device)).argmax(1, keepdim=True).to(torch.float32)
            prediction04 = model04(image04.unsqueeze(0).to(device)).argmax(1, keepdim=True).to(torch.float32)
            size = raw.shape
            prediction03 = functional.interpolate(prediction03, size=size, mode="nearest")[0, 0].cpu().numpy().astype(np.uint8)
            prediction04 = functional.interpolate(prediction04, size=size, mode="nearest")[0, 0].cpu().numpy().astype(np.uint8)
            for class_id in present:
                records = component_coverages(raw == raw_ids[class_id], {
                    "b03": prediction03 == class_id, "b04": prediction04 == class_id,
                })
                for component_index, record in enumerate(records, start=1):
                    component_rows.append({
                        "dataset_index": index, "city": dataset03.samples[index][0].parent.name,
                        "image_id": dataset03.samples[index][0].name.removesuffix("_leftImg8bit.png"),
                        "train_id": class_id, "class_name": names[class_id],
                        "component_index": component_index, **record,
                    })
    size_rows = []
    for class_id in FOCUS_CLASSES:
        selected = [row for row in component_rows if row["train_id"] == class_id]
        areas = np.asarray([row["area_original"] for row in selected], dtype=np.float64)
        low, high = np.quantile(areas, (1 / 3, 2 / 3))
        for row in selected:
            row["size_bin"] = "small" if row["area_original"] <= low else "medium" if row["area_original"] <= high else "large"
        for size_bin in ("small", "medium", "large"):
            group = [row for row in selected if row["size_bin"] == size_bin]
            pixels = sum(row["area_original"] for row in group)
            old_overlap = sum(row["overlap_b03"] for row in group)
            new_overlap = sum(row["overlap_b04"] for row in group)
            size_rows.append({
                "train_id": class_id, "class_name": names[class_id], "size_bin": size_bin,
                "binning_scale": "original_2048x1024_gt_component_area_tertiles",
                "small_threshold_max_original_pixels": low,
                "medium_threshold_max_original_pixels": high,
                "component_count": len(group), "gt_pixels_original": pixels,
                "b03_recall_proxy": old_overlap / pixels, "b04_recall_proxy": new_overlap / pixels,
                "delta_recall_proxy": (new_overlap - old_overlap) / pixels,
            })
    write_csv(OUTPUT_DIR / "small_region_recall_comparison.csv", size_rows)

    comparison = read_json(OUTPUT_DIR / "comparison_summary.json")
    delta_miou = comparison["deltas"]["miou"]
    focus_detail_delta = float(np.mean([
        row["delta_recall_proxy"] for row in size_rows if row["size_bin"] in {"small", "medium"}
    ]))
    boundary_delta = float(np.mean([row["delta_error_rate"] for row in boundary_rows if row["region"] == "boundary"]))
    if delta_miou > 0.01 and focus_detail_delta > 0 and boundary_delta < 0:
        outcome = "A_clear_resolution_win"
    elif delta_miou > 0:
        outcome = "B_modest_improvement_compute_tradeoff"
    elif abs(delta_miou) <= 0.002:
        outcome = "C_neutral"
    elif focus_detail_delta > 0 and boundary_delta < 0:
        outcome = "E_mixed_detail_improves_aggregate_does_not"
    else:
        outcome = "D_regression"
    current_best = EXPERIMENT if delta_miou > 0 else b03.EXPERIMENT
    recommendation = (
        "Analyze residual semantic confusions before another controlled experiment"
        if delta_miou > 0 else
        "Retain Baseline 03 and inspect why added resolution did not improve dev-validation mIoU"
    )
    payload = {
        "outcome_classification": outcome,
        "current_best_development_model": current_best,
        "small_medium_mean_recall_delta": focus_detail_delta,
        "mean_boundary_error_delta": boundary_delta,
        "recommended_next_investigation": recommendation,
        "official_val_test_loaded": False,
    }
    baseline.save_json(OUTPUT_DIR / "resolution_diagnostics_summary.json", payload)
    return payload


def update_experiment_summary(diagnostics: dict) -> None:
    summary = read_json(OUTPUT_DIR / "summary.json")
    baseline_03_runtime = read_json(BASELINE_03_DIR / "summary.json")["total_epoch_wall_seconds"]
    summary.update({
        "resolution_diagnostics": diagnostics,
        "baseline_03_runtime_seconds": baseline_03_runtime,
        "runtime_increase_percent_vs_b03": 100 * (
            summary["total_epoch_wall_seconds"] / baseline_03_runtime - 1
        ),
    })
    baseline.save_json(OUTPUT_DIR / "summary.json", summary)


def run_training(resume: bool) -> None:
    install_locked_overrides()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *( ["--resume"] if resume else [] )]
        baseline.main()
    finally:
        sys.argv = original_argv
    diagnostics = resolution_diagnostics()
    update_experiment_summary(diagnostics)
    print(json.dumps({"post_training_diagnostics_complete": True, **diagnostics}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    install_locked_overrides()
    if args.smoke_only:
        print(json.dumps(smoke_check(), indent=2))
    elif args.analysis_only:
        diagnostics = resolution_diagnostics()
        update_experiment_summary(diagnostics)
        print(json.dumps(diagnostics, indent=2))
    else:
        run_training(args.resume)


if __name__ == "__main__":
    main()
