"""Run locked Baseline 05: Baseline 04 with bottleneck dilation 2 only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import run_baseline_04 as b04
from cityscapes_segmentation.data.transforms import SegmentationTransform
from cityscapes_segmentation.models import UNet as ProjectUNet


baseline = b04.baseline
EXPERIMENT = "baseline_05_dilated_bottleneck"
OUTPUT_DIR = baseline.PROJECT_ROOT / "outputs" / EXPERIMENT
CHECKPOINT_DIR = baseline.PROJECT_ROOT / "checkpoints" / EXPERIMENT
BASELINE_04_DIR = baseline.PROJECT_ROOT / "outputs" / b04.EXPERIMENT
PARAMETER_COUNT = 31_038_803
FOCUS_CLASSES = (14, 16, 15, 6)  # truck, train, bus, traffic light


def dilated_unet(*, in_channels: int = 3, num_classes: int = 19, base_channels: int = 64):
    """Construct the single locked architectural change for Baseline 05."""
    return ProjectUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        bottleneck_dilation=2,
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def configuration(split, train_transform, counts, frequencies, weights) -> dict:
    config = b04.configuration(split, train_transform, counts, frequencies, weights)
    config.update({
        "experiment_name": EXPERIMENT,
        "model": "UNet",
        "bottleneck_dilation": 2,
        "bottleneck_padding": 2,
        "bottleneck_effective_kernel": "5x5",
        "parameter_count": PARAMETER_COUNT,
        "only_change_from_baseline_04": (
            "Both 3x3 bottleneck convolutions: dilation 1 -> 2 and padding 1 -> 2"
        ),
    })
    config.pop("only_change_from_baseline_03", None)
    return config


def prepare_fresh(config: dict) -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError("Baseline 05 output directory already exists; inspect before running")
    if CHECKPOINT_DIR.exists():
        raise FileExistsError("Baseline 05 checkpoint directory already exists; inspect before running")
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

    prior = baseline.read_history(BASELINE_04_DIR / "history.csv")
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, [float(row["val_miou"]) for row in prior], label="Baseline 04: standard bottleneck")
    axis.plot(epochs, [float(row["val_miou"]) for row in rows], label="Baseline 05: dilation 2")
    axis.set(xlabel="Epoch", ylabel="Dev-validation mIoU", xlim=(1, 30))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "miou_comparison_b04_b05.png", dpi=150)
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
        {
            "train_id": other,
            "class_name": names[other],
            "pixels": pixels,
            "percentage": 100 * pixels / denominator if denominator else math.nan,
        }
        for other, pixels in sorted(values, key=lambda item: (-item[1], item[0]))[:5]
    ]


def write_comparison(best: dict, names: dict[int, str]) -> dict:
    old_metrics = read_json(BASELINE_04_DIR / "best_metrics.json")
    old, new = old_metrics["per_class"], best["per_class"]
    rows = []
    for class_id in range(19):
        row = {"train_id": class_id, "class_name": names[class_id]}
        for metric in ("iou", "precision", "recall", "prediction_ratio"):
            row.update({
                f"b04_{metric}": old[class_id][metric],
                f"b05_{metric}": new[class_id][metric],
                f"delta_{metric}": new[class_id][metric] - old[class_id][metric],
            })
        rows.append(row)
    write_csv(OUTPUT_DIR / "comparison_vs_b04.csv", rows)

    old_matrix = np.asarray(old_metrics["confusion_matrix"], dtype=np.int64)
    new_matrix = np.asarray(best["confusion_matrix"], dtype=np.int64)
    confusions = {
        names[class_id]: {
            model: {
                direction: top_confusions(matrix, class_id, names, direction)
                for direction in ("fn_destination", "fp_source")
            }
            for model, matrix in (("baseline_04", old_matrix), ("baseline_05", new_matrix))
        }
        for class_id in FOCUS_CLASSES
    }
    comparison = {
        "baseline_04": {
            "best_epoch": old_metrics["best_epoch"],
            "miou": old_metrics["best_miou"],
            "mean_dice": old_metrics["mean_dice"],
            "pixel_accuracy": old_metrics["pixel_accuracy"],
        },
        "baseline_05": {
            "best_epoch": best["best_epoch"],
            "miou": best["best_miou"],
            "mean_dice": best["mean_dice"],
            "pixel_accuracy": best["pixel_accuracy"],
        },
        "deltas": {
            "miou": best["best_miou"] - old_metrics["best_miou"],
            "mean_dice": best["mean_dice"] - old_metrics["mean_dice"],
            "pixel_accuracy": best["pixel_accuracy"] - old_metrics["pixel_accuracy"],
        },
        "focus_confusions": confusions,
        "official_val_test_loaded": False,
    }
    baseline.save_json(OUTPUT_DIR / "comparison_summary.json", comparison)
    return comparison


def install_locked_overrides() -> None:
    b04.install_locked_overrides()
    baseline.EXPERIMENT = EXPERIMENT
    baseline.OUTPUT_DIR = OUTPUT_DIR
    baseline.CHECKPOINT_DIR = CHECKPOINT_DIR
    baseline.UNet = dilated_unet
    baseline.recompute_verified_weights = b04.locked_baseline_03_weights
    baseline.configuration = configuration
    baseline.prepare_fresh = prepare_fresh
    baseline.make_train_transform = b04.make_train_transform
    baseline.make_val_transform = b04.make_val_transform
    baseline.write_comparison = write_comparison
    baseline.plot_histories = plot_histories


def split_and_datasets():
    split = baseline.load_dev_split()
    resolved = baseline.validate_dev_split(split, baseline.PROJECT_ROOT)
    if len(resolved.dev_train_images) != 2662 or len(resolved.dev_val_images) != 313:
        raise RuntimeError("Development split differs from locked 2662/313")
    official_train = (baseline.PROJECT_ROOT / "leftImg8bit" / "train").resolve()
    if any(
        not path.resolve().is_relative_to(official_train)
        for path in (*resolved.dev_train_images, *resolved.dev_val_images)
    ):
        raise RuntimeError("A development sample is outside official train")
    train = baseline.CityscapesSegmentationDataset(
        baseline.PROJECT_ROOT, resolved.dev_train_images, b04.make_train_transform()
    )
    val = baseline.CityscapesSegmentationDataset(
        baseline.PROJECT_ROOT, resolved.dev_val_images, b04.make_val_transform()
    )
    return resolved, train, val


def smoke_check() -> dict:
    baseline.seed_everything(42)
    resolved, train_dataset, val_dataset = split_and_datasets()
    _, _, weights = b04.locked_baseline_03_weights(resolved.dev_train_images)
    images, targets = next(iter(DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)))
    if tuple(images.shape) != (2, 3, 320, 640) or tuple(targets.shape) != (2, 320, 640):
        raise RuntimeError(f"Smoke batch shape mismatch: {tuple(images.shape)}, {tuple(targets.shape)}")
    device = torch.device("cuda")
    model = dilated_unet().to(device).train()
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("Dilated model parameter count differs from the locked standard U-Net")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        ignore_index=baseline.get_ignore_index(),
    )
    images, targets = images.to(device), targets.to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, targets)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradients_finite = bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "train_dataset_length": len(train_dataset),
        "val_dataset_length": len(val_dataset),
        "images_shape": list(images.shape),
        "targets_shape": list(targets.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": gradients_finite,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "b04_peak_allocated_mib": read_json(BASELINE_04_DIR / "summary.json")["peak_cuda_allocated_mib"],
        "b04_peak_reserved_mib": read_json(BASELINE_04_DIR / "summary.json")["peak_cuda_reserved_mib"],
        "weights_exact_baseline_03": bool(np.array_equal(weights, b04.EXPECTED_WEIGHTS)),
    }
    if not (
        result["loss_finite"]
        and gradients_finite
        and tuple(logits.shape) == (2, 19, 320, 640)
    ):
        raise RuntimeError(f"Smoke training step failed: {result}")
    return result


def matrix_for(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    matrix = np.zeros((19, 19), dtype=np.int64)
    b04.add_matrix(matrix, target, prediction)
    return matrix


def class_stats(matrix: np.ndarray, class_id: int) -> dict:
    true_positive = int(matrix[class_id, class_id])
    support = int(matrix[class_id].sum())
    predicted = int(matrix[:, class_id].sum())
    false_positive = predicted - true_positive
    false_negative = support - true_positive

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else math.nan

    return {
        "support": support,
        "predicted_pixels": predicted,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": ratio(true_positive, predicted),
        "recall": ratio(true_positive, support),
        "iou": ratio(true_positive, true_positive + false_positive + false_negative),
        "prediction_ratio": ratio(predicted, support),
    }


def matrix_metrics(matrix: np.ndarray) -> dict:
    statistics = [class_stats(matrix, class_id) for class_id in range(19)]
    ious = [item["iou"] for item in statistics if math.isfinite(item["iou"])]
    dices = []
    for item in statistics:
        denominator = 2 * item["true_positives"] + item["false_positives"] + item["false_negatives"]
        if denominator:
            dices.append(2 * item["true_positives"] / denominator)
    total = int(matrix.sum())
    return {
        "miou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "pixel_accuracy": float(np.trace(matrix) / total) if total else math.nan,
    }


def load_model(experiment: str, *, dilation: int, device: torch.device):
    model = ProjectUNet(bottleneck_dilation=dilation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    metadata = baseline.load_checkpoint(
        baseline.PROJECT_ROOT / "checkpoints" / experiment / "best.pt",
        model=model,
        optimizer=optimizer,
        map_location=device,
    )
    if metadata.config.get("experiment_name") != experiment:
        raise RuntimeError(f"Checkpoint experiment identity mismatch for {experiment}")
    if int(metadata.config.get("bottleneck_dilation", 1)) != dilation:
        raise RuntimeError(f"Checkpoint bottleneck dilation mismatch for {experiment}")
    model.eval()
    return model, metadata


def infer(model, dataset, device: torch.device):
    matrix = np.zeros((19, 19), dtype=np.int64)
    image_matrices = []
    predictions = []
    with torch.inference_mode():
        for images, targets in DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0):
            batch_predictions = model(images.to(device)).argmax(1).cpu().numpy().astype(np.uint8)
            for target, prediction in zip(targets.numpy().astype(np.uint8), batch_predictions, strict=True):
                item = matrix_for(target, prediction)
                matrix += item
                image_matrices.append(item)
                predictions.append(prediction)
    return matrix, np.stack(image_matrices), predictions


def bootstrap(image_matrices04: np.ndarray, image_matrices05: np.ndarray, samples: int = 1000):
    rng = np.random.default_rng(42)
    requested = {"overall_miou": None, "truck_iou": 14, "train_iou": 16, "bus_iou": 15, "traffic_light_iou": 6}
    values = {key: [] for key in requested}
    for _ in range(samples):
        counts = np.bincount(rng.integers(0, len(image_matrices04), len(image_matrices04)), minlength=len(image_matrices04))
        matrix04 = np.tensordot(counts, image_matrices04, axes=(0, 0))
        matrix05 = np.tensordot(counts, image_matrices05, axes=(0, 0))
        metrics04 = matrix_metrics(matrix04)
        metrics05 = matrix_metrics(matrix05)
        values["overall_miou"].append(metrics05["miou"] - metrics04["miou"])
        for key, class_id in requested.items():
            if class_id is not None:
                values[key].append(class_stats(matrix05, class_id)["iou"] - class_stats(matrix04, class_id)["iou"])
    rows = []
    names = baseline.get_train_id_to_name()
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
            "note": "Image-bootstrap diagnostic; not a formal benchmark confidence interval",
        })
    return rows


def post_training_analysis() -> dict:
    device = torch.device("cuda")
    model04, metadata04 = load_model(b04.EXPERIMENT, dilation=1, device=device)
    model05, metadata05 = load_model(EXPERIMENT, dilation=2, device=device)
    resolved = baseline.validate_dev_split(baseline.load_dev_split(), baseline.PROJECT_ROOT)
    dataset = baseline.CityscapesSegmentationDataset(
        baseline.PROJECT_ROOT,
        resolved.dev_val_images,
        SegmentationTransform(size=b04.RESOLUTION),
    )
    matrix04, image_matrices04, predictions04 = infer(model04, dataset, device)
    matrix05, image_matrices05, predictions05 = infer(model05, dataset, device)
    saved04 = np.asarray(read_json(BASELINE_04_DIR / "best_metrics.json")["confusion_matrix"], dtype=np.int64)
    saved05 = np.asarray(read_json(OUTPUT_DIR / "best_metrics.json")["confusion_matrix"], dtype=np.int64)
    if not np.array_equal(matrix04, saved04) or not np.array_equal(matrix05, saved05):
        raise RuntimeError("Post-run confusion matrix differs from a clean saved best-checkpoint evaluation")

    names = baseline.get_train_id_to_name()
    city04 = defaultdict(lambda: np.zeros((19, 19), dtype=np.int64))
    city05 = defaultdict(lambda: np.zeros((19, 19), dtype=np.int64))
    scene_rows = []
    for index, ((image_path, _), item04, item05) in enumerate(
        zip(dataset.samples, image_matrices04, image_matrices05, strict=True)
    ):
        city = image_path.parent.name
        city04[city] += item04
        city05[city] += item05
        for class_id in (14, 16, 15):
            old, new = class_stats(item04, class_id), class_stats(item05, class_id)
            old_error = old["false_positives"] + old["false_negatives"]
            new_error = new["false_positives"] + new["false_negatives"]
            if old["support"] or new["support"] or old["predicted_pixels"] or new["predicted_pixels"]:
                scene_rows.append({
                    "dataset_index": index,
                    "city": city,
                    "image_id": image_path.name.removesuffix("_leftImg8bit.png"),
                    "train_id": class_id,
                    "class_name": names[class_id],
                    "b04_gt_pixels": old["support"],
                    "b05_gt_pixels": new["support"],
                    "b04_tp": old["true_positives"],
                    "b05_tp": new["true_positives"],
                    "b04_fp": old["false_positives"],
                    "b05_fp": new["false_positives"],
                    "b04_fn": old["false_negatives"],
                    "b05_fn": new["false_negatives"],
                    "error_reduction": old_error - new_error,
                })
    write_csv(OUTPUT_DIR / "scene_class_deltas.csv", scene_rows)

    concentration_rows = []
    for class_id in (14, 16, 15):
        selected = [row for row in scene_rows if row["train_id"] == class_id]
        positive = sorted((max(0, row["error_reduction"]) for row in selected), reverse=True)
        negative = sorted((max(0, -row["error_reduction"]) for row in selected), reverse=True)
        gross_positive, gross_negative = sum(positive), sum(negative)
        row = {
            "train_id": class_id,
            "class_name": names[class_id],
            "images_with_gt": sum(item["b04_gt_pixels"] > 0 for item in selected),
            "images_improved": sum(item["error_reduction"] > 0 for item in selected),
            "images_regressed": sum(item["error_reduction"] < 0 for item in selected),
            "net_error_reduction": sum(item["error_reduction"] for item in selected),
        }
        for count in (1, 3, 5):
            row[f"top_{count}_positive_share"] = sum(positive[:count]) / gross_positive if gross_positive else math.nan
            row[f"top_{count}_negative_share"] = sum(negative[:count]) / gross_negative if gross_negative else math.nan
        concentration_rows.append(row)
    write_csv(OUTPUT_DIR / "scene_concentration_vs_b04.csv", concentration_rows)

    city_rows = []
    for city in ("jena", "krefeld", "ulm"):
        aggregate04 = matrix_metrics(city04[city])
        aggregate05 = matrix_metrics(city05[city])
        for class_id in FOCUS_CLASSES:
            old, new = class_stats(city04[city], class_id), class_stats(city05[city], class_id)
            city_rows.append({
                "city": city,
                "city_b04_miou": aggregate04["miou"],
                "city_b05_miou": aggregate05["miou"],
                "city_delta_miou": aggregate05["miou"] - aggregate04["miou"],
                "train_id": class_id,
                "class_name": names[class_id],
                "b04_iou": old["iou"],
                "b05_iou": new["iou"],
                "delta_iou": new["iou"] - old["iou"],
                "b04_precision": old["precision"],
                "b05_precision": new["precision"],
                "b04_recall": old["recall"],
                "b05_recall": new["recall"],
            })
    write_csv(OUTPUT_DIR / "city_comparison_vs_b04.csv", city_rows)

    bootstrap_rows = bootstrap(image_matrices04, image_matrices05)
    write_csv(OUTPUT_DIR / "bootstrap_diagnostics_vs_b04.csv", bootstrap_rows)
    confusions = {
        names[class_id]: {
            model_name: {
                direction: top_confusions(matrix, class_id, names, direction)
                for direction in ("fn_destination", "fp_source")
            }
            for model_name, matrix in (("baseline_04", matrix04), ("baseline_05", matrix05))
        }
        for class_id in FOCUS_CLASSES
    }
    baseline.save_json(OUTPUT_DIR / "focus_confusions_vs_b04.json", confusions)

    summary = read_json(OUTPUT_DIR / "summary.json")
    comparison = read_json(OUTPUT_DIR / "comparison_summary.json")
    runtime04 = read_json(BASELINE_04_DIR / "summary.json")["total_epoch_wall_seconds"]
    analysis = {
        "checkpoint_verification": {
            "baseline_04_epoch": metadata04.epoch,
            "baseline_05_epoch": metadata05.epoch,
            "baseline_05_dilation": metadata05.config["bottleneck_dilation"],
            "confusion_matrices_exact": True,
        },
        "headline": comparison,
        "city_comparison": city_rows,
        "scene_concentration": concentration_rows,
        "bootstrap": bootstrap_rows,
        "focus_confusions": confusions,
        "runtime_delta_percent_vs_b04": 100 * (summary["total_epoch_wall_seconds"] / runtime04 - 1),
        "official_val_test_loaded": False,
    }
    baseline.save_json(OUTPUT_DIR / "analysis_summary.json", analysis)
    summary.update({
        "baseline_04_runtime_seconds": runtime04,
        "runtime_delta_percent_vs_b04": analysis["runtime_delta_percent_vs_b04"],
        "post_training_analysis_complete": True,
        "official_val_test_loaded": False,
    })
    baseline.save_json(OUTPUT_DIR / "summary.json", summary)
    return analysis


def run_training(resume: bool) -> None:
    install_locked_overrides()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *(["--resume"] if resume else [])]
        baseline.main()
    finally:
        sys.argv = original_argv
    analysis = post_training_analysis()
    print(json.dumps({"post_training_analysis_complete": True, **analysis}, indent=2), flush=True)


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
        print(json.dumps(post_training_analysis(), indent=2))
    else:
        run_training(args.resume)


if __name__ == "__main__":
    main()
