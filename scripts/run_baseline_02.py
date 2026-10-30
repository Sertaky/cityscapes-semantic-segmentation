"""Run locked Baseline 02: Baseline 01 with weighted cross-entropy only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_baseline_01 import diagnostic_rows, top_destinations  # noqa: E402
from compute_class_weights import count_dev_train_pixels  # noqa: E402
from run_baseline_01 import (  # noqa: E402
    HISTORY_FIELDS,
    append_history,
    read_history,
    restore_rng_state,
    rng_state,
    save_json,
)
from train_baseline import seed_everything  # noqa: E402
from cityscapes_segmentation.analysis.class_weights import compute_class_weights  # noqa: E402
from cityscapes_segmentation.analysis.metrics import SegmentationMetrics  # noqa: E402
from cityscapes_segmentation.data.dataset import CityscapesSegmentationDataset  # noqa: E402
from cityscapes_segmentation.data.labels import get_ignore_index, get_train_id_to_name  # noqa: E402
from cityscapes_segmentation.data.splits import load_dev_split, validate_dev_split  # noqa: E402
from cityscapes_segmentation.data.transforms import make_train_transform, make_val_transform  # noqa: E402
from cityscapes_segmentation.engine import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)
from cityscapes_segmentation.models import UNet  # noqa: E402


EXPERIMENT = "baseline_02_weighted_ce"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / EXPERIMENT
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / EXPERIMENT
REVIEWED_WEIGHTS_PATH = OUTPUT_DIR / "class_weights.json"
BASELINE_01_DIR = PROJECT_ROOT / "outputs" / "baseline_01_unet_ce"
PRIOR_WEIGHT_FILES = {"class_weights.json", "class_weights.csv"}
PRIORITY_CLASSES = (12, 17, 16, 14, 15, 6)
CONFUSION_CLASSES = (12, 17, 16, 14, 15)


def recompute_verified_weights(image_paths: tuple[Path, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reviewed = json.loads(REVIEWED_WEIGHTS_PATH.read_text(encoding="utf-8"))
    if reviewed["source"]["image_count"] != 2662 or reviewed["source"]["split"] != "dev_train":
        raise ValueError("Reviewed class-weight artifact does not identify the 2662-image dev-train split")
    counts, ignored_pixels, total_pixels = count_dev_train_pixels(image_paths)
    result = compute_class_weights(counts)
    reviewed_counts = np.asarray([row["pixel_count"] for row in reviewed["classes"]], dtype=np.int64)
    reviewed_weights = np.asarray([row["normalized_weight"] for row in reviewed["classes"]], dtype=np.float64)
    if not np.array_equal(counts, reviewed_counts):
        raise RuntimeError("Recomputed dev-train pixel counts differ from the reviewed calculation")
    if not np.array_equal(result.sqrt_inverse_normalized, reviewed_weights):
        raise RuntimeError("Recomputed full-precision weights differ from the reviewed calculation")
    if ignored_pixels != reviewed["pixels"]["ignored"] or total_pixels != reviewed["pixels"]["total"]:
        raise RuntimeError("Recomputed dev-train total/ignored pixels differ from the reviewed calculation")
    weights = result.sqrt_inverse_normalized
    if not (
        weights.shape == (19,)
        and np.all(np.isfinite(weights))
        and np.all(weights > 0)
        and math.isclose(float(weights.mean()), 1.0, abs_tol=1e-12)
    ):
        raise RuntimeError("Locked class weights failed sanity checks")
    return counts, result.frequencies, weights


def configuration(split, train_transform, counts, frequencies, weights) -> dict:
    return {
        "experiment_name": EXPERIMENT,
        "seed": 42,
        "model": "UNet",
        "in_channels": 3,
        "base_channels": 64,
        "num_classes": 19,
        "resolution": {"width": 512, "height": 256},
        "train_split": {"source": "official_train", "cities": list(split.dev_train_cities), "images": 2662},
        "val_split": {"source": "official_train", "cities": list(split.dev_val_cities), "images": 313},
        "batch_size": 2,
        "epochs": 30,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "T_max": 30,
        "eta_min": 1e-5,
        "loss": "CrossEntropyLoss",
        "loss_type": "weighted_cross_entropy",
        "weighting_method": "normalized_inverse_sqrt_frequency",
        "weight_source": "dev_train_only",
        "class_pixel_counts": [int(value) for value in counts],
        "class_frequencies": [float(value) for value in frequencies],
        "class_weights": [float(value) for value in weights],
        "ignore_index": get_ignore_index(),
        "precision": "FP32",
        "selection_metric": "dev_val_miou",
        "early_stopping": False,
        "train_transform": {"name": "baseline", "horizontal_flip_probability": train_transform.horizontal_flip_probability},
        "val_transform": "deterministic_baseline",
        "num_workers": 0,
        "num_workers_note": "Conservative Windows setting; no multiprocessing workers",
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "cuda_build": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "python_version": platform.python_version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "only_change_from_baseline_01": "CrossEntropyLoss class weights",
    }


def prepare_fresh(config: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unexpected = {item.name for item in OUTPUT_DIR.iterdir()} - PRIOR_WEIGHT_FILES
    if unexpected:
        raise FileExistsError(f"Baseline 02 output directory contains unexpected prior artifacts: {sorted(unexpected)}")
    if CHECKPOINT_DIR.exists():
        raise FileExistsError("Baseline 02 checkpoint directory already exists; inspect before using --resume")
    CHECKPOINT_DIR.mkdir(parents=True)
    save_json(OUTPUT_DIR / "config.json", config)
    with (OUTPUT_DIR / "history.csv").open("x", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writeheader()
        stream.flush()


def prepare_resume(config, model, optimizer, scheduler, train_generator) -> tuple[int, float]:
    saved_config = json.loads((OUTPUT_DIR / "config.json").read_text(encoding="utf-8"))
    if saved_config != config:
        raise ValueError("Saved Baseline 02 config differs from the locked recomputed config")
    rows = read_history(OUTPUT_DIR / "history.csv")
    metadata = load_checkpoint(
        CHECKPOINT_DIR / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location="cpu",
    )
    if [int(row["epoch"]) for row in rows] != list(range(1, metadata.epoch + 1)):
        raise ValueError("History epochs do not match last.pt")
    if not (CHECKPOINT_DIR / "best.pt").is_file():
        raise FileNotFoundError("Resume requires best.pt")
    if not math.isclose(float(rows[-1]["best_miou"]), metadata.best_miou, abs_tol=1e-12):
        raise ValueError("History and last.pt disagree on best mIoU")
    if metadata.extra_state is None:
        raise ValueError("last.pt lacks RNG state required for exact resume")
    restore_rng_state(metadata.extra_state, train_generator)
    if any(
        isinstance(value, torch.Tensor) and key != "step" and value.device.type != "cuda"
        for state in optimizer.state.values()
        for key, value in state.items()
    ):
        raise RuntimeError("Restored optimizer moment state is not on CUDA")
    return metadata.epoch + 1, metadata.best_miou


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

    baseline_rows = read_history(BASELINE_01_DIR / "history.csv")
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, [float(row["val_miou"]) for row in baseline_rows], label="Baseline 01 plain CE")
    axis.plot(epochs, [float(row["val_miou"]) for row in rows], label="Baseline 02 weighted CE")
    axis.set(xlabel="Epoch", ylabel="Dev-validation mIoU", xlim=(1, 30))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "miou_comparison_vs_baseline_01.png", dpi=150)
    plt.close(figure)


def baseline_01_diagnostics(names: dict[int, str]) -> tuple[dict, list[dict]]:
    metrics = json.loads((BASELINE_01_DIR / "best_metrics.json").read_text(encoding="utf-8"))
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    diagnostics_path = BASELINE_01_DIR / "error_analysis" / "per_class_diagnostics.csv"
    if diagnostics_path.is_file():
        with diagnostics_path.open(newline="", encoding="utf-8") as stream:
            saved = list(csv.DictReader(stream))
        diagnostics = [
            {
                key: int(value) if key in {"train_id", "gt_pixels", "predicted_pixels", "true_positives", "false_positives", "false_negatives"}
                else value if key == "class_name"
                else float(value)
                for key, value in row.items()
            }
            for row in saved
        ]
        recomputed = diagnostic_rows(matrix, names)
        if any(not math.isclose(diagnostics[i]["iou"], recomputed[i]["iou"], abs_tol=1e-12) for i in range(19)):
            raise RuntimeError("Baseline 01 error-analysis diagnostics disagree with its confusion matrix")
    else:
        diagnostics = diagnostic_rows(matrix, names)
    return metrics, diagnostics


def clean_best_evaluation(val_loader, criterion, device, best_miou: float, names: dict[int, str]) -> dict:
    model = UNet(in_channels=3, num_classes=19, base_channels=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    metadata = load_checkpoint(CHECKPOINT_DIR / "best.pt", model=model, optimizer=optimizer, map_location="cpu")
    result = validate_one_epoch(model, val_loader, criterion, device, SegmentationMetrics())
    if not math.isclose(result.miou, best_miou, rel_tol=1e-6, abs_tol=1e-8):
        raise RuntimeError("Clean best-checkpoint evaluation differs from saved best mIoU")
    matrix = result.confusion_matrix.numpy()
    diagnostics = diagnostic_rows(matrix, names)
    with (OUTPUT_DIR / "best_per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "train_id", "class_name", "precision", "recall", "iou", "dice", "support",
            "predicted_pixels", "prediction_gt_ratio",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in diagnostics:
            writer.writerow({
                "train_id": row["train_id"], "class_name": row["class_name"],
                "precision": row["precision"], "recall": row["recall"], "iou": row["iou"],
                "dice": row["dice"], "support": row["gt_pixels"],
                "predicted_pixels": row["predicted_pixels"], "prediction_gt_ratio": row["prediction_ratio"],
            })
    payload = {
        "best_epoch": metadata.epoch,
        "best_miou": result.miou,
        "val_loss": result.val_loss,
        "mean_dice": result.mean_dice,
        "pixel_accuracy": result.pixel_accuracy,
        "per_class": diagnostics,
        "confusion_matrix": result.confusion_matrix.tolist(),
    }
    save_json(OUTPUT_DIR / "best_metrics.json", payload)
    return payload


def write_comparison(best: dict, names: dict[int, str]) -> dict:
    baseline_metrics, baseline = baseline_01_diagnostics(names)
    current = best["per_class"]
    fields = (
        "train_id", "class_name", "baseline_01_iou", "baseline_02_iou", "delta_iou",
        "baseline_01_recall", "baseline_02_recall", "delta_recall",
        "baseline_01_precision", "baseline_02_precision", "delta_precision",
    )
    with (OUTPUT_DIR / "comparison_vs_baseline_01.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for class_id in range(19):
            old, new = baseline[class_id], current[class_id]
            writer.writerow({
                "train_id": class_id, "class_name": names[class_id],
                "baseline_01_iou": old["iou"], "baseline_02_iou": new["iou"],
                "delta_iou": new["iou"] - old["iou"],
                "baseline_01_recall": old["recall"], "baseline_02_recall": new["recall"],
                "delta_recall": new["recall"] - old["recall"],
                "baseline_01_precision": old["precision"], "baseline_02_precision": new["precision"],
                "delta_precision": new["precision"] - old["precision"],
            })

    old_matrix = np.asarray(baseline_metrics["confusion_matrix"], dtype=np.int64)
    new_matrix = np.asarray(best["confusion_matrix"], dtype=np.int64)
    confusions = {}
    for class_id in CONFUSION_CLASSES:
        confusions[names[class_id]] = {
            "baseline_01": [
                {"destination": destination, "pixels": pixels, "percentage": percentage}
                for destination, pixels, percentage in top_destinations(old_matrix, class_id, names, count=5)
            ],
            "baseline_02": [
                {"destination": destination, "pixels": pixels, "percentage": percentage}
                for destination, pixels, percentage in top_destinations(new_matrix, class_id, names, count=5)
            ],
        }
    comparison = {
        "baseline_01": {
            "best_epoch": baseline_metrics["best_epoch"], "miou": baseline_metrics["best_miou"],
            "mean_dice": baseline_metrics["mean_dice"], "pixel_accuracy": baseline_metrics["pixel_accuracy"],
        },
        "baseline_02": {
            "best_epoch": best["best_epoch"], "miou": best["best_miou"],
            "mean_dice": best["mean_dice"], "pixel_accuracy": best["pixel_accuracy"],
        },
        "deltas": {
            "miou": best["best_miou"] - baseline_metrics["best_miou"],
            "mean_dice": best["mean_dice"] - baseline_metrics["mean_dice"],
            "pixel_accuracy": best["pixel_accuracy"] - baseline_metrics["pixel_accuracy"],
        },
        "priority_classes": {
            names[class_id]: {
                "baseline_01": baseline[class_id], "baseline_02": current[class_id],
                "delta_iou": current[class_id]["iou"] - baseline[class_id]["iou"],
                "delta_recall": current[class_id]["recall"] - baseline[class_id]["recall"],
                "delta_precision": current[class_id]["precision"] - baseline[class_id]["precision"],
                "delta_prediction_ratio": current[class_id]["prediction_ratio"] - baseline[class_id]["prediction_ratio"],
            }
            for class_id in PRIORITY_CLASSES
        },
        "confusion_destinations": confusions,
        "hypothesis_recorded_before_training": "Weighted CE may increase rare-class recall, possibly at the cost of precision.",
    }
    save_json(OUTPUT_DIR / "comparison_summary.json", comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked Baseline 02 weighted-CE Cityscapes experiment")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Locked Baseline 02 requires CUDA")

    split = load_dev_split()
    resolved = validate_dev_split(split, PROJECT_ROOT)
    if len(resolved.dev_train_images) != 2662 or len(resolved.dev_val_images) != 313:
        raise RuntimeError("Development split differs from locked 2662/313")
    official_train = (PROJECT_ROOT / "leftImg8bit" / "train").resolve()
    if any(not path.resolve().is_relative_to(official_train) for path in (*resolved.dev_train_images, *resolved.dev_val_images)):
        raise RuntimeError("A development sample is outside official train")
    if any(path.parent.name in set(split.dev_val_cities) for path in resolved.dev_train_images):
        raise RuntimeError("A dev-validation city entered dev train")

    counts, frequencies, weights = recompute_verified_weights(resolved.dev_train_images)
    seed_everything(42)
    train_transform = make_train_transform()
    val_transform = make_val_transform()
    config = configuration(split, train_transform, counts, frequencies, weights)
    train_dataset = CityscapesSegmentationDataset(PROJECT_ROOT, resolved.dev_train_images, train_transform)
    val_dataset = CityscapesSegmentationDataset(PROJECT_ROOT, resolved.dev_val_images, val_transform)
    train_generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, generator=train_generator)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    device = torch.device("cuda")
    locked_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    model = UNet(in_channels=3, num_classes=19, base_channels=64).to(device)
    criterion = nn.CrossEntropyLoss(weight=locked_weights.clone(), ignore_index=get_ignore_index())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    metrics = SegmentationMetrics()
    if args.resume:
        start_epoch, best_miou = prepare_resume(config, model, optimizer, scheduler, train_generator)
    else:
        prepare_fresh(config)
        start_epoch, best_miou = 1, -math.inf
    print(json.dumps({
        "experiment": EXPERIMENT, "start_epoch": start_epoch, "train_images": len(train_dataset),
        "val_images": len(val_dataset), "weights_verified": True, "config": str(OUTPUT_DIR / "config.json"),
    }), flush=True)
    if start_epoch > 31:
        raise ValueError("Checkpoint is beyond epoch 30")

    for epoch in range(start_epoch, 31):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val = validate_one_epoch(model, val_loader, criterion, device, metrics)
        if not all(math.isfinite(value) for value in (train.train_loss, val.val_loss, val.miou, val.mean_dice, val.pixel_accuracy)):
            raise RuntimeError(f"Non-finite metric at epoch {epoch}")
        if not torch.equal(criterion.weight, locked_weights):
            raise RuntimeError(f"Class weights changed during epoch {epoch}")
        scheduler.step()
        is_best = val.miou > best_miou
        if is_best:
            best_miou = val.miou
        state = rng_state(train_generator)
        save_checkpoint(
            CHECKPOINT_DIR / "last.pt", epoch=epoch, model=model, optimizer=optimizer,
            best_miou=best_miou, config=config, scheduler=scheduler, extra_state=state,
        )
        if is_best:
            save_checkpoint(
                CHECKPOINT_DIR / "best.pt", epoch=epoch, model=model, optimizer=optimizer,
                best_miou=best_miou, config=config, scheduler=scheduler, extra_state=state,
            )
        torch.cuda.synchronize()
        row = {
            "epoch": epoch, "learning_rate": learning_rate, "train_loss": train.train_loss,
            "val_loss": val.val_loss, "val_miou": val.miou, "val_mean_dice": val.mean_dice,
            "val_pixel_accuracy": val.pixel_accuracy, "epoch_duration_seconds": time.perf_counter() - started,
            "train_batches": train.num_batches, "val_batches": val.num_batches, "is_best": int(is_best),
            "best_miou": best_miou, "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        append_history(OUTPUT_DIR / "history.csv", row)
        print(json.dumps(row, allow_nan=False), flush=True)

    rows = read_history(OUTPUT_DIR / "history.csv")
    if [int(row["epoch"]) for row in rows] != list(range(1, 31)):
        raise RuntimeError("History does not contain exactly epochs 1..30")
    if not (CHECKPOINT_DIR / "best.pt").is_file() or not (CHECKPOINT_DIR / "last.pt").is_file():
        raise RuntimeError("best.pt or last.pt is missing")
    if not math.isclose(optimizer.param_groups[0]["lr"], 1e-5, abs_tol=1e-12):
        raise RuntimeError("Scheduler did not reach eta_min")
    if any(int(row["train_batches"]) != 1331 or int(row["val_batches"]) != 157 for row in rows):
        raise RuntimeError("Epoch batch counts differ from full development splits")
    if not torch.equal(criterion.weight, locked_weights):
        raise RuntimeError("Class weights changed during training")

    names = get_train_id_to_name()
    best = clean_best_evaluation(val_loader, criterion, device, best_miou, names)
    comparison = write_comparison(best, names)
    plot_histories(rows)
    summary = {
        "experiment_name": EXPERIMENT,
        "epochs_completed": len(rows), "train_images": 2662, "val_images": 313,
        "optimizer_steps": sum(int(row["train_batches"]) for row in rows),
        "total_epoch_wall_seconds": sum(float(row["epoch_duration_seconds"]) for row in rows),
        "mean_epoch_seconds": sum(float(row["epoch_duration_seconds"]) for row in rows) / len(rows),
        "best_epoch": best["best_epoch"], "best_miou": best["best_miou"],
        "best_val_loss": best["val_loss"], "best_mean_dice": best["mean_dice"],
        "best_pixel_accuracy": best["pixel_accuracy"], "comparison_deltas": comparison["deltas"],
        "epoch_30": rows[-1], "final_learning_rate": optimizer.param_groups[0]["lr"],
        "peak_cuda_allocated_mib": max(float(row["peak_cuda_allocated_mib"]) for row in rows),
        "peak_cuda_reserved_mib": max(float(row["peak_cuda_reserved_mib"]) for row in rows),
        "class_weights_unchanged": True, "official_val_test_loaded": False,
        "weighted_losses_not_directly_comparable_to_baseline_01_losses": True,
    }
    save_json(OUTPUT_DIR / "summary.json", summary)
    print(json.dumps({
        "complete": True, "best_epoch": best["best_epoch"], "best_miou": best["best_miou"],
        "delta_miou": comparison["deltas"]["miou"], "summary": str(OUTPUT_DIR / "summary.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
