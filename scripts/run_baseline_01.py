"""Run the locked 30-epoch Cityscapes development experiment.

Only ``--resume`` is configurable. Hyperparameters and split membership are
fixed here; the generated config records the environment before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
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

from train_baseline import seed_everything  # noqa: E402
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


EXPERIMENT = "baseline_01_unet_ce"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / EXPERIMENT
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / EXPERIMENT
HISTORY_FIELDS = (
    "epoch", "learning_rate", "train_loss", "val_loss", "val_miou",
    "val_mean_dice", "val_pixel_accuracy", "epoch_duration_seconds",
    "train_batches", "val_batches", "is_best", "best_miou",
    "peak_cuda_allocated_mib", "peak_cuda_reserved_mib",
)


def rng_state(train_generator: torch.Generator) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": (
            numpy_state[0], numpy_state[1].tolist(), numpy_state[2],
            numpy_state[3], numpy_state[4],
        ),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "train_generator_state": train_generator.get_state(),
    }


def restore_rng_state(state: dict, train_generator: torch.Generator) -> None:
    required = {
        "python_random_state", "numpy_random_state", "torch_rng_state",
        "cuda_rng_state_all", "train_generator_state",
    }
    if required - state.keys():
        raise ValueError(f"Checkpoint RNG state is incomplete: {sorted(required - state.keys())}")
    random.setstate(state["python_random_state"])
    algorithm, keys, position, has_gauss, cached_gaussian = state["numpy_random_state"]
    np.random.set_state((algorithm, np.array(keys, dtype=np.uint32), position, has_gauss, cached_gaussian))
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    train_generator.set_state(state["train_generator_state"])


def save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_history(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def read_history(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != HISTORY_FIELDS:
            raise ValueError("History header differs from the locked experiment format")
        return list(reader)


def configuration(split, train_count: int, val_count: int, train_transform) -> dict:
    return {
        "experiment_name": EXPERIMENT,
        "seed": 42,
        "model": "UNet",
        "in_channels": 3,
        "base_channels": 64,
        "num_classes": 19,
        "resolution": {"width": 512, "height": 256},
        "train_split": {"source": "official_train", "cities": list(split.dev_train_cities), "images": train_count},
        "val_split": {"source": "official_train", "cities": list(split.dev_val_cities), "images": val_count},
        "batch_size": 2,
        "epochs": 30,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "T_max": 30,
        "eta_min": 1e-5,
        "loss": "CrossEntropyLoss",
        "ignore_index": get_ignore_index(),
        "precision": "FP32",
        "selection_metric": "dev_val_miou",
        "early_stopping": False,
        "train_transform": {"name": "baseline", "horizontal_flip_probability": train_transform.horizontal_flip_probability},
        "val_transform": "deterministic_baseline",
        "num_workers": 0,
        "num_workers_note": "Conservative Windows setting; no multiprocessing workers",
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision.__version__,
        "cuda_build": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "python_version": platform.python_version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def prepare_fresh(config: dict) -> None:
    if OUTPUT_DIR.exists() or CHECKPOINT_DIR.exists():
        raise FileExistsError("Experiment outputs/checkpoints already exist; inspect them and use --resume only if valid")
    OUTPUT_DIR.mkdir(parents=True)
    CHECKPOINT_DIR.mkdir(parents=True)
    save_json(OUTPUT_DIR / "config.json", config)
    with (OUTPUT_DIR / "history.csv").open("x", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writeheader()
        stream.flush()
        os.fsync(stream.fileno())


def prepare_resume(config: dict, model, optimizer, scheduler, train_generator: torch.Generator) -> tuple[int, float]:
    saved_config = json.loads((OUTPUT_DIR / "config.json").read_text(encoding="utf-8"))
    if saved_config != config:
        raise ValueError("Saved experiment config differs from the locked config or environment")
    rows = read_history(OUTPUT_DIR / "history.csv")
    last_path = CHECKPOINT_DIR / "last.pt"
    best_path = CHECKPOINT_DIR / "best.pt"
    if not last_path.is_file() or not best_path.is_file():
        raise FileNotFoundError("Resume requires both last.pt and best.pt")
    metadata = load_checkpoint(last_path, model=model, optimizer=optimizer, scheduler=scheduler, map_location="cpu")
    if [int(row["epoch"]) for row in rows] != list(range(1, metadata.epoch + 1)):
        raise ValueError("History epochs do not match the last checkpoint; inspect before resuming")
    if not math.isclose(float(rows[-1]["best_miou"]), metadata.best_miou, abs_tol=1e-12):
        raise ValueError("History best mIoU differs from the last checkpoint")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
    if not math.isclose(float(best_payload["best_miou"]), metadata.best_miou, abs_tol=1e-12):
        raise ValueError("Best checkpoint mIoU differs from last checkpoint")
    if metadata.extra_state is None:
        raise ValueError("Last checkpoint lacks RNG state required for resume")
    restore_rng_state(metadata.extra_state, train_generator)
    if any(
        isinstance(value, torch.Tensor) and key != "step" and value.device.type != "cuda"
        for state in optimizer.state.values()
        for key, value in state.items()
    ):
        raise RuntimeError("Restored optimizer moment state is not on CUDA")
    return metadata.epoch + 1, metadata.best_miou


def plot_history(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    plots = (
        ("loss_curve.png", "Loss", (("Train", "train_loss"), ("Dev validation", "val_loss"))),
        ("miou_curve.png", "Dev-validation mIoU", (("mIoU", "val_miou"),)),
        ("dice_curve.png", "Dev-validation mean Dice", (("Mean Dice", "val_mean_dice"),)),
        ("lr_curve.png", "Learning rate", (("AdamW LR", "learning_rate"),)),
    )
    for filename, ylabel, series in plots:
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


def final_evaluation(val_loader, criterion, device, best_miou: float) -> dict:
    model = UNet(in_channels=3, num_classes=19, base_channels=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    metadata = load_checkpoint(CHECKPOINT_DIR / "best.pt", model=model, optimizer=optimizer, map_location="cpu")
    result = validate_one_epoch(model, val_loader, criterion, device, SegmentationMetrics())
    if not math.isclose(result.miou, best_miou, rel_tol=1e-6, abs_tol=1e-8):
        raise RuntimeError("Clean best-checkpoint evaluation differs from saved best mIoU")
    names = get_train_id_to_name()
    per_class = []
    with (OUTPUT_DIR / "best_per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("train_id", "class_name", "iou", "dice", "support"))
        for train_id in range(19):
            iou = float(result.per_class_iou[train_id])
            dice = float(result.per_class_dice[train_id])
            support = int(result.class_support[train_id])
            writer.writerow((train_id, names[train_id], iou if math.isfinite(iou) else "", dice if math.isfinite(dice) else "", support))
            per_class.append({
                "train_id": train_id,
                "class_name": names[train_id],
                "iou": iou if math.isfinite(iou) else None,
                "dice": dice if math.isfinite(dice) else None,
                "support": support,
            })
    payload = {
        "best_epoch": metadata.epoch,
        "best_miou": result.miou,
        "val_loss": result.val_loss,
        "mean_dice": result.mean_dice,
        "pixel_accuracy": result.pixel_accuracy,
        "per_class": per_class,
        "confusion_matrix": result.confusion_matrix.tolist(),
    }
    save_json(OUTPUT_DIR / "best_metrics.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked 30-epoch U-Net baseline on the Cityscapes development split")
    parser.add_argument("--resume", action="store_true", help="Resume only after inspecting last.pt and history.csv")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Locked Baseline 01 requires the verified CUDA GPU; no CPU fallback was selected")
    seed_everything(42)
    split = load_dev_split()
    resolved = validate_dev_split(split, PROJECT_ROOT)
    if len(resolved.dev_train_images) != 2662 or len(resolved.dev_val_images) != 313:
        raise RuntimeError("Development split counts differ from locked 2662/313")
    official_train_root = (PROJECT_ROOT / "leftImg8bit" / "train").resolve()
    if any(not path.resolve().is_relative_to(official_train_root)
           for path in (*resolved.dev_train_images, *resolved.dev_val_images)):
        raise RuntimeError("A development image is outside official train")
    train_transform = make_train_transform()
    val_transform = make_val_transform()
    config = configuration(split, len(resolved.dev_train_images), len(resolved.dev_val_images), train_transform)
    train_dataset = CityscapesSegmentationDataset(PROJECT_ROOT, resolved.dev_train_images, train_transform)
    val_dataset = CityscapesSegmentationDataset(PROJECT_ROOT, resolved.dev_val_images, val_transform)
    train_generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, generator=train_generator)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    device = torch.device("cuda")
    model = UNet(in_channels=3, num_classes=19, base_channels=64).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=get_ignore_index())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    metrics = SegmentationMetrics()
    if args.resume:
        start_epoch, best_miou = prepare_resume(config, model, optimizer, scheduler, train_generator)
    else:
        prepare_fresh(config)
        start_epoch, best_miou = 1, -math.inf
    print(json.dumps({"experiment": EXPERIMENT, "start_epoch": start_epoch,
                      "train_images": len(train_dataset), "val_images": len(val_dataset),
                      "device": str(device), "config": str(OUTPUT_DIR / "config.json")}), flush=True)
    if start_epoch > 31:
        raise ValueError("Checkpoint is beyond the locked 30 epochs")
    # A completed epoch-30 checkpoint can finish post-run evaluation after a
    # technical failure without repeating any training or optimizer steps.

    for epoch in range(start_epoch, 31):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val = validate_one_epoch(model, val_loader, criterion, device, metrics)
        if not all(math.isfinite(value) for value in (
            train.train_loss, val.val_loss, val.miou, val.mean_dice, val.pixel_accuracy
        )):
            raise RuntimeError(f"Non-finite epoch metric at epoch {epoch}")
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
        duration = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train.train_loss,
            "val_loss": val.val_loss,
            "val_miou": val.miou,
            "val_mean_dice": val.mean_dice,
            "val_pixel_accuracy": val.pixel_accuracy,
            "epoch_duration_seconds": duration,
            "train_batches": train.num_batches,
            "val_batches": val.num_batches,
            "is_best": int(is_best),
            "best_miou": best_miou,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        append_history(OUTPUT_DIR / "history.csv", row)
        print(json.dumps(row, allow_nan=False), flush=True)

    rows = read_history(OUTPUT_DIR / "history.csv")
    if [int(row["epoch"]) for row in rows] != list(range(1, 31)):
        raise RuntimeError("History does not contain exactly epochs 1..30")
    if not (CHECKPOINT_DIR / "best.pt").is_file() or not (CHECKPOINT_DIR / "last.pt").is_file():
        raise RuntimeError("Best or last checkpoint is missing")
    if float(rows[0]["learning_rate"]) == float(rows[-1]["learning_rate"]):
        raise RuntimeError("Learning rate did not change across the cosine schedule")
    if not math.isclose(optimizer.param_groups[0]["lr"], 1e-5, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Final learning rate differs from locked eta_min")
    if not all(math.isfinite(float(row[key])) for row in rows for key in (
        "train_loss", "val_loss", "val_miou", "val_mean_dice", "val_pixel_accuracy"
    )):
        raise RuntimeError("History contains a non-finite aggregate metric")
    if any(int(row["train_batches"]) != 1331 or int(row["val_batches"]) != 157 for row in rows):
        raise RuntimeError("History batch counts differ from full development splits")
    best = final_evaluation(val_loader, criterion, device, best_miou)
    if not 1 <= best["best_epoch"] <= 30:
        raise RuntimeError("Best checkpoint epoch is outside 1..30")
    plot_history(rows)
    defined = [item for item in best["per_class"] if item["iou"] is not None]
    ranked = sorted(defined, key=lambda item: item["iou"], reverse=True)
    summary = {
        "experiment_name": EXPERIMENT,
        "epochs_completed": len(rows),
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "optimizer_steps": sum(int(row["train_batches"]) for row in rows),
        "total_epoch_wall_seconds": sum(float(row["epoch_duration_seconds"]) for row in rows),
        "mean_epoch_seconds": sum(float(row["epoch_duration_seconds"]) for row in rows) / len(rows),
        "best_epoch": best["best_epoch"],
        "best_miou": best["best_miou"],
        "best_val_loss": best["val_loss"],
        "best_mean_dice": best["mean_dice"],
        "best_pixel_accuracy": best["pixel_accuracy"],
        "epoch_30": rows[-1],
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "peak_cuda_allocated_mib": max(float(row["peak_cuda_allocated_mib"]) for row in rows),
        "peak_cuda_reserved_mib": max(float(row["peak_cuda_reserved_mib"]) for row in rows),
        "strongest_iou_classes": ranked[:3],
        "weakest_iou_classes": list(reversed(ranked[-3:])),
        "official_val_test_loaded": False,
    }
    save_json(OUTPUT_DIR / "summary.json", summary)
    print(json.dumps({"complete": True, "summary": str(OUTPUT_DIR / "summary.json"),
                      "best_epoch": best["best_epoch"], "best_miou": best["best_miou"]}), flush=True)


if __name__ == "__main__":
    main()
