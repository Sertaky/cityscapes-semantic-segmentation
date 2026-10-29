"""Run the custom U-Net on the city-separated development split.

Use --max-train-batches and --max-val-batches for infrastructure smoke runs.
Full experiments require a separate reviewed configuration and run decision.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cityscapes_segmentation.analysis.metrics import SegmentationMetrics  # noqa: E402
from cityscapes_segmentation.data.dataset import CityscapesSegmentationDataset  # noqa: E402
from cityscapes_segmentation.data.labels import get_ignore_index  # noqa: E402
from cityscapes_segmentation.data.splits import load_dev_split, validate_dev_split  # noqa: E402
from cityscapes_segmentation.data.transforms import make_train_transform, make_val_transform  # noqa: E402
from cityscapes_segmentation.engine import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)
from cityscapes_segmentation.models import UNet  # noqa: E402


def seed_everything(seed: int) -> None:
    """Seed the main process; CUDA kernel determinism is not forced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Cityscapes U-Net on the development split")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "baseline_01")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("epochs and batch-size and learning-rate must be positive; num-workers must be nonnegative")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        parser.error("max-train-batches must be positive")
    if args.max_val_batches is not None and args.max_val_batches <= 0:
        parser.error("max-val-batches must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    split = load_dev_split()
    resolved = validate_dev_split(split, args.dataset_root)
    train_dataset = CityscapesSegmentationDataset(args.dataset_root, resolved.dev_train_images, make_train_transform())
    val_dataset = CityscapesSegmentationDataset(args.dataset_root, resolved.dev_val_images, make_val_transform())
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = UNet(in_channels=3, num_classes=19, base_channels=64).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=get_ignore_index())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    metrics = SegmentationMetrics()
    config = {
        "model": "UNet",
        "in_channels": 3,
        "num_classes": 19,
        "base_channels": 64,
        "input_width": 512,
        "input_height": 256,
        "ignore_index": get_ignore_index(),
        "optimizer": "AdamW",
        "source_split": split.source_split,
        "dev_val_cities": list(split.dev_val_cities),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
    }
    best_miou = -math.inf
    start_epoch = 1
    if args.resume is not None:
        restored = load_checkpoint(args.resume, model=model, optimizer=optimizer, map_location=device)
        best_miou = restored.best_miou
        start_epoch = restored.epoch + 1
    if start_epoch > args.epochs:
        raise ValueError(f"Requested --epochs={args.epochs} does not exceed restored epoch {start_epoch - 1}")

    for epoch in range(start_epoch, args.epochs + 1):
        train = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_batches=args.max_train_batches
        )
        val = validate_one_epoch(
            model, val_loader, criterion, device, metrics, max_batches=args.max_val_batches
        )
        if math.isfinite(val.miou) and val.miou > best_miou:
            best_miou = val.miou
            save_checkpoint(
                args.checkpoint_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_miou=best_miou,
                config=config,
            )
        save_checkpoint(
            args.checkpoint_dir / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_miou=best_miou,
            config=config,
        )
        print(json.dumps({
            "epoch": epoch,
            "train_loss": train.train_loss,
            "val_loss": val.val_loss,
            "miou": val.miou,
            "mean_dice": val.mean_dice,
            "pixel_accuracy": val.pixel_accuracy,
            "train_batches": train.num_batches,
            "val_batches": val.num_batches,
            "train_samples_seen": train.num_samples,
            "val_samples_seen": val.num_samples,
            "best_miou": best_miou,
            "checkpoint_dir": str(args.checkpoint_dir),
        }), flush=True)


if __name__ == "__main__":
    main()
