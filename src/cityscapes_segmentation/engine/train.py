"""One-epoch training and validation for semantic segmentation."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.optim import Optimizer

from cityscapes_segmentation.analysis.metrics import SegmentationMetrics


BatchLoader = Iterable[tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class TrainSummary:
    train_loss: float
    num_samples: int
    num_batches: int


@dataclass(frozen=True)
class ValidationSummary:
    val_loss: float
    miou: float
    mean_dice: float
    pixel_accuracy: float
    per_class_iou: torch.Tensor
    per_class_dice: torch.Tensor
    confusion_matrix: torch.Tensor
    class_support: torch.Tensor
    num_samples: int
    num_batches: int


def _batches(loader: BatchLoader, max_batches: int | None) -> BatchLoader:
    if max_batches is None:
        return loader
    if not isinstance(max_batches, int) or isinstance(max_batches, bool) or max_batches <= 0:
        raise ValueError("max_batches must be a positive integer or None")
    return itertools.islice(loader, max_batches)


def _loss_value(loss: torch.Tensor, batch_index: int) -> float:
    if loss.ndim != 0:
        raise ValueError(f"Loss must be scalar at batch {batch_index}; got shape {tuple(loss.shape)}")
    value = float(loss.detach().item())
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite loss at batch {batch_index}: {value}")
    return value


def train_one_epoch(
    model: nn.Module,
    loader: BatchLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
    *,
    max_batches: int | None = None,
) -> TrainSummary:
    """Train for one loader pass or an explicitly limited smoke pass.

    Epoch loss is ``sum(batch_loss * batch_size) / total_samples``. Since the
    criterion may average over valid pixels within each batch, this is a
    sample-weighted mean of batch losses, not a global per-pixel mean.
    The caller places ``model`` on ``device`` before invoking this function.
    """
    model.train()
    device = torch.device(device)
    weighted_loss = 0.0
    num_samples = num_batches = 0
    for batch_index, (images, targets) in enumerate(_batches(loader, max_batches)):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        value = _loss_value(loss, batch_index)
        loss.backward()
        optimizer.step()
        batch_size = images.shape[0]
        weighted_loss += value * batch_size
        num_samples += batch_size
        num_batches += 1
    if num_samples == 0:
        raise ValueError("Training loader produced no samples")
    return TrainSummary(weighted_loss / num_samples, num_samples, num_batches)


def validate_one_epoch(
    model: nn.Module,
    loader: BatchLoader,
    criterion: nn.Module,
    device: torch.device | str,
    metrics: SegmentationMetrics,
    *,
    max_batches: int | None = None,
) -> ValidationSummary:
    """Evaluate one loader pass using a fresh accumulated confusion matrix.

    Validation loss uses the same sample-weighted convention as training.
    IoU, Dice, and accuracy come from the combined confusion matrix, never an
    average of batch metrics. The caller places ``model`` on ``device``.
    """
    model.eval()
    metrics.reset()
    device = torch.device(device)
    weighted_loss = 0.0
    num_samples = num_batches = 0
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(_batches(loader, max_batches)):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            value = _loss_value(loss, batch_index)
            predictions = logits.argmax(dim=1)
            metrics.update(predictions, targets)
            batch_size = images.shape[0]
            weighted_loss += value * batch_size
            num_samples += batch_size
            num_batches += 1
    if num_samples == 0:
        raise ValueError("Validation loader produced no samples")
    result = metrics.compute()
    return ValidationSummary(
        val_loss=weighted_loss / num_samples,
        miou=result.miou,
        mean_dice=result.mean_dice,
        pixel_accuracy=result.pixel_accuracy,
        per_class_iou=result.per_class_iou,
        per_class_dice=result.per_class_dice,
        confusion_matrix=result.confusion_matrix,
        class_support=result.support,
        num_samples=num_samples,
        num_batches=num_batches,
    )
