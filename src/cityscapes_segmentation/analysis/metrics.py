"""Semantic-segmentation metrics accumulated over pixels, not batches.

Confusion-matrix rows are ground-truth classes and columns are predictions:
``matrix[target, prediction] += 1``. Target pixels equal to the ignore index
are excluded before counting. A class absent from both target and prediction
has undefined (NaN) IoU and Dice and is omitted from the corresponding mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from cityscapes_segmentation.data.labels import (
    get_ignore_index,
    get_train_id_classes,
    get_train_id_to_name,
)


NUM_CLASSES = len(get_train_id_classes())
IGNORE_INDEX = get_ignore_index()


@dataclass(frozen=True)
class SegmentationMetricResult:
    """CPU snapshot of accumulated counts and derived metrics.

    Per-class tensors use train-ID order. Undefined values are NaN; aggregate
    metrics are NaN when no class or pixel contributes a defined value.
    """

    confusion_matrix: torch.Tensor
    true_positives: torch.Tensor
    false_positives: torch.Tensor
    false_negatives: torch.Tensor
    support: torch.Tensor
    per_class_iou: torch.Tensor
    miou: float
    per_class_dice: torch.Tensor
    mean_dice: float
    pixel_accuracy: float
    class_names: dict[int, str]


class SegmentationMetrics:
    """Accumulate an int64 confusion matrix across segmentation batches.

    ``update`` accepts integer PyTorch tensors or NumPy arrays with matching
    shapes, normally ``[B, H, W]``. It also accepts unbatched or flattened
    arrays. CUDA inputs are counted on CUDA; only two compact validation flags
    are synchronized per update. ``compute`` transfers the 19x19 state to CPU.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX) -> None:
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if not isinstance(ignore_index, int) or isinstance(ignore_index, bool) or 0 <= ignore_index < num_classes:
            raise ValueError("ignore_index must be an integer outside valid class IDs")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        """Clear all previously accumulated pixels."""
        self._confusion: torch.Tensor | None = None

    @staticmethod
    def _integer_tensor(value: torch.Tensor | np.ndarray, name: str) -> torch.Tensor:
        if not isinstance(value, (torch.Tensor, np.ndarray)):
            raise TypeError(f"{name} must be an integer PyTorch tensor or NumPy array")
        tensor = torch.as_tensor(value)
        try:
            torch.iinfo(tensor.dtype)
        except TypeError as exc:
            raise TypeError(f"{name} must have an integer dtype; got {tensor.dtype}") from exc
        return tensor.to(dtype=torch.int64)

    def update(self, predictions: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray) -> None:
        """Validate and add one batch without averaging batch-level scores."""
        prediction = self._integer_tensor(predictions, "predictions")
        target = self._integer_tensor(targets, "targets")
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction and target shapes differ: {tuple(prediction.shape)} versus {tuple(target.shape)}")
        if prediction.device != target.device:
            raise ValueError(f"Prediction and target devices differ: {prediction.device} versus {target.device}")

        invalid_prediction = ((prediction < 0) | (prediction >= self.num_classes)).any()
        invalid_target = ((target != self.ignore_index) & ((target < 0) | (target >= self.num_classes))).any()
        bad_prediction, bad_target = torch.stack((invalid_prediction, invalid_target)).cpu().tolist()
        if bad_prediction:
            raise ValueError(f"Predictions contain a class ID outside 0..{self.num_classes - 1}")
        if bad_target:
            raise ValueError(f"Targets contain a value outside 0..{self.num_classes - 1} or ignore index {self.ignore_index}")

        valid = target != self.ignore_index
        indices = target[valid] * self.num_classes + prediction[valid]
        batch_matrix = torch.bincount(indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        if self._confusion is None:
            self._confusion = batch_matrix
        else:
            self._confusion += batch_matrix.to(self._confusion.device)

    def compute(self) -> SegmentationMetricResult:
        """Compute scores from the total confusion matrix seen so far."""
        matrix = (
            torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
            if self._confusion is None
            else self._confusion.detach().to(device="cpu", dtype=torch.int64).clone()
        )
        tp = matrix.diagonal().clone()
        support = matrix.sum(dim=1)
        fn = support - tp
        fp = matrix.sum(dim=0) - tp
        iou_denominator = tp + fp + fn
        dice_denominator = 2 * tp + fp + fn
        nan_values = torch.full((self.num_classes,), torch.nan, dtype=torch.float64)
        iou = torch.where(
            iou_denominator > 0,
            tp.to(torch.float64) / iou_denominator.clamp_min(1),
            nan_values,
        )
        dice = torch.where(
            dice_denominator > 0,
            (2 * tp).to(torch.float64) / dice_denominator.clamp_min(1),
            nan_values,
        )
        valid_pixels = int(matrix.sum())
        return SegmentationMetricResult(
            confusion_matrix=matrix,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            support=support,
            per_class_iou=iou,
            miou=float(torch.nanmean(iou)),
            per_class_dice=dice,
            mean_dice=float(torch.nanmean(dice)),
            pixel_accuracy=float(tp.sum()) / valid_pixels if valid_pixels else math.nan,
            class_names=get_train_id_to_name() if self.num_classes == NUM_CLASSES else {},
        )
