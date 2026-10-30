"""Dev-train pixel counting and deterministic class-weight calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cityscapes_segmentation.data.labels import (
    get_ignore_index,
    get_train_id_classes,
    raw_id_to_train_id_map,
)


NUM_CLASSES = len(get_train_id_classes())


@dataclass(frozen=True)
class MaskPixelCounts:
    """Mapped class counts plus excluded pixels for one or more raw masks."""

    class_counts: np.ndarray
    ignored_pixels: int
    total_pixels: int

    @property
    def valid_pixels(self) -> int:
        return int(self.class_counts.sum())


@dataclass(frozen=True)
class ClassWeightResult:
    """Float64 frequencies and mean-one candidate weight vectors."""

    frequencies: np.ndarray
    sqrt_inverse_raw: np.ndarray
    sqrt_inverse_normalized: np.ndarray
    inverse_normalized: np.ndarray


def count_raw_label_mask(mask: np.ndarray) -> MaskPixelCounts:
    """Count train IDs from a raw Cityscapes labelIds mask without remapping pixels.

    The authoritative project raw-ID mapping is applied to each observed raw ID.
    Unknown values raise rather than entering the ignored count silently.
    """
    array = np.asarray(mask)
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"Raw-ID mask must have an integer dtype; got {array.dtype}")
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional raw labelIds mask; got shape {array.shape}")

    mapping = raw_id_to_train_id_map()
    values, counts = np.unique(array, return_counts=True)
    unknown = [int(value) for value in values if int(value) not in mapping]
    if unknown:
        raise ValueError(f"Unknown Cityscapes raw label ID(s): {unknown}")

    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    ignored_pixels = 0
    ignore_index = get_ignore_index()
    for raw_id, count in zip(values, counts, strict=True):
        train_id = mapping[int(raw_id)]
        if train_id == ignore_index:
            ignored_pixels += int(count)
        else:
            class_counts[train_id] += int(count)
    total_pixels = int(array.size)
    if int(class_counts.sum()) + ignored_pixels != total_pixels:
        raise RuntimeError("Mapped valid and ignored counts do not cover the raw mask")
    return MaskPixelCounts(class_counts, ignored_pixels, total_pixels)


def compute_class_weights(class_counts: np.ndarray) -> ClassWeightResult:
    """Compute mean-one sqrt-inverse and diagnostic inverse weights.

    Frequencies use valid class pixels only. Every class must have nonzero
    support because a finite reciprocal-frequency weight is required.
    """
    counts = np.asarray(class_counts)
    if counts.shape != (NUM_CLASSES,):
        raise ValueError(f"class_counts must have shape ({NUM_CLASSES},); got {counts.shape}")
    if not np.issubdtype(counts.dtype, np.number) or not np.all(np.isfinite(counts)):
        raise TypeError("class_counts must contain finite numeric values")
    if np.any(counts < 0):
        raise ValueError("class_counts cannot contain negative values")
    zero_classes = np.flatnonzero(counts == 0)
    if zero_classes.size:
        raise ValueError(f"Cannot weight zero-frequency class IDs: {zero_classes.tolist()}")

    frequencies = counts.astype(np.float64) / float(counts.sum(dtype=np.float64))
    sqrt_inverse_raw = np.reciprocal(np.sqrt(frequencies))
    sqrt_inverse_normalized = sqrt_inverse_raw / sqrt_inverse_raw.mean()
    inverse_raw = np.reciprocal(frequencies)
    inverse_normalized = inverse_raw / inverse_raw.mean()
    return ClassWeightResult(
        frequencies=frequencies,
        sqrt_inverse_raw=sqrt_inverse_raw,
        sqrt_inverse_normalized=sqrt_inverse_normalized,
        inverse_normalized=inverse_normalized,
    )
