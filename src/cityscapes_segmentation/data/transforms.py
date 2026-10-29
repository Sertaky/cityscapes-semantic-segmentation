"""Synchronized image and categorical-mask transforms for the baseline."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageOps


BASELINE_SIZE = (512, 256)  # PIL uses (width, height).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegmentationTransform:
    """Resize a paired RGB image and mapped train-ID mask, then make tensors.

    ``mask`` must already contain train IDs 0..18 or ignore index 255.
    A single random draw controls horizontal flipping of both inputs.
    """

    def __init__(
        self,
        *,
        size: tuple[int, int] = BASELINE_SIZE,
        horizontal_flip_probability: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        if len(size) != 2 or any(not isinstance(dimension, int) or dimension <= 0 for dimension in size):
            raise ValueError("size must be a positive (width, height) pair")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be between 0 and 1")
        self.size = size
        self.horizontal_flip_probability = horizontal_flip_probability
        self.rng = rng if rng is not None else random
        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB image; got mode {image.mode}")
        if mask.mode != "L":
            raise ValueError(f"Expected mapped single-channel mask in L mode; got {mask.mode}")
        if image.size != mask.size:
            raise ValueError(f"Image and mask dimensions differ: {image.size} versus {mask.size}")

        image = image.resize(self.size, resample=Image.Resampling.BILINEAR)
        mask = mask.resize(self.size, resample=Image.Resampling.NEAREST)
        if self.horizontal_flip_probability and self.rng.random() < self.horizontal_flip_probability:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        image_array = np.array(image, dtype=np.uint8, copy=True)
        mask_array = np.array(mask, dtype=np.uint8, copy=True)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).to(torch.float32).div_(255.0)
        image_tensor = (image_tensor - self._mean) / self._std
        target_tensor = torch.from_numpy(mask_array.astype(np.int64))
        return image_tensor, target_tensor


def make_train_transform(
    *, horizontal_flip_probability: float = 0.5, rng: random.Random | None = None
) -> SegmentationTransform:
    """Baseline training resize, optional synchronized flip, and normalization."""
    return SegmentationTransform(horizontal_flip_probability=horizontal_flip_probability, rng=rng)


def make_val_transform() -> SegmentationTransform:
    """Deterministic baseline validation resize and normalization."""
    return SegmentationTransform()
