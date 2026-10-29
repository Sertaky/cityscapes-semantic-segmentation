"""Synthetic geometry and tensor-contract checks; no raw dataset required."""

import random

import numpy as np
import pytest
import torch
from PIL import Image

from cityscapes_segmentation.data.transforms import (
    make_train_transform,
    make_val_transform,
)


def test_resize_dtypes_nearest_labels_and_ignore_value():
    image = Image.fromarray(np.array([[[255, 0, 0], [0, 0, 255]], [[0, 255, 0], [255, 255, 0]]], dtype=np.uint8))
    mask = Image.fromarray(np.array([[0, 18], [255, 0]], dtype=np.uint8))

    image_tensor, target = make_val_transform()(image, mask)

    assert image_tensor.shape == (3, 256, 512)
    assert target.shape == (256, 512)
    assert image_tensor.dtype == torch.float32
    assert target.dtype == torch.int64
    assert set(torch.unique(target).tolist()) == {0, 18, 255}
    assert target[0, 0] == 0
    assert target[0, -1] == 18
    assert target[-1, 0] == 255


def test_horizontal_flip_uses_one_decision_for_image_and_mask():
    image = Image.fromarray(np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8))
    mask = Image.fromarray(np.array([[0, 18]], dtype=np.uint8))
    unflipped_image, unflipped_mask = make_train_transform(horizontal_flip_probability=0)(image, mask)
    flipped_image, flipped_mask = make_train_transform(horizontal_flip_probability=1)(image, mask)

    torch.testing.assert_close(flipped_image, torch.flip(unflipped_image, dims=(-1,)))
    torch.testing.assert_close(flipped_mask, torch.flip(unflipped_mask, dims=(-1,)))
    assert flipped_mask[0, 0] == 18
    assert flipped_mask[0, -1] == 0


def test_seeded_flip_decision_is_controllable():
    image = Image.new("RGB", (2, 1), color=(1, 2, 3))
    mask = Image.fromarray(np.array([[0, 18]], dtype=np.uint8))
    _, no_flip = make_train_transform(horizontal_flip_probability=0.5, rng=random.Random(0))(image, mask)
    _, flip = make_train_transform(horizontal_flip_probability=0.5, rng=random.Random(1))(image, mask)
    assert no_flip[0, 0] == 0
    assert flip[0, 0] == 18


def test_validation_transform_is_deterministic():
    image = Image.new("RGB", (4, 2), color=(30, 80, 120))
    mask = Image.fromarray(np.array([[0, 18, 255, 0], [18, 0, 18, 255]], dtype=np.uint8))
    transform = make_val_transform()
    first_image, first_target = transform(image, mask)
    second_image, second_target = transform(image, mask)
    torch.testing.assert_close(first_image, second_image)
    torch.testing.assert_close(first_target, second_target)


def test_mismatched_image_and_mask_dimensions_raise():
    with pytest.raises(ValueError, match="dimensions differ"):
        make_val_transform()(Image.new("RGB", (2, 2)), Image.new("L", (3, 2)))
