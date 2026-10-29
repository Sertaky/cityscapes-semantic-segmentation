"""Synthetic sample tests and optional local Cityscapes smoke checks."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cityscapes_segmentation.data.dataset import CityscapesSegmentationDataset
from cityscapes_segmentation.data.splits import load_dev_split, validate_dev_split
from cityscapes_segmentation.data.transforms import make_val_transform


@pytest.fixture
def sample_paths(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "leftImg8bit" / "train" / "testcity"
    mask_dir = tmp_path / "gtFine" / "train" / "testcity"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image_path = image_dir / "testcity_000000_000019_leftImg8bit.png"
    mask_path = mask_dir / "testcity_000000_000019_gtFine_labelIds.png"
    Image.fromarray(np.full((2, 4, 3), (20, 80, 140), dtype=np.uint8)).save(image_path)
    Image.fromarray(np.array([[7, 8, 11, 26], [33, 0, 255, 7]], dtype=np.uint8)).save(mask_path)
    return image_path, mask_path


def test_dataset_length_shape_dtypes_mapping_and_valid_values(sample_paths: tuple[Path, Path], tmp_path: Path):
    image_path, _ = sample_paths
    dataset = CityscapesSegmentationDataset(tmp_path, [image_path])
    assert len(dataset) == 1

    image, target = dataset[0]
    assert image.shape == (3, 256, 512)
    assert target.shape == (256, 512)
    assert image.dtype == torch.float32
    assert target.dtype == torch.int64
    assert [target[0, column].item() for column in (0, 128, 256, 384)] == [0, 1, 2, 13]
    assert [target[-1, column].item() for column in (0, 128, 256, 384)] == [18, 255, 255, 0]
    assert set(torch.unique(target).tolist()) == {0, 1, 2, 13, 18, 255}


def test_missing_mask_raises_at_construction(sample_paths: tuple[Path, Path], tmp_path: Path):
    image_path, mask_path = sample_paths
    mask_path.unlink()
    with pytest.raises(FileNotFoundError, match="semantic labelIds mask is missing"):
        CityscapesSegmentationDataset(tmp_path, [image_path])


def test_unknown_raw_id_raises(sample_paths: tuple[Path, Path], tmp_path: Path):
    image_path, mask_path = sample_paths
    Image.fromarray(np.array([[7, 34], [8, 0]], dtype=np.uint8)).save(mask_path)
    with pytest.raises(ValueError, match="Unknown Cityscapes raw label ID.*34"):
        CityscapesSegmentationDataset(tmp_path, [image_path])[0]


def test_grayscale_source_image_is_loaded_as_rgb(sample_paths: tuple[Path, Path], tmp_path: Path):
    image_path, _ = sample_paths
    Image.new("L", (4, 2), color=100).save(image_path)
    image, _ = CityscapesSegmentationDataset(tmp_path, [image_path])[0]
    assert image.shape[0] == 3
    torch.testing.assert_close(image[0] * 0.229 + 0.485, image[1] * 0.224 + 0.456)


def test_only_labelids_is_used_as_target(sample_paths: tuple[Path, Path], tmp_path: Path):
    image_path, _ = sample_paths
    color_path = tmp_path / "gtFine" / "train" / "testcity" / "testcity_000000_000019_gtFine_color.png"
    Image.new("RGB", (4, 2), color=(255, 0, 0)).save(color_path)
    _, target = CityscapesSegmentationDataset(tmp_path, [image_path])[0]
    assert set(torch.unique(target).tolist()) == {0, 1, 2, 13, 18, 255}


def test_local_dev_samples_when_dataset_available():
    root = Path(__file__).resolve().parents[1]
    if not (root / "leftImg8bit").exists() and not (root / "gtFine").exists():
        pytest.skip("Raw Cityscapes data is unavailable")
    split = validate_dev_split(load_dev_split(), root)
    train_dataset = CityscapesSegmentationDataset(root, split.dev_train_images, make_val_transform())
    val_dataset = CityscapesSegmentationDataset(root, split.dev_val_images, make_val_transform())
    assert len(train_dataset) == 2662
    assert len(val_dataset) == 313
    for dataset in (train_dataset, val_dataset):
        for index in (0, len(dataset) // 2, len(dataset) - 1):
            image, target = dataset[index]
            assert image.shape == (3, 256, 512)
            assert target.shape == (256, 512)
            assert image.dtype == torch.float32
            assert target.dtype == torch.int64
            assert set(torch.unique(target).tolist()) <= set(range(19)) | {255}
