"""City-level split checks; only the optional integration test needs raw data."""

from dataclasses import replace
from pathlib import Path

import pytest

from cityscapes_segmentation.data.splits import (
    DevSplit,
    SplitValidationError,
    collect_city_images,
    load_dev_split,
    validate_dev_split,
)


OFFICIAL_TRAIN_CITIES = {
    "aachen", "bochum", "bremen", "cologne", "darmstadt", "dusseldorf",
    "erfurt", "hamburg", "hanover", "jena", "krefeld", "monchengladbach",
    "strasbourg", "stuttgart", "tubingen", "ulm", "weimar", "zurich",
}


def test_tracked_manifest_loads_and_covers_official_train_cities():
    split = load_dev_split()
    assert split.source_split == "train"
    assert split.dev_val_cities == ("jena", "krefeld", "ulm")
    assert not set(split.dev_train_cities) & set(split.dev_val_cities)
    assert set(split.dev_train_cities) | set(split.dev_val_cities) == OFFICIAL_TRAIN_CITIES


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    for split, cities in (("train", ("aachen", "bochum", "jena")), ("val", ("frankfurt",)), ("test", ("berlin",))):
        for city in cities:
            (tmp_path / "leftImg8bit" / split / city).mkdir(parents=True)
    for city in ("aachen", "bochum", "jena"):
        (tmp_path / "gtFine" / "train" / city).mkdir(parents=True)
        frames = (2, 1) if city == "aachen" else (1,)
        for frame in frames:
            stem = f"{city}_000000_{frame:06d}"
            (tmp_path / "leftImg8bit" / "train" / city / f"{stem}_leftImg8bit.png").touch()
            (tmp_path / "gtFine" / "train" / city / f"{stem}_gtFine_labelIds.png").touch()
    return tmp_path


@pytest.fixture
def synthetic_split() -> DevSplit:
    return DevSplit("train", ("bochum", "aachen"), ("jena",))


def test_sorted_membership_pairs_and_no_leakage(synthetic_dataset: Path, synthetic_split: DevSplit):
    result = validate_dev_split(synthetic_split, synthetic_dataset)
    assert len(result.dev_train_images) == 3
    assert len(result.dev_val_images) == 1
    assert result.missing_masks == ()
    assert result.dev_train_images == tuple(sorted(result.dev_train_images))
    assert result.dev_val_images == tuple(sorted(result.dev_val_images))
    assert collect_city_images(synthetic_dataset, synthetic_split.dev_train_cities) == result.dev_train_images
    assert not set(result.dev_train_images) & set(result.dev_val_images)
    assert all(path.is_relative_to(synthetic_dataset / "leftImg8bit" / "train") for path in (*result.dev_train_images, *result.dev_val_images))
    assert validate_dev_split(synthetic_split, synthetic_dataset) == result


def test_overlapping_cities_rejected(synthetic_dataset: Path, synthetic_split: DevSplit):
    bad = replace(synthetic_split, dev_val_cities=("aachen",))
    with pytest.raises(SplitValidationError, match="overlap"):
        validate_dev_split(bad, synthetic_dataset)


def test_unknown_train_city_rejected(synthetic_dataset: Path, synthetic_split: DevSplit):
    bad = replace(synthetic_split, dev_train_cities=(*synthetic_split.dev_train_cities, "nowhere"))
    with pytest.raises(SplitValidationError, match="Unknown official train cities"):
        validate_dev_split(bad, synthetic_dataset)


def test_omitted_train_city_rejected(synthetic_dataset: Path, synthetic_split: DevSplit):
    bad = replace(synthetic_split, dev_train_cities=("aachen",))
    with pytest.raises(SplitValidationError, match="omitted.*bochum"):
        validate_dev_split(bad, synthetic_dataset)


@pytest.mark.parametrize("forbidden_city", ["frankfurt", "berlin"])
def test_official_val_or_test_city_rejected(synthetic_dataset: Path, synthetic_split: DevSplit, forbidden_city: str):
    bad = replace(synthetic_split, dev_train_cities=(*synthetic_split.dev_train_cities, forbidden_city))
    with pytest.raises(SplitValidationError, match="Official val/test cities"):
        validate_dev_split(bad, synthetic_dataset)


def test_missing_semantic_mask_rejected(synthetic_dataset: Path, synthetic_split: DevSplit):
    (synthetic_dataset / "gtFine" / "train" / "jena" / "jena_000000_000001_gtFine_labelIds.png").unlink()
    with pytest.raises(SplitValidationError, match="Missing 1 semantic masks"):
        validate_dev_split(synthetic_split, synthetic_dataset)


@pytest.mark.parametrize("bad_json", [
    '{"source_split":"train","dev_train_cities":["aachen"],"dev_val_cities":["aachen"]}',
    '{"source_split":"val","dev_train_cities":["aachen"],"dev_val_cities":["jena"]}',
    '{"source_split":"train","dev_train_cities":"aachen","dev_val_cities":["jena"]}',
])
def test_malformed_manifest_rejected(tmp_path: Path, bad_json: str):
    path = tmp_path / "split.json"
    path.write_text(bad_json, encoding="utf-8")
    with pytest.raises(SplitValidationError):
        load_dev_split(path)


def test_local_cityscapes_counts_and_masks_when_dataset_available():
    root = Path(__file__).resolve().parents[1]
    images_present = (root / "leftImg8bit").is_dir()
    masks_present = (root / "gtFine").is_dir()
    if not images_present and not masks_present:
        pytest.skip("Raw Cityscapes data is not present in this checkout")
    assert images_present and masks_present, "Local Cityscapes data is incomplete"
    result = validate_dev_split(load_dev_split(), root)
    assert len(result.dev_train_images) == 2662
    assert len(result.dev_val_images) == 313
    assert len(result.dev_train_images) + len(result.dev_val_images) == 2975
    assert result.missing_masks == ()
