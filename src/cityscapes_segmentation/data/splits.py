"""City-separated development membership within official Cityscapes train."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "configs/splits/cityscapes_dev_split.json"
_CITY_NAME = re.compile(r"[a-z]+\Z")
_IMAGE_SUFFIX = "_leftImg8bit.png"


class SplitValidationError(ValueError):
    """The manifest or local split layout is inconsistent."""


@dataclass(frozen=True)
class DevSplit:
    source_split: str
    dev_train_cities: tuple[str, ...]
    dev_val_cities: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedDevSplit:
    dev_train_images: tuple[Path, ...]
    dev_val_images: tuple[Path, ...]
    missing_masks: tuple[Path, ...]


def _check_city_lists(split: DevSplit) -> None:
    if split.source_split != "train":
        raise SplitValidationError("Development split source_split must be 'train'")
    for name, cities in (("dev_train_cities", split.dev_train_cities), ("dev_val_cities", split.dev_val_cities)):
        if not cities:
            raise SplitValidationError(f"{name} must be non-empty")
        if any(not isinstance(city, str) or not _CITY_NAME.fullmatch(city) for city in cities):
            raise SplitValidationError(f"{name} must contain lowercase Cityscapes city names")
        if len(cities) != len(set(cities)):
            raise SplitValidationError(f"{name} contains duplicate cities")
    overlap = set(split.dev_train_cities) & set(split.dev_val_cities)
    if overlap:
        raise SplitValidationError(f"Development train/val cities overlap: {sorted(overlap)}")


def load_dev_split(manifest_path: Path | None = None) -> DevSplit:
    """Read the tracked city-level manifest without accessing raw images."""
    path = DEFAULT_MANIFEST if manifest_path is None else Path(manifest_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitValidationError(f"Cannot read development split manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SplitValidationError("Development split manifest must be a JSON object")
    for key in ("dev_train_cities", "dev_val_cities"):
        if not isinstance(data.get(key), list):
            raise SplitValidationError(f"Manifest {key} must be a list")
    split = DevSplit(
        source_split=data.get("source_split"),
        dev_train_cities=tuple(data["dev_train_cities"]),
        dev_val_cities=tuple(data["dev_val_cities"]),
    )
    _check_city_lists(split)
    return split


def _city_dirs(path: Path) -> set[str]:
    if not path.is_dir():
        raise SplitValidationError(f"Required Cityscapes split directory is missing: {path}")
    return {item.name for item in path.iterdir() if item.is_dir()}


def collect_city_images(dataset_root: Path, cities: tuple[str, ...]) -> tuple[Path, ...]:
    """Return sorted official-train image paths for the supplied cities."""
    root = Path(dataset_root).expanduser().resolve() / "leftImg8bit" / "train"
    images = []
    for city in sorted(cities):
        directory = root / city
        if not directory.is_dir():
            raise SplitValidationError(f"Official train city directory is missing: {directory}")
        city_images = sorted(directory.glob(f"*{_IMAGE_SUFFIX}"))
        if not city_images:
            raise SplitValidationError(f"No Cityscapes images found in official train city: {directory}")
        for image in city_images:
            if not image.is_file() or not image.name.startswith(f"{city}_"):
                raise SplitValidationError(f"Invalid image in official train city: {image}")
            if not image.resolve().is_relative_to(root):
                raise SplitValidationError(f"Image resolves outside official train: {image}")
        images.extend(city_images)
    return tuple(sorted(images))


def validate_dev_split(split: DevSplit, dataset_root: Path) -> ValidatedDevSplit:
    """Check city coverage, leakage, and semantic-mask pairing on disk.

    Only directory entries and filenames are inspected; image pixels are not read.
    """
    _check_city_lists(split)
    root = Path(dataset_root).expanduser().resolve()
    official_train = _city_dirs(root / "leftImg8bit" / "train")
    official_val = _city_dirs(root / "leftImg8bit" / "val")
    official_test = _city_dirs(root / "leftImg8bit" / "test")
    _city_dirs(root / "gtFine" / "train")
    configured = set(split.dev_train_cities) | set(split.dev_val_cities)
    forbidden = configured & (official_val | official_test)
    if forbidden:
        raise SplitValidationError(f"Official val/test cities cannot enter development split: {sorted(forbidden)}")
    unknown = configured - official_train
    if unknown:
        raise SplitValidationError(f"Unknown official train cities in development split: {sorted(unknown)}")
    omitted = official_train - configured
    if omitted:
        raise SplitValidationError(f"Official train cities omitted from development split: {sorted(omitted)}")
    train_images = collect_city_images(root, split.dev_train_cities)
    val_images = collect_city_images(root, split.dev_val_cities)
    if set(train_images) & set(val_images):
        raise SplitValidationError("Development train and validation image paths overlap")
    train_root = (root / "leftImg8bit" / "train").resolve()
    if any(not path.resolve().is_relative_to(train_root) for path in (*train_images, *val_images)):
        raise SplitValidationError("Development image resolves outside official train")
    missing_masks = tuple(
        root / "gtFine" / "train" / path.parent.name / f"{path.name.removesuffix(_IMAGE_SUFFIX)}_gtFine_labelIds.png"
        for path in (*train_images, *val_images)
        if not (root / "gtFine" / "train" / path.parent.name / f"{path.name.removesuffix(_IMAGE_SUFFIX)}_gtFine_labelIds.png").is_file()
    )
    if missing_masks:
        raise SplitValidationError(f"Missing {len(missing_masks)} semantic masks; first: {missing_masks[0]}")
    return ValidatedDevSplit(train_images, val_images, missing_masks)
