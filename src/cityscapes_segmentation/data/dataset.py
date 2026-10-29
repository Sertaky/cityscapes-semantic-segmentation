"""Load Cityscapes RGB images with semantic raw-label masks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cityscapes_segmentation.data.labels import map_raw_ids_to_train_ids
from cityscapes_segmentation.data.transforms import make_val_transform


PairTransform = Callable[[Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]]
_IMAGE_SUFFIX = "_leftImg8bit.png"


class CityscapesSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Represent image paths already selected by the split utility.

    The caller supplies membership; this class pairs, loads, and maps samples.
    Without an explicit transform, it uses deterministic baseline validation
    preprocessing so the output is immediately usable as tensors.
    """

    def __init__(
        self,
        dataset_root: Path,
        image_paths: Iterable[Path],
        transform: PairTransform | None = None,
    ) -> None:
        root = Path(dataset_root).expanduser().resolve()
        image_root = root / "leftImg8bit"
        self.transform = transform if transform is not None else make_val_transform()
        samples: list[tuple[Path, Path]] = []
        for supplied_path in image_paths:
            image_path = Path(supplied_path).expanduser().resolve()
            try:
                relative = image_path.relative_to(image_root)
            except ValueError as exc:
                raise ValueError(f"Image is outside {image_root}: {image_path}") from exc
            if len(relative.parts) != 3 or relative.parts[0] not in {"train", "val"}:
                raise ValueError(f"Expected a train/val Cityscapes image path: {image_path}")
            if not image_path.name.endswith(_IMAGE_SUFFIX):
                raise ValueError(f"Expected a {_IMAGE_SUFFIX} image: {image_path}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Cityscapes image is missing: {image_path}")
            mask_path = root / "gtFine" / relative.parent / f"{image_path.name.removesuffix(_IMAGE_SUFFIX)}_gtFine_labelIds.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Cityscapes semantic labelIds mask is missing: {mask_path}")
            samples.append((image_path, mask_path))
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        with Image.open(mask_path) as source_mask:
            raw_mask = np.array(source_mask)
        if raw_mask.ndim != 2:
            raise ValueError(f"Expected a single-channel raw labelIds mask: {mask_path}")
        try:
            mapped_mask = map_raw_ids_to_train_ids(raw_mask)
        except ValueError as exc:
            raise ValueError(f"{mask_path}: {exc}") from exc
        target = Image.fromarray(mapped_mask)
        return self.transform(image, target)
