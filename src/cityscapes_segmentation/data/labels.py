"""19-class training labels derived from the official Cityscapes definitions.

Inputs to ``map_raw_ids_to_train_ids`` are raw ``*_labelIds.png`` values,
not masks that already contain train IDs.
"""

from __future__ import annotations

import numpy as np
from cityscapesscripts.helpers.labels import Label, labels


_IGNORE_INDEX = 255


def get_ignore_index() -> int:
    """Return the target value excluded from training and evaluation."""
    return _IGNORE_INDEX


def get_train_id_classes() -> tuple[Label, ...]:
    """Return the official evaluated classes in train-ID order."""
    classes = tuple(
        sorted(
            (label for label in labels if 0 <= label.trainId < _IGNORE_INDEX and not label.ignoreInEval),
            key=lambda label: label.trainId,
        )
    )
    train_ids = [label.trainId for label in classes]
    if train_ids != list(range(19)):
        raise RuntimeError(f"Expected 19 unique Cityscapes train IDs 0..18; got {train_ids}")
    return classes


def raw_id_to_train_id_map() -> dict[int, int]:
    """Map each official raw ID to its train ID, plus the 255 ignore sentinel.

    The official ``license plate`` raw ID and train ID are both -1; it is
    represented as 255 in our training targets.
    """
    mapping = {
        label.id: (_IGNORE_INDEX if label.ignoreInEval or label.trainId < 0 else label.trainId)
        for label in labels
    }
    mapping[_IGNORE_INDEX] = _IGNORE_INDEX
    return mapping


def map_raw_ids_to_train_ids(mask: np.ndarray) -> np.ndarray:
    """Convert an integer raw-label mask to uint8 train IDs.

    Unknown IDs raise ``ValueError`` rather than being silently ignored.
    """
    array = np.asarray(mask)
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"Raw-ID mask must have an integer dtype; got {array.dtype}")

    mapping = raw_id_to_train_id_map()
    values = np.unique(array)
    unknown = [int(value) for value in values if int(value) not in mapping]
    if unknown:
        raise ValueError(f"Unknown Cityscapes raw label ID(s): {unknown}")

    output = np.empty(array.shape, dtype=np.uint8)
    for value in values:
        output[array == value] = mapping[int(value)]
    return output


def get_train_id_to_name() -> dict[int, str]:
    """Return a deterministic train-ID-to-class-name mapping."""
    return {label.trainId: label.name for label in get_train_id_classes()}
