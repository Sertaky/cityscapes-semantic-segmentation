"""Synthetic checks; no Cityscapes dataset files are needed."""

import numpy as np
import pytest
from cityscapesscripts.helpers.labels import labels as official_labels

from cityscapes_segmentation.data.labels import (
    get_ignore_index,
    get_train_id_classes,
    get_train_id_to_name,
    map_raw_ids_to_train_ids,
    raw_id_to_train_id_map,
)


def test_exactly_nineteen_training_classes_with_contiguous_ids():
    classes = get_train_id_classes()
    assert len(classes) == 19
    assert [label.trainId for label in classes] == list(range(19))
    assert all(not label.ignoreInEval for label in classes)


def test_ignore_index():
    assert get_ignore_index() == 255


def test_every_official_raw_id_uses_its_official_train_id():
    mapping = raw_id_to_train_id_map()
    for label in official_labels:
        expected = 255 if label.ignoreInEval or label.trainId < 0 else label.trainId
        assert mapping[label.id] == expected

    raw_ids = np.array([[7, 8, 11, 24], [26, 31, 32, 33]], dtype=np.uint8)
    expected = np.array([[0, 1, 2, 11], [13, 16, 17, 18]], dtype=np.uint8)
    np.testing.assert_array_equal(map_raw_ids_to_train_ids(raw_ids), expected)


def test_ignored_raw_ids_and_ignore_sentinel_map_to_255():
    raw_ids = np.array([[0, 9, 14, 29, 30, -1, 255]], dtype=np.int16)
    output = map_raw_ids_to_train_ids(raw_ids)
    np.testing.assert_array_equal(output, np.full(raw_ids.shape, 255, dtype=np.uint8))


def test_output_has_uint8_dtype_and_preserves_shape():
    raw_ids = np.array([[[7, 255], [24, 0]]], dtype=np.int32)
    output = map_raw_ids_to_train_ids(raw_ids)
    assert output.dtype == np.uint8
    assert output.shape == raw_ids.shape


@pytest.mark.parametrize("unknown", [-2, 34, 256, 999])
def test_unknown_raw_id_raises(unknown):
    with pytest.raises(ValueError, match="Unknown Cityscapes raw label ID"):
        map_raw_ids_to_train_ids(np.array([[7, unknown]], dtype=np.int32))


def test_non_integer_input_raises():
    with pytest.raises(TypeError, match="integer dtype"):
        map_raw_ids_to_train_ids(np.array([[7.0]], dtype=np.float32))


def test_train_id_names_are_deterministic_and_derived_from_official_labels():
    expected = {label.trainId: label.name for label in get_train_id_classes()}
    assert list(get_train_id_to_name()) == list(range(19))
    assert get_train_id_to_name() == expected
    assert get_train_id_to_name() == get_train_id_to_name()
