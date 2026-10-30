"""Synthetic class-counting and weighting tests; no raw dataset required."""

import numpy as np
import pytest

from cityscapes_segmentation.analysis.class_weights import (
    NUM_CLASSES,
    compute_class_weights,
    count_raw_label_mask,
)


def test_raw_counting_excludes_ignored_pixels():
    raw = np.array([[7, 7, 8], [0, 255, 11]], dtype=np.uint8)
    result = count_raw_label_mask(raw)
    assert result.total_pixels == 6
    assert result.valid_pixels == 4
    assert result.ignored_pixels == 2
    np.testing.assert_array_equal(result.class_counts[:3], [2, 1, 1])
    assert int(result.class_counts[3:].sum()) == 0


def test_frequencies_formula_and_normalized_mean():
    counts = np.square(np.arange(1, NUM_CLASSES + 1, dtype=np.float64))
    result = compute_class_weights(counts)
    expected_frequencies = counts / counts.sum()
    expected_raw = 1.0 / np.sqrt(expected_frequencies)
    np.testing.assert_allclose(result.frequencies, expected_frequencies, rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.sqrt_inverse_raw, expected_raw, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        result.sqrt_inverse_normalized,
        expected_raw / expected_raw.mean(),
        rtol=0,
        atol=1e-15,
    )
    assert result.frequencies.sum() == pytest.approx(1.0)
    assert result.sqrt_inverse_normalized.mean() == pytest.approx(1.0)


def test_weight_vector_has_nineteen_finite_positive_values():
    result = compute_class_weights(np.arange(1, NUM_CLASSES + 1, dtype=np.int64))
    assert result.sqrt_inverse_normalized.shape == (19,)
    assert np.all(np.isfinite(result.sqrt_inverse_normalized))
    assert np.all(result.sqrt_inverse_normalized > 0)


def test_zero_frequency_class_raises_clearly():
    counts = np.ones(NUM_CLASSES, dtype=np.int64)
    counts[12] = 0
    with pytest.raises(ValueError, match="zero-frequency class IDs.*12"):
        compute_class_weights(counts)


def test_wrong_count_vector_shape_raises():
    with pytest.raises(ValueError, match=r"shape \(19,\)"):
        compute_class_weights(np.ones(NUM_CLASSES - 1, dtype=np.int64))
