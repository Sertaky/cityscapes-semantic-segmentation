"""Hand-calculated confusion-matrix tests; no dataset or model required."""

import math

import numpy as np
import pytest
import torch

from cityscapes_segmentation.analysis.metrics import (
    IGNORE_INDEX,
    NUM_CLASSES,
    SegmentationMetrics,
)
from cityscapes_segmentation.data.labels import get_train_id_to_name


def test_defaults_and_official_class_names():
    result = SegmentationMetrics().compute()
    assert NUM_CLASSES == 19
    assert IGNORE_INDEX == 255
    assert result.confusion_matrix.shape == (19, 19)
    assert result.confusion_matrix.dtype == torch.int64
    assert result.class_names == get_train_id_to_name()


def test_perfect_prediction_and_absent_classes():
    target = torch.tensor([[[0, 1], [1, 0]]])
    meter = SegmentationMetrics()
    meter.update(target, target)
    result = meter.compute()
    torch.testing.assert_close(result.confusion_matrix[:2, :2], torch.tensor([[2, 0], [0, 2]]))
    assert result.per_class_iou[:2].tolist() == [1.0, 1.0]
    assert result.per_class_dice[:2].tolist() == [1.0, 1.0]
    assert torch.isnan(result.per_class_iou[2:]).all()
    assert torch.isnan(result.per_class_dice[2:]).all()
    assert result.miou == 1.0
    assert result.mean_dice == 1.0
    assert result.pixel_accuracy == 1.0


def test_exact_false_positive_false_negative_and_matrix_orientation():
    target = torch.tensor([[[0, 0], [1, 1]]])
    prediction = torch.tensor([[[0, 1], [1, 1]]])
    meter = SegmentationMetrics()
    meter.update(prediction, target)
    result = meter.compute()

    # Rows are targets, columns are predictions.
    torch.testing.assert_close(result.confusion_matrix[:2, :2], torch.tensor([[1, 1], [0, 2]]))
    assert result.true_positives[:2].tolist() == [1, 2]
    assert result.false_positives[:2].tolist() == [0, 1]
    assert result.false_negatives[:2].tolist() == [1, 0]
    assert result.support[:2].tolist() == [2, 2]
    assert result.per_class_iou[0].item() == pytest.approx(0.5)
    assert result.per_class_iou[1].item() == pytest.approx(2 / 3)
    assert result.per_class_dice[0].item() == pytest.approx(2 / 3)
    assert result.per_class_dice[1].item() == pytest.approx(0.8)
    assert torch.isnan(result.per_class_iou[2:]).all()
    assert torch.isnan(result.per_class_dice[2:]).all()
    assert result.miou == pytest.approx(7 / 12)
    assert result.mean_dice == pytest.approx(11 / 15)
    assert result.pixel_accuracy == pytest.approx(0.75)


def test_ignored_target_pixel_contributes_nothing():
    target = torch.tensor([[[0, 255], [1, 255]]])
    first = SegmentationMetrics()
    first.update(torch.tensor([[[0, 0], [0, 1]]]), target)
    second = SegmentationMetrics()
    second.update(torch.tensor([[[0, 18], [0, 17]]]), target)
    torch.testing.assert_close(first.compute().confusion_matrix, second.compute().confusion_matrix)
    assert int(first.compute().confusion_matrix.sum()) == 2
    assert first.compute().pixel_accuracy == 0.5


@pytest.mark.parametrize("prediction,target", [
    ([[1]], [[0]]),  # Class 1 is prediction-only: FP > 0.
    ([[0]], [[1]]),  # Class 1 is target-only: FN > 0.
])
def test_prediction_only_and_target_only_classes_score_zero(prediction, target):
    meter = SegmentationMetrics()
    meter.update(torch.tensor(prediction), torch.tensor(target))
    result = meter.compute()
    assert result.true_positives[1] == 0
    assert result.per_class_iou[1] == 0
    assert result.per_class_dice[1] == 0
    assert (result.false_positives[1] > 0) or (result.false_negatives[1] > 0)


def test_multiple_updates_match_one_combined_update():
    predictions = torch.tensor([[[0, 1, 2]], [[1, 0, 2]]])
    targets = torch.tensor([[[0, 0, 255]], [[1, 1, 2]]])
    accumulated = SegmentationMetrics()
    accumulated.update(predictions[:1], targets[:1])
    accumulated.update(predictions[1:], targets[1:])
    combined = SegmentationMetrics()
    combined.update(predictions, targets)
    left, right = accumulated.compute(), combined.compute()
    torch.testing.assert_close(left.confusion_matrix, right.confusion_matrix)
    torch.testing.assert_close(left.per_class_iou, right.per_class_iou, equal_nan=True)
    torch.testing.assert_close(left.per_class_dice, right.per_class_dice, equal_nan=True)
    assert left.miou == right.miou
    assert left.mean_dice == right.mean_dice
    assert left.pixel_accuracy == right.pixel_accuracy


def test_reset_clears_accumulated_state_and_prior_result_is_snapshot():
    meter = SegmentationMetrics()
    meter.update(torch.tensor([[0]]), torch.tensor([[0]]))
    before_reset = meter.compute()
    meter.reset()
    after_reset = meter.compute()
    assert int(before_reset.confusion_matrix.sum()) == 1
    assert int(after_reset.confusion_matrix.sum()) == 0
    assert math.isnan(after_reset.miou)
    assert math.isnan(after_reset.mean_dice)
    assert math.isnan(after_reset.pixel_accuracy)


@pytest.mark.parametrize("bad_prediction", [-1, 19, 255])
def test_invalid_prediction_raises_even_at_ignored_target(bad_prediction):
    with pytest.raises(ValueError, match="Predictions contain a class ID"):
        SegmentationMetrics().update(torch.tensor([[bad_prediction]]), torch.tensor([[255]]))


@pytest.mark.parametrize("bad_target", [-1, 19, 256])
def test_invalid_target_raises(bad_target):
    with pytest.raises(ValueError, match="Targets contain a value"):
        SegmentationMetrics().update(torch.tensor([[0]]), torch.tensor([[bad_target]]))


def test_shape_mismatch_and_noninteger_values_raise():
    meter = SegmentationMetrics()
    with pytest.raises(ValueError, match="shapes differ"):
        meter.update(torch.zeros((1, 2), dtype=torch.int64), torch.zeros((2, 1), dtype=torch.int64))
    with pytest.raises(TypeError, match="integer dtype"):
        meter.update(torch.tensor([[0.0]]), torch.tensor([[0]]))
    with pytest.raises(TypeError, match="integer dtype"):
        meter.update(torch.tensor([[0]]), np.array([[0.0]]))


def test_all_targets_ignored_and_empty_input_have_undefined_metrics():
    for prediction, target in (
        (torch.tensor([[0, 18]]), torch.tensor([[255, 255]])),
        (torch.empty((0,), dtype=torch.int64), torch.empty((0,), dtype=torch.int64)),
    ):
        meter = SegmentationMetrics()
        meter.update(prediction, target)
        result = meter.compute()
        assert int(result.confusion_matrix.sum()) == 0
        assert torch.isnan(result.per_class_iou).all()
        assert torch.isnan(result.per_class_dice).all()
        assert math.isnan(result.miou)
        assert math.isnan(result.mean_dice)
        assert math.isnan(result.pixel_accuracy)


def test_numpy_integer_arrays_are_supported():
    meter = SegmentationMetrics(num_classes=2)
    meter.update(np.array([[0, 1]], dtype=np.uint16), np.array([[0, 1]], dtype=np.uint8))
    assert meter.compute().pixel_accuracy == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_inputs_accumulate_and_compute_on_cpu():
    meter = SegmentationMetrics()
    prediction = torch.tensor([[[0, 1], [1, 1]]], device="cuda")
    target = torch.tensor([[[0, 0], [1, 255]]], device="cuda")
    meter.update(prediction, target)
    result = meter.compute()
    assert result.confusion_matrix.device.type == "cpu"
    torch.testing.assert_close(result.confusion_matrix[:2, :2], torch.tensor([[1, 1], [0, 1]]))
    assert result.pixel_accuracy == pytest.approx(2 / 3)
