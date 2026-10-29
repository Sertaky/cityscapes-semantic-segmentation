"""Fast synthetic checks for one-epoch train and validation behavior."""

import math

import pytest
import torch
from torch import nn

from cityscapes_segmentation.analysis.metrics import SegmentationMetrics
from cityscapes_segmentation.engine.train import train_one_epoch, validate_one_epoch


class PixelScoreModel(nn.Module):
    """Predict class 1 for positive pixels and class 0 for negative pixels."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        score = images[:, 0] * self.scale
        return torch.stack((-score, score), dim=1)


def test_train_one_epoch_updates_parameters_and_reports_counts():
    torch.manual_seed(3)
    model = nn.Conv2d(3, 2, kernel_size=1)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    images = torch.randn(2, 3, 4, 4)
    targets = torch.randint(0, 2, (2, 4, 4))

    summary = train_one_epoch(
        model, [(images, targets)], optimizer, nn.CrossEntropyLoss(ignore_index=255), "cpu"
    )

    assert model.training
    assert summary.num_samples == 2
    assert summary.num_batches == 1
    assert math.isfinite(summary.train_loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


class TargetMeanLoss(nn.Module):
    """Make batch losses 0 and 1 to expose incorrect unweighted averaging."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return targets.float().mean() + logits.sum() * 0


def test_train_and_validation_loss_use_sample_weighting():
    model = nn.Conv2d(3, 2, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [
        (torch.zeros(2, 3, 2, 2), torch.zeros(2, 2, 2, dtype=torch.int64)),
        (torch.zeros(1, 3, 2, 2), torch.ones(1, 2, 2, dtype=torch.int64)),
    ]
    criterion = TargetMeanLoss()
    train = train_one_epoch(model, batches, optimizer, criterion, "cpu")
    val = validate_one_epoch(model, batches, criterion, "cpu", SegmentationMetrics(num_classes=2))
    assert train.train_loss == pytest.approx(1 / 3)
    assert val.val_loss == pytest.approx(1 / 3)
    assert train.num_samples == val.num_samples == 3
    assert train.num_batches == val.num_batches == 2


def test_validation_argmax_accumulated_metrics_no_parameter_updates_and_reset():
    model = PixelScoreModel()
    before = model.scale.detach().clone()
    images = torch.tensor([[[[-1.0, 1.0], [1.0, 1.0]]]])
    targets = torch.tensor([[[0, 0], [1, 255]]])
    batch = [(images, targets)]
    metrics = SegmentationMetrics(num_classes=2)

    first = validate_one_epoch(model, batch, nn.CrossEntropyLoss(ignore_index=255), "cpu", metrics)
    second = validate_one_epoch(model, batch, nn.CrossEntropyLoss(ignore_index=255), "cpu", metrics)

    assert not model.training
    torch.testing.assert_close(model.scale.detach(), before)
    assert model.scale.grad is None
    torch.testing.assert_close(first.confusion_matrix, torch.tensor([[1, 1], [0, 1]]))
    torch.testing.assert_close(second.confusion_matrix, first.confusion_matrix)
    assert first.class_support.tolist() == [2, 1]
    assert first.per_class_iou.tolist() == pytest.approx([0.5, 0.5])
    assert first.per_class_dice.tolist() == pytest.approx([2 / 3, 2 / 3])
    assert first.miou == pytest.approx(0.5)
    assert first.mean_dice == pytest.approx(2 / 3)
    assert first.pixel_accuracy == pytest.approx(2 / 3)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_loss_names_batch_and_value(bad_value: float):
    class BadLoss(nn.Module):
        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return logits.sum() * 0 + bad_value

    model = nn.Conv2d(3, 2, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = [(torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2, dtype=torch.int64))]
    with pytest.raises(RuntimeError, match=r"Non-finite loss at batch 0: (nan|inf)"):
        train_one_epoch(model, batch, optimizer, BadLoss(), "cpu")
    with pytest.raises(RuntimeError, match=r"Non-finite loss at batch 0: (nan|inf)"):
        validate_one_epoch(model, batch, BadLoss(), "cpu", SegmentationMetrics(num_classes=2))


def test_max_batches_limits_work_and_empty_loaders_raise():
    model = nn.Conv2d(3, 2, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = (torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2, dtype=torch.int64))
    limited = train_one_epoch(model, [batch, batch, batch], optimizer, nn.CrossEntropyLoss(), "cpu", max_batches=2)
    assert (limited.num_batches, limited.num_samples) == (2, 2)
    with pytest.raises(ValueError, match="no samples"):
        validate_one_epoch(model, [], nn.CrossEntropyLoss(), "cpu", SegmentationMetrics(num_classes=2))
