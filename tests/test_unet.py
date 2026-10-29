"""Structural and forward checks for the custom U-Net baseline."""

import pytest
import torch
from torch import nn

from cityscapes_segmentation.models import UNet


def test_default_forward_shape_dtype_and_batch_preservation():
    model = UNet().eval()
    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 256, 512, dtype=torch.float32))
    assert logits.shape == (2, 19, 256, 512)
    assert logits.dtype == torch.float32


def test_single_sample_forward_shape():
    model = UNet(base_channels=8).eval()
    with torch.inference_mode():
        logits = model(torch.randn(1, 3, 256, 512))
    assert logits.shape == (1, 19, 256, 512)


def test_configurable_input_channels_and_classes():
    model = UNet(in_channels=1, num_classes=5, base_channels=8).eval()
    with torch.inference_mode():
        logits = model(torch.randn(1, 1, 64, 96))
    assert logits.shape == (1, 5, 64, 96)


def test_output_head_is_raw_logits_without_activation():
    model = UNet(base_channels=8)
    assert isinstance(model.classifier, nn.Conv2d)
    assert model.classifier.in_channels == 8
    assert model.classifier.out_channels == 19
    assert model.classifier.kernel_size == (1, 1)
    assert not any(isinstance(module, (nn.Softmax, nn.Sigmoid, nn.LogSoftmax)) for module in model.modules())


def test_encoder_decoder_channels_and_skip_concatenation():
    model = UNet()
    assert [block[0].out_channels for block in (
        model.encoder1, model.encoder2, model.encoder3, model.encoder4, model.bottleneck
    )] == [64, 128, 256, 512, 1024]
    for decoder, incoming, output in (
        (model.decoder4, 1024, 512),
        (model.decoder3, 512, 256),
        (model.decoder2, 256, 128),
        (model.decoder1, 128, 64),
    ):
        assert decoder.upsample.in_channels == incoming
        assert decoder.upsample.out_channels == output
        assert decoder.upsample.kernel_size == (2, 2)
        assert decoder.upsample.stride == (2, 2)
        assert decoder.conv[0].in_channels == output * 2
        assert decoder.conv[0].out_channels == output


@pytest.mark.parametrize("shape", [(1, 3, 255, 512), (1, 3, 256, 510), (1, 3, 16, 32)])
def test_invalid_spatial_dimensions_raise_clearly(shape):
    with pytest.raises(ValueError, match="at least 32 and divisible by 16"):
        UNet(base_channels=4)(torch.randn(shape))


def test_invalid_channel_count_raises_clearly():
    with pytest.raises(ValueError, match="Expected 3 input channels"):
        UNet(base_channels=4)(torch.randn(1, 1, 32, 32))


def test_gradients_reach_trainable_parameters():
    model = UNet(base_channels=4)
    logits = model(torch.randn(1, 3, 32, 32))
    logits.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    assert any(torch.count_nonzero(parameter.grad) > 0 for parameter in model.parameters() if parameter.requires_grad)


def test_cross_entropy_accepts_logits_and_int64_target_with_ignore():
    model = UNet(base_channels=4).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 32, 32))
        target = torch.randint(0, 19, (1, 32, 32), dtype=torch.int64)
        target[0, 0, 0] = 255
        loss = nn.CrossEntropyLoss(ignore_index=255)(logits, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
