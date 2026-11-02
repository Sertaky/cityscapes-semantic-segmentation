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


def test_default_unet_uses_standard_dilation_everywhere():
    model = UNet(base_channels=8)
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    assert model.bottleneck_dilation == 1
    assert all(convolution.dilation == (1, 1) for convolution in convolutions)
    assert all(convolution.padding in {(0, 0), (1, 1)} for convolution in convolutions)


def test_dilated_unet_changes_only_bottleneck_convolution_geometry():
    standard = UNet(base_channels=8)
    dilated = UNet(base_channels=8, bottleneck_dilation=2)
    standard_convolutions = dict(standard.named_modules())
    dilated_convolutions = dict(dilated.named_modules())

    changed = []
    for name, module in standard_convolutions.items():
        if not isinstance(module, nn.Conv2d):
            continue
        counterpart = dilated_convolutions[name]
        assert isinstance(counterpart, nn.Conv2d)
        if module.dilation != counterpart.dilation or module.padding != counterpart.padding:
            changed.append(name)

    assert changed == ["bottleneck.0", "bottleneck.3"]
    for name in changed:
        convolution = dilated_convolutions[name]
        assert convolution.dilation == (2, 2)
        assert convolution.padding == (2, 2)


def test_dilated_bottleneck_preserves_spatial_and_output_shapes():
    model = UNet(base_channels=4, bottleneck_dilation=2).eval()
    bottleneck_shape = None

    def capture_shape(_module, _inputs, output):
        nonlocal bottleneck_shape
        bottleneck_shape = output.shape

    handle = model.bottleneck.register_forward_hook(capture_shape)
    with torch.inference_mode():
        logits = model(torch.randn(1, 3, 320, 640))
    handle.remove()

    assert bottleneck_shape == (1, 64, 20, 40)
    assert logits.shape == (1, 19, 320, 640)


def test_dilation_does_not_change_parameter_count_or_state_dict_contract():
    standard = UNet()
    dilated = UNet(bottleneck_dilation=2)
    assert sum(parameter.numel() for parameter in standard.parameters()) == 31_038_803
    assert sum(parameter.numel() for parameter in dilated.parameters()) == 31_038_803
    assert standard.state_dict().keys() == dilated.state_dict().keys()
    for name, tensor in standard.state_dict().items():
        assert tensor.shape == dilated.state_dict()[name].shape

    restored_standard = UNet()
    restored_standard.load_state_dict(standard.state_dict(), strict=True)


def test_invalid_bottleneck_dilation_raises_clearly():
    with pytest.raises(ValueError, match="bottleneck_dilation must be a positive integer"):
        UNet(bottleneck_dilation=0)


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
