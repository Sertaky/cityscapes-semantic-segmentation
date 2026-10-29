"""Custom four-level U-Net that returns semantic class logits."""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    """Two spatially preserving Conv-BatchNorm-ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    """Double spatial size, concatenate the matching skip, and refine."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                f"Upsampled and skip spatial shapes differ: {tuple(x.shape[-2:])} "
                f"versus {tuple(skip.shape[-2:])}"
            )
        return self.conv(torch.cat((skip, x), dim=1))


class UNet(nn.Module):
    """Four downsampling stages with 64-base-channel default progression.

    Input is ``[B, in_channels, H, W]`` with H and W at least 32 and
    divisible by 16. Output is raw ``[B, num_classes, H, W]`` logits.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 19, base_channels: int = 64) -> None:
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("num_classes", num_classes),
            ("base_channels", base_channels),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        self.in_channels = in_channels
        c = base_channels
        self.encoder1 = DoubleConv(in_channels, c)
        self.encoder2 = DoubleConv(c, 2 * c)
        self.encoder3 = DoubleConv(2 * c, 4 * c)
        self.encoder4 = DoubleConv(4 * c, 8 * c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(8 * c, 16 * c)
        self.decoder4 = UpBlock(16 * c, 8 * c)
        self.decoder3 = UpBlock(8 * c, 4 * c)
        self.decoder2 = UpBlock(4 * c, 2 * c)
        self.decoder1 = UpBlock(2 * c, c)
        self.classifier = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B,C,H,W]; got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels; got {x.shape[1]}")
        height, width = x.shape[-2:]
        if height < 32 or width < 32 or height % 16 or width % 16:
            raise ValueError(f"Input H and W must be at least 32 and divisible by 16; got {height}x{width}")

        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        skip4 = self.encoder4(self.pool(skip3))
        x = self.bottleneck(self.pool(skip4))
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        return self.classifier(x)
