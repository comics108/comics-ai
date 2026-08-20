"""Small binary U-Net used by the autonomous Gold segmenter competition."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class CompactBinaryUNet(nn.Module):
    """Two-level 16/32/64-channel U-Net; one logit per foreground pixel."""

    def __init__(self):
        super().__init__()
        self.encoder1 = ConvBlock(3, 16)
        self.encoder2 = ConvBlock(16, 32)
        self.bottleneck = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.decoder1 = ConvBlock(32, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.encoder1(value)
        second = self.encoder2(self.pool(first))
        center = self.bottleneck(self.pool(second))
        decoded2 = self.decoder2(torch.cat((self.up2(center), second), dim=1))
        decoded1 = self.decoder1(torch.cat((self.up1(decoded2), first), dim=1))
        return self.output(decoded1)

