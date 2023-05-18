from typing import Type

import torch
import torch.nn.functional as F
from torch import nn

from mmit.base import upsamplers as up


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        upsample_layer: Type[nn.Module] = up.ConvTranspose2d,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
    ):
        super().__init__()

        self.up = upsample_layer(in_channels)

        conv_channels = in_channels + skip_channels
        self.conv = DoubleConvBlock(conv_channels, out_channels, norm_layer, activation)

        self.extra_layer = extra_layer()

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            x = self.concatenate(x, skip)

        x = self.extra_layer(x)
        x = self.conv(x)
        return x

    def concatenate(self, x, skip):
        x_size, skip_size = x.shape[2:], skip.shape[2:]

        if x_size != skip_size:
            x = F.interpolate(x, size=skip_size, mode="bilinear")

        return torch.cat([x, skip], dim=1)


class DoubleConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = norm_layer(out_channels)
        self.activation1 = activation()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = norm_layer(out_channels)
        self.activation2 = activation()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        return x
