from typing import Type

import torch.nn as nn
import torch.nn.functional as F

from mmit.base import modules as md


class ASPPConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn = norm_layer(out_channels)
        self.relu = activation()
        self.extra_layer = extra_layer()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ASPPPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
    ):
        super().__init__()
        specs = norm_layer, activation, extra_layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = md.ConvNormAct(in_channels, out_channels, 1, *specs)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.gap(x)
        x = self.conv(x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x
