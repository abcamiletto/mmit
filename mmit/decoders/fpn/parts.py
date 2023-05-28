from typing import Type

import torch.nn as nn
from torch import Tensor

from mmit.base import mismatch as mm
from mmit.base import upsamplers as up


class ConvNormActivation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_layer: Type[nn.Module] = up.ConvTranspose2d,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.up = upsample_layer(in_channels)
        self.norm = norm_layer(out_channels)
        self.activation = activation()
        self.extra = extra_layer()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.up(x)
        x = self.extra(x)
        return x


class SkipBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        upsample_layer: Type[nn.Module] = up.ConvTranspose2d,
        mismatch_layer: Type[nn.Module] = mm.Pad,
    ):
        super().__init__()
        self.up = upsample_layer(input_channels)
        self.skip_conv = nn.Conv2d(skip_channels, input_channels, kernel_size=1)
        self.fix_mismatch = mismatch_layer()

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            skip = self.skip_conv(skip)
            x, skip = self.fix_mismatch(x, skip)
            x = x + skip

        return x
