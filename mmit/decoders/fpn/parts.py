from typing import Type

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmit.base import upsamplers as up


class ConvNormReLU(nn.Module):
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
        self.extra_layer = extra_layer()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.up(x)
        x = self.extra_layer(x)
        return x


class SkipBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        upsample_layer: Type[nn.Module] = up.ConvTranspose2d,
        mismatch_fix_strategy: str = "interpolate",
    ):
        super().__init__()
        self.up = upsample_layer(input_channels)
        self.skip_conv = nn.Conv2d(skip_channels, input_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            x_size, skip_size = x.shape[2:], skip.shape[2:]

            if x_size != skip_size:
                x = F.interpolate(x, size=skip_size, mode="bilinear")

            skip = self.skip_conv(skip)
            x = x + skip

        return x
