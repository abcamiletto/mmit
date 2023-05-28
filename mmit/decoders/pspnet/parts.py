from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing module
    """

    def __init__(
        self,
        in_channels: int,
        sizes: List[int] = (1, 2, 3, 6),
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
    ) -> None:
        super().__init__()

        specs = norm_layer, activation, extra_layer

        out_channels = in_channels
        blocks = []
        for size in sizes:
            out_ch = in_channels // len(sizes)
            blocks.append(PoolBlock(in_channels, out_ch, size, *specs))
            out_channels += out_ch

        self.blocks = nn.ModuleList(blocks)
        self._out_channels = out_channels

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x

    @property
    def out_channels(self) -> int:
        return self._out_channels


class PoolBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.conv = ConvNormActivation(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_layer=norm_layer if pool_size > 1 else nn.Identity,
            activation=activation,
            extra_layer=extra_layer,
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x


class ConvNormActivation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm = norm_layer(out_channels)
        self.activation = activation()
        self.extra = extra_layer()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.extra(x)
        return x
