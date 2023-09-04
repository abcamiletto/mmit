from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmit.base import modules as md


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
        return_features: bool = False,
    ) -> None:
        super().__init__()

        specs = norm_layer, activation, extra_layer

        out_channels = in_channels
        blocks = []
        for size in sizes:
            out_ch = in_channels // len(sizes)
            blocks.append(PoolBlock(in_channels, out_ch, size, *specs))
            out_channels += out_ch

        self.stages = nn.ModuleList(blocks)
        self._out_channels = out_channels
        self.return_features = return_features

    def forward(self, x):
        xs = [block(x) for block in self.stages] + [x]

        if self.return_features:
            return xs

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
        self.conv = md.ConvNormAct(
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
