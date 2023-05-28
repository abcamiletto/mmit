from typing import Type

import torch.nn as nn

from mmit.base import modules as md


class LinkBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_layer: Type[nn.Module] = nn.Upsample,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
        mismatch_layer: Type[nn.Module] = nn.Identity,
    ) -> None:
        super().__init__()
        specs = norm_layer, activation, extra_layer

        self.conv1 = md.ConvNormAct(in_channels, in_channels // 4, 1, *specs)

        up = upsample_layer(in_channels // 4)
        bn = norm_layer(in_channels // 4)
        act = activation()

        self.up = nn.Sequential(up, bn, act)

        self.conv2 = md.ConvNormAct(in_channels // 4, out_channels, 1, *specs)

        self.fix_mismatch = mismatch_layer()

    def forward(self, x, skip=None):
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)
        if skip is not None:
            x, skip = self.fix_mismatch(x, skip)
            x = x + skip
        return x
