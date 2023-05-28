from typing import Type

import torch.nn as nn
import torch.nn.functional as F

from mmit.base import modules as md


class GAU(nn.Module):
    """Global Attention Upsample Block of Figure 4 in the paper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
    ):
        super().__init__()

        specs = norm_layer, activation

        pool = nn.AdaptiveAvgPool2d(1)
        conv = md.ConvNormAct(out_channels, out_channels, 1, norm_layer, nn.Sigmoid)
        self.conv_high = nn.Sequential(pool, conv)

        self.conv_low = md.ConvNormAct(in_channels, out_channels, 3, *specs)

        self.extra = extra_layer()

    def forward(self, low_level, high_level):
        h, w = low_level.shape[-2:]

        low_level = self.conv_low(low_level)

        pooled_high_level = self.conv_high(high_level)
        new_low_level = low_level * pooled_high_level

        high_level = F.interpolate(high_level, size=(h, w), mode="bilinear")

        result = new_low_level + high_level
        return self.extra(result)
