from typing import Type

import torch.nn as nn
import torch.nn.functional as F

from mmit.base import mismatch as mm
from mmit.base import modules as md
from mmit.base import upsamplers as up


class FPA(nn.Module):
    """Feature Pyramid Attention module from Figure 3 of the paper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
        upsample_layer: Type[nn.Module] = up.Upsample,
        mismatch_layer: Type[nn.Module] = mm.Pad,
    ):
        super().__init__()
        specs = norm_layer, activation, extra_layer
        full_specs = specs + (upsample_layer, mismatch_layer)

        self.top_branch = GlobalPoolingBranch(in_channels, out_channels, *specs)
        self.mid_branch = md.ConvNormAct(in_channels, out_channels, 1, *specs)
        self.bot_branch = ConvBranch(in_channels, out_channels, *full_specs)

        self.fix_mismatch = mismatch_layer()

    def forward(self, x):
        x_top = self.top_branch(x)
        x_mid = self.mid_branch(x)
        x_bottom = self.bot_branch(x)

        x_bottom, x_mid = self.fix_mismatch(x_bottom, x_mid)
        x_bottom_mid = x_bottom * x_mid

        x_end = x_top + x_bottom_mid
        return x_end


class GlobalPoolingBranch(nn.Module):
    """Global pooling branch of FPA module. The on top of Figure 3 in the paper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
    ) -> None:
        super().__init__()
        specs = norm_layer, activation, extra_layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = md.ConvNormAct(in_channels, out_channels, 1, *specs)

    def forward(self, x):
        h, w = x.shape[2:]
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x


class ConvBranch(nn.Module):
    """Convolutional branch of FPA module. The on the bottom of Figure 3 in the paper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
        upsample_layer: Type[nn.Module] = up.Upsample,
        mismatch_layer: Type[nn.Module] = mm.Pad,
    ):
        super().__init__()
        specs = norm_layer, activation, extra_layer
        mid_channels = out_channels // 4

        self.fix_mismatch = mismatch_layer()

        bigpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        bigconv1 = md.ConvNormAct(in_channels, mid_channels, 7, *specs)
        self.bigdown1 = nn.Sequential(bigpool1, bigconv1)

        midpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        midconv1 = md.ConvNormAct(mid_channels, mid_channels, 5, *specs)
        self.middown1 = nn.Sequential(midpool1, midconv1)

        smallpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        smallconv1 = md.ConvNormAct(mid_channels, mid_channels, 3, *specs)
        self.smalldown1 = nn.Sequential(smallpool1, smallconv1)

        self.bigconv2 = md.ConvNormAct(mid_channels, out_channels, 7, *specs)
        self.midconv2 = md.ConvNormAct(mid_channels, out_channels, 5, *specs)
        self.smallconv2 = md.ConvNormAct(mid_channels, out_channels, 3, *specs)

        self.up1 = upsample_layer(out_channels, scale=2)
        self.up2 = upsample_layer(out_channels, scale=2)
        self.up3 = upsample_layer(out_channels, scale=2)

    def forward(self, x):
        x1 = self.bigdown1(x)
        x2 = self.middown1(x1)
        x3 = self.smalldown1(x2)

        x1 = self.bigconv2(x1)
        x2 = self.midconv2(x2)
        x3 = self.smallconv2(x3)

        x3 = self.up3(x3)
        x3, x2 = self.fix_mismatch(x3, x2)

        x2 = x3 + x2
        x2 = self.up2(x2)
        x2, x1 = self.fix_mismatch(x2, x1)

        x1 = x2 + x1
        x = self.up1(x1)
        return x
