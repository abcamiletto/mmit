from typing import List, Type

import torch
import torch.nn as nn

from mmit.base import modules as md

from .parts import ASPPConv, ASPPPooling


class ASPP(nn.Module):
    """
    Implements the Atrous Spatial Pyramid Pooling (ASPP) module of DeepLabV3.

    This module applies a series of convolutions at different dilation rates, as well as
    global average pooling to capture different levels of image context. The outputs are
    then concatenated and passed through a projection layer to reduce the channel dimension.
    The size of the output is the same as the input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates: List[int],
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
        asppconv_layer: Type[nn.Module] = ASPPConv,
    ) -> None:
        super().__init__()

        specs = norm_layer, activation, extra_layer

        modules = []
        conv = md.ConvNormAct(in_channels, out_channels, 1, *specs)
        modules.append(conv)

        for rate in atrous_rates:
            asppconv = asppconv_layer(in_channels, out_channels, rate, *specs)
            modules.append(asppconv)

        modules.append(ASPPPooling(in_channels, out_channels, *specs))

        self.convs = nn.ModuleList(modules)

        in_channels = len(self.convs) * out_channels
        self.project = md.ConvNormAct(in_channels, out_channels, 1, *specs)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.project(res)
        return self.dropout(res)
