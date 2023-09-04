from typing import List, Optional, Type

import torch
from torch import nn

from mmit.base import modules as md
from mmit.base import upsamplers as up
from mmit.factory import register

from ..basedecoder import BaseDecoder
from ..utils import size_control
from .aspp import ASPP

__all__ = ["DeepLabV3"]

DEFAULT_CHANNEL = 256
DEFAULT_ATROUS_RATES = [12, 24, 36]


@register
class DeepLabV3(BaseDecoder):
    """
    Implementation of the DeepLabV3 decoder. Paper: https://arxiv.org/abs/1706.05587
    To follow the paper as much as possible, we only process the feature map closest to the stride 8 by default.

    Args:
        input_channels: The channels of the input features.
        input_reductions: The reduction factor of the input features.
        decoder_channel: The channel to use on the decoder.
        atrous_rates: The atrous rates to use on the ASPP module.
        feature_index: The index of the feature to use.
        upsample_layer: Upsampling layer to use.
        norm_layer: Normalization layer to use.
        activation_layer: Activation function to use.
        extra_layer: Addional layer to use.
        return_features: Whether to return the intermediate results of the decoder.

    """

    def __init__(
        self,
        input_channels: List[int],
        input_reductions: List[int],
        decoder_channel: int = DEFAULT_CHANNEL,
        atrous_rates: List[int] = DEFAULT_ATROUS_RATES,
        feature_index: Optional[int] = None,
        upsample_layer: Type[nn.Module] = up.Upsample,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation_layer: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
        return_features: bool = False,
    ):
        super().__init__(input_channels, input_reductions, return_features)
        self.input_index = feature_index or self._get_index(input_reductions)
        self._out_classes = decoder_channel

        in_channel = input_channels[self.input_index]
        specs = norm_layer, activation_layer, extra_layer

        self.aspp = ASPP(in_channel, decoder_channel, atrous_rates, *specs)
        self.conv = md.ConvNormAct(decoder_channel, decoder_channel, 3, *specs)
        scale = self.input_reductions[self.input_index]
        self.up = upsample_layer(decoder_channel, scale=scale)

    @size_control
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        x0 = features[self.input_index]
        x1 = self.aspp(x0)
        x2 = self.conv(x1)
        x3 = self.up(x2)

        if self.return_features:
            return [x1, x2, x3]

        return x3

    def _get_index(self, input_reductions: List[int]) -> int:
        closest_index = None
        closest_value = float("inf")
        for i, red in enumerate(input_reductions):
            if abs(red - 8) <= closest_value:
                closest_value = abs(red - 8)
                closest_index = i
        return closest_index

    @property
    def out_classes(self) -> int:
        return self._out_classes
