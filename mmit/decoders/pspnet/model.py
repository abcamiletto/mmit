from functools import partial
from typing import List, Optional, Type

import torch.nn as nn

from mmit.base import modules as md
from mmit.base import upsamplers as up
from mmit.factory import register

from ..basedecoder import BaseDecoder
from ..utils import size_control
from .parts import PSPModule

__all__ = ["PSPNet"]

DEFAULT_CHANNEL = 256


@register
class PSPNet(BaseDecoder):
    """
    Implementation of the PSPNet decoder. Paper: https://arxiv.org/abs/1612.01105.
    To follow the paper as much as possible, we only process the feature map closest to the stride 8 by default.

    Args:
        input_channels: The channels of the input features.
        input_reductions: The reduction factor of the input features.
        decoder_channel: The channel to use on the decoder.
        dropout: The dropout to use.
        sizes: The sizes to use on the PSP module.
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
        dropout: float = 0.2,
        sizes: List[int] = (1, 2, 3, 6),
        feature_index: Optional[int] = None,
        upsample_layer: Type[nn.Module] = up.Upsample,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation_layer: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
        return_features: bool = False,
    ):
        super().__init__(input_channels, input_reductions, return_features)

        self.input_index = feature_index or self._get_index(input_reductions)
        final_up = self._format_upsample_layers(input_reductions, upsample_layer)
        self._out_classes = decoder_channel

        specs = norm_layer, activation_layer, extra_layer

        in_ch = input_channels[self.input_index]

        self.psp = PSPModule(in_ch, sizes, *specs, return_features=return_features)

        self.conv = md.ConvNormAct(self.psp.out_channels, decoder_channel, 1, *specs)

        self.dropout = nn.Dropout2d(p=dropout)
        self.up = final_up(decoder_channel)

    @size_control
    def forward(self, *features):
        x = features[self.input_index]

        x = self.psp(x)

        if self.return_features:
            return x

        x = self.conv(x)
        x = self.dropout(x)
        x = self.up(x)
        return x

    @property
    def out_classes(self) -> int:
        return self._out_classes

    def _get_index(self, input_reductions: List[int]) -> int:
        closest_index = None
        closest_value = float("inf")
        for i, red in enumerate(input_reductions):
            if abs(red - 8) <= closest_value:
                closest_value = abs(red - 8)
                closest_index = i
        return closest_index

    def _format_upsample_layers(
        self, input_reductions: List[int], upsample_layer: Type[nn.Module]
    ) -> nn.Module:
        scale = input_reductions[self.input_index]
        return partial(upsample_layer, scale=scale)
