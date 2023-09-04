from functools import partial
from typing import List, Type

import torch
from torch import nn

from mmit.base import mismatch as mm
from mmit.base import modules as md
from mmit.base import upsamplers as up
from mmit.factory import register

from ..basedecoder import BaseDecoder
from ..utils import size_control
from .aspp import ASPP
from .parts import DWSConvNormActivation

__all__ = ["DeepLabV3Plus"]

DEFAULT_CHANNELS = [256, 48]
DEFAULT_ATROUS_RATES = [12, 24, 36]


@register
class DeepLabV3Plus(BaseDecoder):
    """
    Implementation of the DeepLabV3+ decoder. Paper: https://arxiv.org/abs/1802.02611
    To make it compatible with any encoder, the first skip connection is with
    the "middle" feature and the second one with the last.

    Args:
        input_channels: The channels of the input features.
        input_reductions: The reduction factor of the input features.
        decoder_channel: The channel to use on the decoder.
        atrous_rates: The atrous rates to use on the ASPP module.
        norm_layer: Normalization layer to use.
        activation_layer: Activation function to use.
        extra_layer: Addional layer to use.
        return_features: Whether to return the intermediate results of the decoder.

    """

    def __init__(
        self,
        input_channels: List[int],
        input_reductions: List[int],
        decoder_channels: List[int] = DEFAULT_CHANNELS,
        atrous_rates: List[int] = DEFAULT_ATROUS_RATES,
        upsample_layer: Type[nn.Module] = up.Upsample,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation_layer: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
        mismatch_layer: Type[nn.Module] = mm.Pad,
        return_features: bool = False,
    ):
        super().__init__(input_channels, input_reductions, return_features)
        self.skip_idxes = self._get_skip_indexes(input_reductions)
        skip_reds = self._get_skip_reductions(input_reductions)
        skip_chans = self._get_skip_channels(input_channels)
        self._out_classes = decoder_channels[-1]
        specs = norm_layer, activation_layer, extra_layer

        uplays = self._format_upsample_layers(skip_reds, upsample_layer)

        init_ch = decoder_channels[0]
        skip_ch = skip_chans[-1]
        up_layer = uplays[-1]

        aspp = ASPP(skip_ch, init_ch, atrous_rates)
        conv = DWSConvNormActivation(init_ch, init_ch, 3, *specs)
        up = up_layer(init_ch)
        self.aspp_block = nn.Sequential(aspp, conv, up)

        final_ch = decoder_channels[-1]
        skip_ch = skip_chans[-2]
        up_layer = uplays[-2]

        # Setting up the skip connection
        self.skip_block = md.ConvNormAct(skip_ch, skip_ch, 1, *specs)
        self.fix_mismatch = mismatch_layer()

        # Setting up the final block
        conv = DWSConvNormActivation(skip_ch + init_ch, final_ch, 3, *specs)
        up = up_layer(final_ch)
        self.final_block = nn.Sequential(conv, up)

    @size_control
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        x = features[self.skip_idxes[-1]]
        x = self.aspp_block(x)

        skip = features[self.skip_idxes[0]]
        skip = self.skip_block(skip)
        x, skip = self.fix_mismatch(x, skip)

        if self.return_features:
            return [skip, x]

        x = torch.cat([x, skip], dim=1)

        x = self.final_block(x)
        return x

    @property
    def out_classes(self) -> int:
        return self._out_classes

    def _get_skip_indexes(self, input_reductions: List[int]) -> List[int]:
        n_layers = len(input_reductions)
        return [n_layers // 2, n_layers - 1]

    def _get_skip_reductions(self, input_reductions: List[int]) -> List[int]:
        return [input_reductions[idx] for idx in self.skip_idxes]

    def _get_skip_channels(self, input_channels: List[int]) -> List[int]:
        return [input_channels[idx] for idx in self.skip_idxes]

    def _format_upsample_layers(self, skip_reductions, upsample_layer):
        skip_reductions = [1] + skip_reductions
        upsample_layers = []
        for i in range(1, len(skip_reductions)):
            scale = skip_reductions[i] // skip_reductions[i - 1]

            # Partially initialize the upsample layer with the scale
            layer = partial(upsample_layer, scale=scale)
            upsample_layers.append(layer)

        return upsample_layers
