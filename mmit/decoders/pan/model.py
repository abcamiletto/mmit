from functools import partial
from typing import List, Type

import torch.nn as nn

from mmit.base import mismatch as mm
from mmit.base import upsamplers as up
from mmit.factory import register

from ..basedecoder import BaseDecoder
from ..utils.resize import size_control
from .fpa import FPA
from .gau import GAU

__all__ = ["PAN"]

DEFAULT_CHANNEL = 32


@register
class PAN(BaseDecoder):
    """
    Implementation of the PAN decoder. Paper: https://arxiv.org/abs/1805.10180

    Args:
        input_channels: The channels of the input features.
        input_reductions: The reduction factor of the input features.
        decoder_channel: The channels on each layer of the decoder.
        upsample_layer: Layer to use for the upsampling.
        norm_layer: Normalization layer to use.
        activation_layer: Activation function to use.
        extra_layer: Addional layer to use.
        mismatch_layer: Strategy to deal with odd resolutions.
        return_features: Whether to return the intermediate results of the decoder.

    """

    def __init__(
        self,
        input_channels: List[int],
        input_reductions: List[int],
        decoder_channel: int = DEFAULT_CHANNEL,
        upsample_layer: Type[nn.Module] = up.Upsample,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation_layer: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
        mismatch_layer: Type[nn.Module] = mm.Pad,
        return_features: bool = False,
    ):
        super().__init__(input_channels, input_reductions, return_features)

        in_ch, out_ch = self._format_channels(input_channels, decoder_channel)
        up_lay = self._format_upsample_layer(input_reductions, upsample_layer)
        self._out_classes = out_ch[-1]

        specs = norm_layer, activation_layer, extra_layer

        self.fpa = FPA(in_ch[0], out_ch[0])

        gaus = []

        for ic, oc in zip(in_ch[1:], out_ch[1:]):
            gau = GAU(ic, oc, *specs)
            gaus.append(gau)

        self.gaus = nn.ModuleList(gaus)

        self.up = up_lay(out_ch[-1])

    @size_control
    def forward(self, *features):
        features = features[1:]
        bottleneck, features = features[-1], features[:-1]

        self._check_size(bottleneck)

        x = self.fpa(bottleneck)

        inters = [x]

        for feature, gau in zip(features[::-1], self.gaus):
            x = gau(feature, x)

            if self.return_features:
                inters.append(x)

        x = self.up(x)

        return x if not self.return_features else inters

    @property
    def out_classes(self) -> int:
        return self._out_classes

    def _format_channels(self, input_channels, decoder_channel):
        # We drop the first channel since we don't use the input image
        input_channels = input_channels[1:]

        # We reverse the input channels since we're going from the bottom up
        input_channels = input_channels[::-1]

        out_channels = [decoder_channel] * len(input_channels)
        return input_channels, out_channels

    def _format_upsample_layer(self, input_reductions, upsample_layer):
        # We need to find the scale for the last upsampling layer
        scale = input_reductions[1] // input_reductions[0]
        return partial(upsample_layer, scale=scale)

    def _check_size(self, bottleneck):
        h, w = bottleneck.shape[-2:]

        if h < 8 or w < 8:
            size = (h, w)
            raise ValueError(
                f"Bottleneck resolution is too small: {size}, should be at least 8x8. "
                f"Consider using a larger input image or a smaller encoder."
            )
