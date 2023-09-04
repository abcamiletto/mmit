from functools import partial
from typing import List, Type

import torch.nn as nn

from mmit.base import mismatch as mm
from mmit.base import upsamplers as up
from mmit.factory import register

from ..basedecoder import BaseDecoder
from ..utils import size_control
from .parts import LinkBlock

__all__ = ["LinkNet"]

DEFAULT_CHANNEL = 32


@register
class LinkNet(BaseDecoder):
    """
    Implementation of the Linknet decoder. Paper: https://arxiv.org/abs/1707.03718

    Args:
        input_channels: The channels of the input features.
        input_reductions: The reduction factor of the input features.
        decoder_channel: The channel for the output of the decoder.
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
        upsample_layer: Type[nn.Module] = up.ConvTranspose2d,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation_layer: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
        mismatch_layer: Type[nn.Module] = mm.Pad,
        return_features: bool = False,
    ):
        super().__init__(input_channels, input_reductions, return_features)
        self._out_classes = decoder_channel
        in_ch, out_ch = self._format_channels(input_channels, decoder_channel)
        up_lays = self._format_upsample_layers(input_reductions, upsample_layer)
        specs = norm_layer, activation_layer, extra_layer, mismatch_layer

        blocks = []
        for ic, oc, up_lay in zip(in_ch, out_ch, up_lays):
            block = LinkBlock(ic, oc, up_lay, *specs)
            blocks.append(block)

        self.stages = nn.ModuleList(blocks)

    @size_control
    def forward(self, *features):
        features = features[1:]  # remove first skip
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skips = features[1:]

        inters = []

        for i, decoder_block in enumerate(self.stages):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

            if self.return_features:
                inters.append(x)

        return x if not self.return_features else inters

    @property
    def out_classes(self) -> int:
        return self._out_classes

    def _format_channels(self, input_channels: List[int], decoder_channel: int):
        # remove first skip
        input_channels = input_channels[1:]
        # reverse channels to start from head of encoder
        input_channels = input_channels[::-1]

        out_channels = input_channels[1:] + [decoder_channel]

        return input_channels, out_channels

    def _format_upsample_layers(self, input_reductions, upsample_layer):
        # We reverse the input reductions since we're going from the bottom up
        input_reductions = input_reductions[::-1]

        # We build a mask to filter out the layers that don't need upsampling
        upsample_layers = []
        for i in range(1, len(input_reductions)):
            scale = input_reductions[i - 1] // input_reductions[i]

            # Partially initialize the upsample layer with the scale
            layer = partial(upsample_layer, scale=scale)
            upsample_layers.append(layer)

        return upsample_layers
