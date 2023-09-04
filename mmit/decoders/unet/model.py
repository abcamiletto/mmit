from functools import partial
from typing import List, Optional, Type

import torch
from torch import nn

from mmit.base import mismatch as mm
from mmit.base import upsamplers as up
from mmit.factory import register

from ..basedecoder import BaseDecoder
from ..utils.resize import size_control
from .parts import UBlock

__all__ = ["UNet"]

DEFAULT_CHANNELS = (256, 128, 64, 32, 16, 8)


@register
class UNet(BaseDecoder):
    """
    Implementation of the U-Net decoder. Paper: https://arxiv.org/abs/1505.04597

    Args:
        input_channels: The channels of the input features.
        input_reductions: The reduction factor of the input features.
        decoder_channels: The channels on each layer of the decoder.
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
        decoder_channels: Optional[List[int]] = None,
        upsample_layer: Type[nn.Module] = up.ConvTranspose2d,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation_layer: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
        mismatch_layer: Type[nn.Module] = mm.Pad,
        return_features: bool = False,
    ):
        super().__init__(input_channels, input_reductions, return_features)

        if decoder_channels is None:
            decoder_channels = DEFAULT_CHANNELS[: len(input_channels) - 1]

        in_ch, skip_ch, out_ch = self._format_channels(input_channels, decoder_channels)
        up_lays = self._format_upsample_layers(input_reductions, upsample_layer)

        specs = norm_layer, activation_layer, extra_layer, mismatch_layer
        blocks = []

        for ic, sc, oc, up_lay in zip(in_ch, skip_ch, out_ch, up_lays):
            ublock = UBlock(ic, sc, oc, up_lay, *specs)
            blocks.append(ublock)

        self.stages = nn.ModuleList(blocks)
        self._output_classes = out_ch[-1]

    @size_control
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        # Dropping the first channel since we don't use the input image
        features = features[1:]

        # Reversing the input channels since we're going from the bottom up
        features = features[::-1]

        skips = features[1:]
        x = features[0]

        inters = []
        for i, decoder_block in enumerate(self.stages):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

            if self.return_features:
                inters.append(x)

        return inters if self.return_features else x

    @property
    def out_classes(self) -> int:
        return self._output_classes

    def _format_channels(self, input_channels, decoder_channels):
        # We drop the first channel since we don't use the input image
        input_channels = input_channels[1:]

        # We reverse the input channels since we're going from the bottom up
        input_channels = input_channels[::-1]

        # On the last layer we don't have a skip connection
        skip_channels = input_channels[1:] + [0]

        # The first layer has the same number of channels as the input
        # The rest has the output channels of the previous layer
        in_channels = [input_channels[0]] + list(decoder_channels[:-1])

        return in_channels, skip_channels, decoder_channels

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
