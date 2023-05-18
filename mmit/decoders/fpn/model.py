from functools import partial
from typing import List, Type

import torch
from torch import nn

from mmit.base import mismatch as mm
from mmit.base import upsamplers as up
from mmit.factory import register

from ..basedecoder import BaseDecoder
from ..utils import size_control
from .parts import ConvNormReLU, SkipBlock

__all__ = ["FPN"]

DEFAULT_CHANNEL = 256


@register
class FPN(BaseDecoder):
    def __init__(
        self,
        input_channels: List[int],
        input_reductions: List[int],
        decoder_channel: int = DEFAULT_CHANNEL,
        upsample_layer: Type[nn.Module] = up.ConvTranspose2d,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
        mismatch_layer: Type[nn.Module] = mm.Pad,
    ):
        super().__init__(input_channels, input_reductions)

        up_lays = self._format_upsample_layers(input_reductions, upsample_layer)

        specs = norm_layer, activation, extra_layer, mismatch_layer
        input_channels = input_channels[1:][::-1]

        # Setting up the skip blocks
        skip_channels = input_channels[1:]
        skip_blocks = []
        for skip_channel, up_lay in zip(skip_channels, up_lays):
            block = SkipBlock(decoder_channel, skip_channel, up_lay)
            skip_blocks.append(block)

        self.skip_blocks = nn.ModuleList(skip_blocks)

        # Setting up the output blocks
        specs = norm_layer, activation, extra_layer

        out_blocks = []
        for red in input_reductions[1:][::-1]:
            up_lay = partial(upsample_layer, scale=red)
            block = ConvNormReLU(decoder_channel, decoder_channel, up_lay, *specs)
            out_blocks.append(block)

        self.out_blocks = nn.ModuleList(out_blocks)

        # Input block for the first layer
        self.input_block = nn.Conv2d(input_channels[0], decoder_channel, 1)

        # Mismatch layer in case for weird sizes
        self.mismatch_layer = mismatch_layer()

    @size_control
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        input_image = features[0]

        # Dropping the first channel since we don't use the input image
        features = features[1:]

        # Reversing the input channels since we're going from the bottom up
        features = features[::-1]

        skips = features[1:]
        x = features[0]

        x = self.input_block(x)

        out_maps = [x]
        for skip_block, skip_feature in zip(self.skip_blocks, skips):
            x = skip_block(x, skip_feature)
            out_maps.append(x)

        outputs = []
        for out_block, out_map in zip(self.out_blocks, out_maps):
            x = out_block(out_map)
            outputs.append(x.clone())

        outputs = self._fix_output_sizes(outputs, input_image)

        result = torch.cat(outputs, dim=1)

        return result

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

    def _fix_output_sizes(self, outputs, input_image):
        new_outputs = []
        for output in outputs:
            resized, _ = self.mismatch_layer(output, input_image)
            new_outputs.append(resized)

        return new_outputs
