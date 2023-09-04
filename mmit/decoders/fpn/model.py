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
from .parts import SkipBlock

__all__ = ["FPN"]

DEFAULT_CHANNEL = 256


@register
class FPN(BaseDecoder):
    """
    Implementation of the FPN decoder. Paper: https://arxiv.org/abs/1612.03144

    Args:
        input_channels: The channels of the input features.
        input_reductions: The reduction factor of the input features.
        decoder_channel: The channel to use on the decoder.
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
        merge_mode: str = "sum",
        return_features: bool = False,
    ):
        super().__init__(input_channels, input_reductions, return_features)

        up_lays = self._format_upsample_layers(input_reductions, upsample_layer)

        specs = norm_layer, activation_layer, extra_layer, mismatch_layer
        input_channels = input_channels[1:][::-1]

        # Setting up the skip blocks
        skip_channels = input_channels[1:]
        skip_blocks = []
        for skip_channel, up_lay in zip(skip_channels, up_lays):
            block = SkipBlock(decoder_channel, skip_channel, up_lay)
            skip_blocks.append(block)

        self.skip_blocks = nn.ModuleList(skip_blocks)

        # Setting up the output blocks
        specs = norm_layer, activation_layer, extra_layer

        out_blocks = []
        for red in input_reductions[1:][::-1]:
            up_lay = partial(upsample_layer, scale=red)
            block = md.ConvNormAct(decoder_channel, decoder_channel, 3, *specs, up_lay)
            out_blocks.append(block)

        self.out_blocks = nn.ModuleList(out_blocks)

        # Input block for the first layer
        self.input_block = nn.Conv2d(input_channels[0], decoder_channel, 1)

        # Mismatch layer in case for weird sizes
        self.mismatch_layer = mismatch_layer()

        # Merge mode
        self.merge_mode = merge_mode
        if merge_mode not in ["sum", "cat"]:
            raise ValueError(f"Merge mode {merge_mode} not 'sum' or 'cat'.")

        # Output classes
        factor = len(out_blocks) if merge_mode == "cat" else 1
        self._out_classes = decoder_channel * factor

    @size_control
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        # Dropping the first channel since we don't use the input image
        features = features[1:]

        # Reversing the input channels since we're going from the bottom up
        features = features[::-1]

        # Splitting the features into the input map and the skip connections
        skips = features[1:]
        x = features[0]

        # We process the input map
        x = self.input_block(x)

        # We store build the pyramid of features
        out_maps = [x]
        for skip_block, skip_feature in zip(self.skip_blocks, skips):
            x = skip_block(x, skip_feature)
            out_maps.append(x)

        # We process the pyramid of features
        pyram = []
        for out_block, out_map in zip(self.out_blocks, out_maps):
            x = out_block(out_map)
            pyram.append(x.clone())

        # We fix the output sizes in case of weird shapes
        pyram = self._fix_output_sizes(pyram)

        if self.return_features:
            return pyram

        result = self._merge_pyramid(pyram)
        return result

    @property
    def out_classes(self) -> int:
        return self._out_classes

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

    def _fix_output_sizes(self, outputs):
        # We fix the sizes to be all exactly the same as the last one
        higher_res_output = outputs[-1]
        new_outputs = [higher_res_output]
        for output in outputs[:-1]:
            resized, _ = self.mismatch_layer(output, higher_res_output)
            new_outputs.append(resized)

        return new_outputs

    def _merge_pyramid(self, pyram):
        return torch.cat(pyram, dim=1) if self.merge_mode == "cat" else sum(pyram)
