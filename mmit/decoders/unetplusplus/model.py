from functools import partial
from typing import List, Type

import torch
import torch.nn as nn

from mmit.base import mismatch as mm
from mmit.base import upsamplers as up
from mmit.factory import register

from ..basedecoder import BaseDecoder
from ..unet.parts import UBlock
from ..utils.resize import size_control

__all__ = ["UNetPlusPlus"]

DEFAULT_CHANNELS = (256, 128, 64, 32, 16, 8)


@register
class UNetPlusPlus(BaseDecoder):
    """
    Implementation of the U-Net++ decoder. Paper: https://arxiv.org/abs/1807.10165.

    In this implementation, we follow the following naming convention referring to Figure 1.a in the paper:
        - `lidx` is the layer index, i.e. the index that spans horizontally.
        - `didx` is the depth index, i.e. the index that spans vertically.
        - `i_j` will be the key of the block that is at depth `i` and layer `j`.
        - `i_j` will also be the key of the resulting tensor after the block that is at depth `i` and layer `j`.

    Since we implement only the decoder, there will be no `i_0` blocks, and the `i_0` tensors will be the input tensors.

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
        decoder_channels: List[int] = None,
        upsample_layer: Type[nn.Module] = up.ConvTranspose2d,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation_layer: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
        mismatch_layer: Type[nn.Module] = mm.Pad,
        return_features: bool = False,
    ):
        super().__init__(input_channels, input_reductions, return_features)

        self.depth = len(input_channels)
        if decoder_channels is None:
            decoder_channels = DEFAULT_CHANNELS[: self.depth - 1]

        channels = self._format_channels(input_channels, decoder_channels)
        up_lays = self._format_upsample_layers(input_reductions, upsample_layer)
        specs = norm_layer, activation_layer, extra_layer, mismatch_layer

        blocks = {}
        for key, (in_ch, skip_ch, out_ch) in channels.items():
            didx = int(key.split("_")[0])
            up_lay = up_lays[didx]
            blocks[key] = UBlock(in_ch, skip_ch, out_ch, up_lay, *specs)

        self.blocks = nn.ModuleDict(blocks)
        self._out_classes = out_ch
        self.return_features = return_features

    @size_control
    def forward(self, *features):
        features = self._preprocess_features(features)
        for lidx in range(1, self.depth):
            for didx in range(self.depth - lidx):
                # Get the corresponding block
                block = self.blocks[f"{didx}_{lidx}"]

                input_tensor = features[f"{didx+1}_{lidx-1}"]
                skip_tensors = [features[f"{didx}_{li}"] for li in range(lidx)]
                skip_tensors = torch.cat(skip_tensors, dim=1)

                # Apply the block to the input tensor
                output = block(input_tensor, skip_tensors)

                # Store the output in the dictionary
                features[f"{didx}_{lidx}"] = output

        # The final output is the output of the last block
        final_output = features[f"0_{self.depth-1}"]

        if self.return_features:
            inters = [features[f"0_{lidx}"] for lidx in range(1, self.depth)]
            return inters

        return final_output

    @property
    def out_classes(self) -> int:
        return self._out_classes

    def _preprocess_features(self, features):
        """Preprocess features by removing the first feature and reversing the order."""
        features = {f"{i}_0": f for i, f in enumerate(features)}
        return features

    def _format_channels(self, input_channels, decoder_channels):
        decoder_channels = decoder_channels[::-1]

        # We start by defining the input, skip and output channels for the first layer
        in_channels = {f"{i}_1": c for i, c in enumerate(input_channels[1:])}
        skip_channels = {f"{i}_1": c for i, c in enumerate(input_channels[:-1])}
        out_channels = {f"{i}_1": c for i, c in enumerate(decoder_channels)}

        # We recursively define the input, skip and output channels for the other layers
        for lidx in range(2, self.depth):
            for didx in range(self.depth - lidx):
                # We define the keys of the current block and its neighbors
                current = f"{didx}_{lidx}"
                bottom_left = f"{didx+1}_{lidx-1}"
                left = f"{didx}_{lidx-1}"

                # We define the input, skip and output channels of the current block
                in_channels[current] = out_channels[bottom_left]
                skip_channels[current] = skip_channels[left] + out_channels[left]
                out_channels[current] = out_channels[left]

        return {k: (in_channels[k], skip_channels[k], out_channels[k]) for k in in_channels}

    def _format_upsample_layers(self, input_reductions, upsample_layer):
        # We build a mask to filter out the layers that don't need upsampling
        upsample_layers = []
        for i in range(1, len(input_reductions)):
            scale = input_reductions[i] // input_reductions[i - 1]

            # Partially initialize the upsample layer with the scale
            layer = partial(upsample_layer, scale=scale)
            upsample_layers.append(layer)

        return upsample_layers
