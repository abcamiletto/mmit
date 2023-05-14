import torch
import torch.nn as nn

from ...base import upsamplers as up
from ..basedecoder import BaseDecoder, size_control
from ..unet.parts import UpBlock

DEFAULT_CHANNELS = (256, 128, 64, 32, 16)


class UNetPlusPlus(BaseDecoder):
    def __init__(
        self,
        input_channels: list[int],
        input_reductions: list[int],
        decoder_channels: list[int] = None,
        upsample_layer: nn.Module = up.ConvTranspose2d,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
        extra_layer: nn.Module = nn.Identity(),
    ):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = DEFAULT_CHANNELS[: len(input_channels) - 1]

        in_ch, skip_ch, out_ch = self.format_channels(input_channels, decoder_channels)
        self.in_channels = in_ch
        self.skip_channels = skip_ch
        self.out_channels = out_ch

        # combine decoder keyword arguments
        specs = upsample_layer, norm_layer, activation, extra_layer

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                ic = in_ch[layer_idx] if depth_idx == 0 else skip_ch[layer_idx - 1]
                sc = skip_ch[layer_idx] * (layer_idx + 1 - depth_idx)
                oc = out_ch[layer_idx] if depth_idx == 0 else skip_ch[layer_idx]

                bname = f"x_{depth_idx}_{layer_idx}"
                blocks[bname] = UpBlock(ic, sc, oc, *specs)

        # Add final block
        bname = f"x_{0}_{len(in_ch)-1}"
        blocks[bname] = UpBlock(in_ch[-1], 0, out_ch[-1], *specs)

        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(in_ch) - 1

    @size_control
    def forward(self, *features):
        features = self._preprocess_features(features)
        intermediate_outputs = self._generate_intermediate_outputs(features)
        final_output = self._process_final_layer(intermediate_outputs)
        return final_output

    def _preprocess_features(self, features):
        """Preprocess features by removing the first feature and reversing the order."""
        features_without_first = features[1:]  # Remove first feature
        reversed_features = features_without_first[::-1]  # Reverse feature order
        return reversed_features

    def _generate_intermediate_outputs(self, features):
        """Generate intermediate outputs for each layer in the network."""
        intermediate_outputs = {}  # Dictionary to store intermediate outputs

        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                output_key = f"x_{depth_idx}_{depth_idx + layer_idx}"
                output_value = self._process_layer(
                    depth_idx, layer_idx, intermediate_outputs, features
                )
                intermediate_outputs[output_key] = output_value

        return intermediate_outputs

    def _process_layer(self, depth_idx, layer_idx, intermediate_outputs, features):
        """Process a single layer in the network."""
        if layer_idx == 0:
            return self._process_first_layer(depth_idx, features)

        concat_features = self._concatenate_features(
            depth_idx, layer_idx, intermediate_outputs, features
        )
        prev_output_key = f"x_{depth_idx}_{depth_idx + layer_idx - 1}"
        current_block_key = f"x_{depth_idx}_{depth_idx + layer_idx}"

        prev_output = intermediate_outputs[prev_output_key]
        current_block = self.blocks[current_block_key]

        return current_block(prev_output, concat_features)

    def _process_first_layer(self, depth_idx, features):
        """Process the first layer in the network."""
        input1 = features[depth_idx]
        input2 = features[depth_idx + 1]
        current_block = self.blocks[f"x_{depth_idx}_{depth_idx}"]
        return current_block(input1, input2)

    def _concatenate_features(self, depth_idx, layer_idx, inter_outputs, features):
        """Concatenate intermediate features and the next feature in the sequence."""
        intermediate_keys = [
            f"x_{idx}_{depth_idx + layer_idx}"
            for idx in range(depth_idx + 1, depth_idx + layer_idx + 1)
        ]
        intermediate_features = [inter_outputs[key] for key in intermediate_keys]
        next_feature = features[depth_idx + layer_idx + 1]
        return torch.cat(intermediate_features + [next_feature], dim=1)

    def _process_final_layer(self, inter_outputs):
        """Process the final layer in the network."""
        final_input_key = f"x_{0}_{self.depth - 1}"
        final_input = inter_outputs[final_input_key]
        final_block = self.blocks[f"x_{0}_{self.depth}"]
        return final_block(final_input)

    def format_channels(self, input_channels, decoder_channels):
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
