import torch
from torch import nn

from .parts import UpBlock

DEFAULT_CHANNELS = (256, 128, 64, 32, 16)


class UNetDecoder(nn.Module):
    def __init__(
        self,
        input_channels: list[int],
        decoder_channels: list[int] = None,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
        extra_layer: nn.Module = nn.Identity(),
    ):
        super().__init__()

        if decoder_channels is None:
            decoder_channels = DEFAULT_CHANNELS[: len(input_channels) - 1]

        in_ch, skip_ch, out_ch = self.format_channels(input_channels, decoder_channels)

        blocks = []
        for ic, sc, oc in zip(in_ch, skip_ch, out_ch):
            upblock = UpBlock(ic, sc, oc, norm_layer, activation, extra_layer)
            blocks.append(upblock)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        # Dropping the first channel since we don't use the input image
        features = features[1:]

        # Reversing the input channels since we're going from the bottom up
        features = features[::-1]

        skips = features[1:]
        x = features[0]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x

    def format_channels(self, input_channels, decoder_channels):
        # We drop the first channel since we don't use the input image
        input_channels = input_channels[1:]

        # We reverse the input channels since we're going from the bottom up
        input_channels = input_channels[::-1]

        # On the last layer we don't have a skip connection
        skip_channels = input_channels[1:] + [0]

        return input_channels, skip_channels, decoder_channels
