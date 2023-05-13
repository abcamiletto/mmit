import torch
from torch import nn

from .parts import UpBlock


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

        in_ch, out_ch = self.get_in_out_channels(input_channels, decoder_channels)

        blocks = []
        for ic, oc in zip(in_ch, out_ch):
            upblock = UpBlock(ic, oc, norm_layer, activation, extra_layer)
            blocks.append(upblock)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

    def get_in_out_channels(self, input_channels, decoder_channels):
        # We drop the first channel since we don't use the input image
        input_channels = input_channels[1:]

        # We reverse the input channels since we're going from the bottom up
        input_channels = input_channels[::-1]

        skip_channels = input_channels[1:] + [0]

        in_ch = [ic + sc for ic, sc in zip(input_channels, skip_channels)]
        out_ch = decoder_channels

        return in_ch, out_ch
