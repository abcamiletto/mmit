from typing import Type

import torch.nn as nn

from mmit.base import mismatch as mm
from mmit.base import upsamplers as up


class SkipBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        upsample_layer: Type[nn.Module] = up.ConvTranspose2d,
        mismatch_layer: Type[nn.Module] = mm.Pad,
    ):
        super().__init__()
        self.up = upsample_layer(input_channels)
        self.skip_conv = nn.Conv2d(skip_channels, input_channels, kernel_size=1)
        self.fix_mismatch = mismatch_layer()

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            skip = self.skip_conv(skip)
            x, skip = self.fix_mismatch(x, skip)
            x = x + skip

        return x
