import torch.nn as nn


class Upsample(nn.Upsample):
    def __init__(self, in_channels: int, scale: int = 2):
        super().__init__(scale_factor=scale, mode="bilinear")


class ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        scale: int = 2,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=scale,
            stride=scale,
        )
