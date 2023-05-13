import torch.nn as nn


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(self, x):
        return self.conv(x)


class Identity(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()

    def forward(self, x):
        return x
