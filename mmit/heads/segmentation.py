from typing import Type

import torch.nn as nn
from torch import Tensor

__all__ = ["SegmentationHead"]


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        activation_layer: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activation = activation_layer()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.conv(x))
