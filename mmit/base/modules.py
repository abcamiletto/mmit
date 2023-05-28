from typing import Type

import torch.nn as nn
from torch import Tensor


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        extra_layer: Type[nn.Module] = nn.Identity,
        upsample_layer: Type[nn.Module] = nn.Identity,
        use_bias: bool = False,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=use_bias,
        )
        self.norm = norm_layer(out_channels)
        self.activation = activation()
        self.extra = extra_layer()
        self.upsample = upsample_layer(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.extra(x)
        x = self.upsample(x)
        return x
