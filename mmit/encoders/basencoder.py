from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class BaseEncoder(nn.Module):
    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward pass of the encoder

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            A list of tensors of shape (B, C, H // fi, W // fi) with the features.
            The first element is the input tensor, fi is the reduction factor of the i-th feature.
        """
        raise NotImplementedError

    @property
    def out_channels(self) -> tuple[int, ...]:
        """Number of channels of the output tensors"""
        raise NotImplementedError

    @property
    def out_reductions(self) -> tuple[int, ...]:
        """Reduction factor of the output tensors"""
        raise NotImplementedError
