from typing import List, Tuple

import torch.nn as nn
from torch import Tensor


class BaseEncoder(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        in_chans: int = 3,
        out_indices: tuple = (0, 1, 2, 3, 4),
        output_stride: int = 32,
    ):
        super().__init__()

    def forward(self, x: Tensor) -> List[Tensor]:
        """Forward pass of the encoder

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            A list of tensors of shape (B, C, H // fi, W // fi) with the features.
            The first element is the input tensor, fi is the reduction factor of the i-th feature.
        """
        raise NotImplementedError

    @property
    def out_channels(self) -> Tuple[int, ...]:
        """Number of channels of the output tensors"""
        raise NotImplementedError

    @property
    def out_reductions(self) -> Tuple[int, ...]:
        """Reduction factor of the output tensors"""
        raise NotImplementedError
