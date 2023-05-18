from __future__ import annotations

from typing import List, Union

import torch.nn as nn
from torch import Tensor


class BaseDecoder(nn.Module):
    def __init__(
        self,
        input_channels: List[int],
        input_reductions: List[int],
        mismatch_fix_strategy: str = "pad",
    ):
        super().__init__()
        self._validate_input(input_channels, input_reductions)

        self.input_channels = input_channels
        self.input_reductions = input_reductions
        self.mismatch_handling_mode = mismatch_fix_strategy

    def forward(self, *features: Tensor) -> Union[Tensor, List[Tensor]]:
        """Forward pass of the decoder.

        Args:
            *features (Tensor): Features from the encoder.
            The first feature is the one with the highest resolution.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_channels={self.input_channels}, "
            f"input_reductions={self.input_reductions}, "
            f"mismatch_handling_mode={self.mismatch_handling_mode}"
            f")"
        )

    def _validate_input(self, channels: List[int], reductions: List[int]):
        if len(channels) != len(reductions):
            raise ValueError(
                "The number of input channels and input reductions must match."
            )
        if len(channels) == 0:
            raise ValueError("The number of input channels must be greater than 0.")

        if len(channels) > 6:
            raise ValueError("The number of input features must be less than 6.")

        # Check if reductions are powers of 2
        for reduction in reductions:
            if reduction < 1:
                raise ValueError("The input reduction must be greater or equal to 1.")
            elif reduction & (reduction - 1) != 0:
                raise ValueError("The input reduction must be a power of 2.")
