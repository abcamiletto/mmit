from __future__ import annotations

from typing import List, Union

import torch.nn as nn
from torch import Tensor

from .utils import interpolate_to_match, pad_to_match


class BaseDecoder(nn.Module):
    def __init__(
        self,
        input_channels: List[int],
        input_reductions: List[int],
        mismatch_handling_mode: str = "pad",
    ):
        super().__init__()
        if len(input_channels) != len(input_reductions):
            raise ValueError(
                "The number of input channels and input reductions must match."
            )

        self.input_channels = input_channels
        self.input_reductions = input_reductions
        self.mismatch_handling_mode = mismatch_handling_mode

        if mismatch_handling_mode == "pad":
            self.fix_size = pad_to_match
        elif mismatch_handling_mode == "interpolate":
            self.fix_size = interpolate_to_match
        else:
            raise ValueError(
                f"Unknown mismatch handling mode: {mismatch_handling_mode}."
            )

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
