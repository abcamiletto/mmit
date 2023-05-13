from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class BaseEncoder(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    def out_channels(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def out_strides(self) -> tuple[int, ...]:
        raise NotImplementedError
