from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class BaseDecoder(nn.Module):
    def forward(self, *features: Tensor) -> Tensor:
        raise NotImplementedError
