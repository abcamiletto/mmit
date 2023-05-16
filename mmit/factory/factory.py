from typing import List

import torch.nn as nn

from .registry import get_decoder, get_encoder

__all__ = ["create_encoder", "create_decoder"]


def create_encoder(name: str, **kwargs) -> nn.Module:
    """Create an encoder from a name and kwargs."""
    encoder = get_encoder(name)
    return encoder(**kwargs)


def create_decoder(
    name: str, out_channels: List[int], out_reductions: List[int]
) -> nn.Module:
    """Create a decoder from a name and kwargs."""
    decoder = get_decoder(name)
    return decoder(out_channels, out_reductions)
