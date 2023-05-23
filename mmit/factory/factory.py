from typing import List

import torch.nn as nn

from .components import build_components
from .registry import get_decoder, get_encoder

__all__ = ["create_encoder", "create_decoder"]


class Factory:
    out_channels: List[int]
    out_reductions: List[int]

    @classmethod
    def create_encoder(cls, name: str, **kwargs) -> nn.Module:
        """Create an encoder from a name and kwargs."""
        Encoder = get_encoder(name)
        encoder = Encoder(**kwargs)
        cls.out_channels = encoder.out_channels
        cls.out_reductions = encoder.out_reductions
        return encoder

    @classmethod
    def create_decoder(cls, name: str, **kwargs) -> nn.Module:
        Decoder = get_decoder(name)
        components = build_components(kwargs)

        kwargs.update(components)
        out_channels = kwargs.pop("out_channels", cls.out_channels)
        out_reductions = kwargs.pop("out_reductions", cls.out_reductions)
        return Decoder(out_channels, out_reductions, **kwargs)


create_encoder = Factory.create_encoder
create_decoder = Factory.create_decoder
