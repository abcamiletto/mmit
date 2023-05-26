import inspect
from functools import partial
from typing import Dict, List, Type

import timm
import torch.nn as nn

__all__ = [
    "list_encoders",
    "list_decoders",
    "register_decoder",
    "register_encoder",
    "register",
]


class Registry:
    _decoders: Dict[str, Type[nn.Module]] = {}
    _encoders: Dict[str, Type[nn.Module]] = {}
    _heads: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def _register(cls, item: Type[nn.Module], registry: Dict[str, Type[nn.Module]]):
        """Helper method to register encoders or decoders."""
        classname = item.__name__.lower()
        registry[classname] = item

    @classmethod
    def register_decoder(cls, decoder: Type[nn.Module]):
        cls._register(decoder, cls._decoders)

    @classmethod
    def register_encoder(cls, encoder: Type[nn.Module]):
        cls._register(encoder, cls._encoders)

    @classmethod
    def register_head(cls, head: Type[nn.Module]):
        cls._register(head, cls._heads)

    @classmethod
    def list_all_decoders(cls) -> List[str]:
        return list(cls._decoders.keys())

    @classmethod
    def list_all_encoders(cls) -> List[str]:
        return list(cls._encoders.keys())

    @classmethod
    def list_all_heads(cls) -> List[str]:
        return list(cls._heads.keys())

    @classmethod
    def get_decoder(cls, name: str) -> Type[nn.Module]:
        if name not in cls._decoders:
            raise KeyError(f"Decoder {name} is not registered")
        return cls._decoders[name]

    @classmethod
    def get_encoder(cls, name: str) -> Type[nn.Module]:
        # If the name is a timm model, return a partially initialized timm encoder
        if name in timm.list_models():
            encoder = cls._encoders["timmencoder"]
            return partial(encoder, name=name)

        if name not in cls._encoders:
            raise KeyError(f"Encoder {name} is not registered")

        return cls._encoders[name]

    @classmethod
    def get_head(cls, name: str) -> Type[nn.Module]:
        if name not in cls._heads:
            raise KeyError(f"Head {name} is not registered")
        return cls._heads[name]


register_decoder = Registry.register_decoder
list_decoders = Registry.list_all_decoders
get_decoder = Registry.get_decoder
register_encoder = Registry.register_encoder
list_encoders = Registry.list_all_encoders
get_encoder = Registry.get_encoder
register_head = Registry.register_head
list_heads = Registry.list_all_heads
get_head = Registry.get_head


def register(cls: Type[nn.Module]):
    """Decorator helper to register encoders or decoders classes in this package."""
    module_name = inspect.getmodule(cls).__name__

    # Split the module name by "." and check if parent name is in the list
    module_parts = module_name.split(".")
    if "decoders" in module_parts:
        Registry.register_decoder(cls)
    elif "encoders" in module_parts:
        Registry.register_encoder(cls)
    elif "heads" in module_parts:
        Registry.register_head(cls)
    else:
        raise ValueError(
            "Invalid module for registration. Class must be in a subpackage named 'encoders' or 'decoders'."
        )

    return cls
