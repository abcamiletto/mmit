from .unet.model import UNet
from .unetplusplus.model import UNetPlusPlus

__all__ = ["UNet", "UNetPlusPlus"]


def list_decoders():
    return [s.lower() for s in __all__]


def create_decoder(name, *args, **kwargs):
    name = name.lower()
    if name == "unet":
        return UNet(*args, **kwargs)
    elif name == "unetplusplus":
        return UNetPlusPlus(*args, **kwargs)
    else:
        raise ValueError(f"Unknown decoder {name}.")
