from .unet.model import UNet

__all__ = ["UNet"]


def list_decoders():
    return [s.lower() for s in __all__]


def create_decoder(name, *args, **kwargs):
    name = name.lower()
    if name == "unet":
        return UNet(*args, **kwargs)
    else:
        raise ValueError(f"Unknown decoder {name}.")
