from typing import Type

import torch.nn as nn

__all__ = ["Upsample", "ConvTranspose2d"]


class Upsample(nn.Upsample):
    def __init__(self, in_channels: int, scale: int = 2):
        super().__init__(scale_factor=scale, mode="bilinear")


class ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        scale: int = 2,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=scale,
            stride=scale,
        )


def get_upsamples_class(upsampler_layer_name: str) -> Type[nn.Module]:
    """
    Create an upsampler layer based on the provided upsampler layer name.

    Args:
        upsampler_layer_name: Name of the upsampler layer.

    Returns:
        Class of the upsampler layer.

    Raises:
        ValueError: If an invalid upsampler layer name is provided.

    Available upsampler layer names:
        - 'interpolate': Upsample layer.
        - 'convtransposed': ConvTranspose2d layer.
    """
    upsampler_builder = {
        "interpolate": Upsample,
        "convtransposed": ConvTranspose2d,
    }

    try:
        return upsampler_builder[upsampler_layer_name]
    except KeyError:
        raise ValueError(
            f"Invalid upsampler layer: {upsampler_layer_name}. "
            f"Available options are: {', '.join(upsampler_builder.keys())}"
        )
