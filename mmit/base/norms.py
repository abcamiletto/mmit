from typing import Type

import torch.nn as nn


def get_norm_class(norm_layer_name: str) -> Type[nn.Module]:
    """
    Create a normalization layer based on the provided normalization layer name.

    Args:
        norm_layer_name: Name of the normalization layer.

    Returns:
        Class of the normalization layer.

    Raises:
        ValueError: If an invalid normalization layer name is provided.

    Available normalization layer names:
        - 'batch': BatchNorm2d layer.
        - 'instance': InstanceNorm2d layer.
        - 'none': Identity layer.
    """
    norm_builder = {
        "batch": nn.BatchNorm2d,
        "instance": nn.InstanceNorm2d,
        "none": nn.Identity,
    }

    try:
        return norm_builder[norm_layer_name]
    except KeyError:
        raise ValueError(
            f"Invalid normalization layer: {norm_layer_name}. "
            f"Available options are: {', '.join(norm_builder.keys())}"
        )
