from typing import Type

import torch.nn as nn


def get_extra_class(extra_layer_name: str) -> Type[nn.Module]:
    """
    Create an extra layer based on the provided extra layer name.

    Args:
        extra_layer_name: Name of the extra layer.

    Returns:
        Class of the extra layer.

    Raises:
        ValueError: If an invalid extra layer name is provided.

    Available extra layer names:
        - 'none': Identity layer.
    """
    extra_builder = {
        "none": nn.Identity,
    }

    try:
        return extra_builder[extra_layer_name]
    except KeyError:
        raise ValueError(
            f"Invalid extra layer: {extra_layer_name}. "
            f"Available options are: {', '.join(extra_builder.keys())}"
        )
