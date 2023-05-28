from typing import Type

import torch.nn as nn


def get_activation_class(activation_layer_name: str) -> Type[nn.Module]:
    """
    Create an activation layer based on the provided activation layer name.

    Args:
        Name of the activation layer.

    Returns:
        Class of the activation layer.

    Raises:
        ValueError: If an invalid activation layer name is provided.


    Available activation layer names:
        - 'relu': ReLU activation.
        - 'leaky_relu': LeakyReLU activation.
        - 'elu': ELU activation.
        - 'selu': SELU activation.
        - 'none': Identity activation.
    """
    activation_builder = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "none": nn.Identity,
    }

    try:
        return activation_builder[activation_layer_name]
    except KeyError:
        raise ValueError(
            f"Invalid activation layer: {activation_layer_name}. "
            f"Available options are: {', '.join(activation_builder.keys())}"
        )
