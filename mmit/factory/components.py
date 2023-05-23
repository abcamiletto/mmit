from typing import Dict, Union

import torch.nn as nn

from ..base import (
    activation_builder,
    extra_builder,
    mismatch_builder,
    norm_builder,
    upsampler_builder,
)


def build_components(cfg: Dict) -> Dict:
    """Build components from a config dict."""

    components = {}

    if "upsample_layer" in cfg:
        name = cfg["upsample_layer"]
        components["upsample_layer"] = get_comp(upsampler_builder, name)

    if "norm_layer" in cfg:
        name = cfg["norm_layer"]
        components["norm_layer"] = get_comp(norm_builder, name)

    if "activation_layer" in cfg:
        name = cfg["activation_layer"]
        components["activation_layer"] = get_comp(activation_builder, name)

    if "mismatch_layer" in cfg:
        name = cfg["mismatch_layer"]
        components["mismatch_layer"] = get_comp(mismatch_builder, name)

    if "extra_layer" in cfg:
        name = cfg["extra_layer"]
        components["extra_layer"] = get_comp(extra_builder, name)

    return components


def get_comp(builder: Dict[str, nn.Module], name: Union[str, nn.Module]) -> nn.Module:
    """Get a component from a builder."""
    if isinstance(name, nn.Module):
        return name

    if name not in builder:
        raise KeyError(f"{name} is not in {builder.keys()}")

    return builder[name]
