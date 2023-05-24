from typing import Dict


from ..base import (
    get_activation_class,
    get_extra_class,
    get_mismatch_class,
    get_norm_class,
    get_upsamples_class,
)


def build_components(cfg: Dict) -> Dict:
    """Build components from a config dict."""

    components = {}

    if "upsample_layer" in cfg:
        name = cfg["upsample_layer"]
        layer = get_upsamples_class(name) if isinstance(name, str) else name
        components["upsample_layer"] = layer

    if "norm_layer" in cfg:
        name = cfg["norm_layer"]
        layer = get_norm_class(name) if isinstance(name, str) else name
        components["norm_layer"] = layer

    if "activation_layer" in cfg:
        name = cfg["activation_layer"]
        layer = get_activation_class(name) if isinstance(name, str) else name
        components["activation_layer"] = layer

    if "mismatch_layer" in cfg:
        name = cfg["mismatch_layer"]
        layer = get_mismatch_class(name) if isinstance(name, str) else name
        components["mismatch_layer"] = layer

    if "extra_layer" in cfg:
        name = cfg["extra_layer"]
        layer = get_extra_class(name) if isinstance(name, str) else name
        components["extra_layer"] = layer

    return components
