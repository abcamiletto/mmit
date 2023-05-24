import warnings
from typing import Type

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Pad", "Interpolate"]


class Pad(nn.Module):
    def __init__(self, warn_if_resize_too_big=True):
        super().__init__()
        self.warn_if_resize_too_big = warn_if_resize_too_big

    def forward(self, x, skip):
        x_size, skip_size = x.shape[2:], skip.shape[2:]

        if x_size == skip_size:
            return x, skip

        if self.warn_if_resize_too_big:
            check_if_resizing_is_too_big(x_size, skip_size)

        padding = self._get_padding(x_size, skip_size)

        x = F.pad(x, padding)
        return x, skip

    def _get_padding(self, x_size, skip_size):
        vpad = skip_size[0] - x_size[0]
        hpad = skip_size[1] - x_size[1]
        lpad = hpad // 2
        rpad = hpad - lpad
        tpad = vpad // 2
        bpad = vpad - tpad

        return lpad, rpad, tpad, bpad


class Interpolate(nn.Module):
    def __init__(self, mode="bilinear", warn_if_resize_too_big=True):
        super().__init__()
        self.mode = mode
        self.warn_if_resize_too_big = warn_if_resize_too_big

    def forward(self, x, skip):
        x_size, skip_size = x.shape[2:], skip.shape[2:]

        if x_size == skip_size:
            return x, skip

        if self.warn_if_resize_too_big:
            check_if_resizing_is_too_big(x_size, skip_size)

        x = F.interpolate(x, size=skip_size, mode=self.mode)
        return x, skip


def check_if_resizing_is_too_big(img_size, out_size):
    is_resizing_too_big = any(
        outdim < featdim / 1.5 or outdim > featdim * 1.5
        for outdim, featdim in zip(out_size, img_size)
    )

    if is_resizing_too_big:
        warnings.warn(
            f"""
            End Resizing Warning: Something might be wrong with the decoder.
            Output shape: {out_size} - Input shape: {img_size}
            """
        )


def get_mismatch_class(mismatch_layer_name: str) -> Type[nn.Module]:
    """
    Create a mismatch layer based on the provided mismatch layer name.

    Args:
        mismatch_layer_name: Name of the mismatch layer.

    Returns:
        Class of the mismatch layer.

    Raises:
        ValueError: If an invalid mismatch layer name is provided.

    Available mismatch layer names:
        - 'pad': Pad layer.
        - 'interpolate': Interpolate layer.
    """
    mismatch_builder = {
        "pad": Pad,
        "interpolate": Interpolate,
    }

    try:
        return mismatch_builder[mismatch_layer_name]
    except KeyError:
        raise ValueError(
            f"Invalid mismatch layer: {mismatch_layer_name}. "
            f"Available options are: {', '.join(mismatch_builder.keys())}"
        )
