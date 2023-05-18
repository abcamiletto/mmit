from __future__ import annotations

import warnings
from functools import wraps

import torch.nn.functional as F
from torch import Tensor


def pad_to_match(x: Tensor, skip: Tensor) -> Tensor:
    x_size, skip_size = x.shape[2:], skip.shape[2:]

    if x_size == skip_size:
        return x, skip

    hpad = skip_size[0] - x_size[0]
    vpad = skip_size[1] - x_size[1]
    lpad = hpad // 2
    rpad = hpad - lpad
    tpad = vpad // 2
    bpad = vpad - tpad

    padding = (lpad, rpad, tpad, bpad)

    x = F.pad(x, padding)
    return x, skip


def interpolate_to_match(x: Tensor, skip: Tensor) -> Tensor:
    x_size, skip_size = x.shape[2:], skip.shape[2:]

    if x_size == skip_size:
        return x, skip

    x = F.interpolate(x, size=skip_size, mode="bilinear")
    return x, skip


def size_control(func):
    @wraps(func)
    def wrapper(self, *features, **kwargs):
        img_size = features[0].shape[-2:]

        output = func(self, *features, **kwargs)

        # Raise a warnin if size is very different
        out_size = output.shape[-2:]
        check_if_resizing_is_too_big(img_size, out_size)

        # If output shape doesn't match first feature shape, interpolate
        if out_size != img_size:
            output = F.interpolate(output, size=img_size)

        return output

    return wrapper


def check_if_resizing_is_too_big(img_size, out_size):
    for outdim, featdim in zip(out_size, img_size):
        if outdim < featdim / 1.5 or outdim > featdim * 1.5:
            warnings.warn(
                f"""
                End Resizing Warning: Something might be wrong with the decoder.
                Output shape: {out_size} - Input shape: {img_size}
                """
            )
