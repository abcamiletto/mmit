from __future__ import annotations

import warnings
from functools import wraps

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseDecoder(nn.Module):
    def forward(self, *features: Tensor) -> Tensor:
        raise NotImplementedError


def size_control(func):
    @wraps(func)
    def wrapper(self, *features, **kwargs):
        img_size = features[0].shape[-2:]

        output = func(self, *features, **kwargs)

        # Raise a warnin if size is very different
        out_size = output.shape[-2:]
        for outdim, featdim in zip(out_size, img_size):
            if outdim < featdim / 1.5 or outdim > featdim * 1.5:
                warnings.warn(
                    f"""
                    End Resizing Warning: Something might be wrong with the decoder.
                    Output shape: {out_size} - Input shape: {img_size}
                    """
                )

        # If output shape doesn't match first feature shape, interpolate
        if out_size != img_size:
            output = F.interpolate(output, size=img_size)

        return output

    return wrapper
