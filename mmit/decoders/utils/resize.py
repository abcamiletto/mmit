from __future__ import annotations

from functools import wraps

import torch.nn.functional as F

from .resizing_warning import check_if_resizing_is_too_big


def size_control(func):
    @wraps(func)
    def wrapper(self, *features, **kwargs):
        img_size = features[0].shape[-2:]

        output = func(self, *features, **kwargs)

        # If the user wants the features, return them
        if self.return_features:
            return output

        # Raise a warnin if size is very different
        out_size = output.shape[-2:]
        check_if_resizing_is_too_big(img_size, out_size)

        # If output shape doesn't match first feature shape, interpolate
        if out_size != img_size:
            output = F.interpolate(output, size=img_size)

        return output

    return wrapper
