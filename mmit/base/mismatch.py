import warnings

import torch.nn as nn
import torch.nn.functional as F


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
