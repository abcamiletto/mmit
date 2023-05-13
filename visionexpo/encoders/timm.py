from __future__ import annotations

import timm

from ..base import BaseEncoder


class TimmEncoder(BaseEncoder):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 5,
        output_stride: int = 32,
        weights: str = None,
    ):
        super().__init__()

        model_kwargs = {
            "pretrained": pretrained,
            "in_chans": in_channels,
            "features_only": True,
            "out_indices": tuple(range(depth)),
        }

        if output_stride != 32:
            model_kwargs["output_stride"] = output_stride

        self.model = timm.create_model(name, **model_kwargs)
        self.in_channels = in_channels

    def forward(self, x):
        features = self.model(x)

        if x.dtype != features[0].dtype:
            x = x.to(features[0].dtype)

        return [x] + features

    @property
    def out_channels(self):
        feature_channels = self.model.feature_info.channels()
        in_channels = self.in_channels
        return [in_channels] + feature_channels

    @property
    def out_strides(self):
        feature_strides = self.model.feature_info.strides()
        in_stride = 1
        return [in_stride] + feature_strides
