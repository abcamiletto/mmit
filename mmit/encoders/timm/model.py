from __future__ import annotations

import timm

from mmit.factory import register

from ..basencoder import BaseEncoder

__all__ = ["TimmEncoder"]


@register
class TimmEncoder(BaseEncoder):
    """Wrapper for timm encoders. Inspired by segmentation_models_pytorch."""

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        layers: int | tuple = 5,
        output_stride: int = 32,
        weights: str = None,
    ):
        super().__init__()

        if isinstance(layers, (int, float)):
            layers = tuple(range(layers))

        model_kwargs = {
            "pretrained": pretrained,
            "in_chans": in_channels,
            "features_only": True,
            "out_indices": layers,
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
    def out_reductions(self):
        feature_strides = self.model.feature_info.reduction()
        in_stride = 1
        return [in_stride] + feature_strides
