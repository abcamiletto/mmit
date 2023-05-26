from typing import Type

import torch.nn as nn

from mmit.factory import register

__all__ = ["ClassificationHead"]


@register
class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        activation_layer: Type[nn.Module] = nn.ReLU,
        pooling_layer: Type[nn.Module] = nn.AdaptiveAvgPool2d,
    ):
        super().__init__()
        self.avg_pool = pooling_layer(1)
        self.fc = nn.Linear(in_channels, num_classes)
        self.activation = activation_layer()

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.activation(x)
