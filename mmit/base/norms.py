import torch.nn as nn

norm_builder = {
    "batchnorm": nn.BatchNorm2d,
    "instance": nn.InstanceNorm2d,
    "layer": nn.LayerNorm,
    "group": nn.GroupNorm,
    "none": nn.Identity,
}
