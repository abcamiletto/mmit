import torch.nn as nn

activation_builder = {
    "relu": nn.ReLU,
    "leaky-relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "none": nn.Identity,
}
