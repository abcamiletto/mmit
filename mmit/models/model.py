import torch.nn as nn

__all__ = ["MmitModel"]


class MmitModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(*features)
        out = self.head(out)
        return out
