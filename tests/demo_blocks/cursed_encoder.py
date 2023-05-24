import torch

from mmit.encoders.basencoder import BaseEncoder


class CursedNet(BaseEncoder):
    def __init__(self, *args, in_chans=3, out_indices=tuple(range(6)), **kwargs):
        super().__init__()
        self._out_channels = [in_chans, 19, 71, 131, 223, 419]
        self._out_reductions = [1, 1, 8, 8, 16, 32]
        self.out_indices = out_indices

    def forward(self, x):
        B, C, H, W = x.shape

        outputs = [x]
        for ch, red in zip(self.out_channels[1:], self.out_reductions[1:]):
            h, w = H // red, W // red
            outputs.append(torch.rand(B, ch, h, w))

        return outputs

    @property
    def out_channels(self):
        return [self._out_channels[i] for i in self.out_indices]

    @property
    def out_reductions(self):
        return [self._out_reductions[i] for i in self.out_indices]
