from typing import List, Optional

import torch.nn as nn

from ..heads import heads_builder
from .components import build_components
from .registry import get_decoder, get_encoder

__all__ = ["create_encoder", "create_decoder", "create_model"]


class Factory:
    out_channels: List[int]
    out_reductions: List[int]

    @classmethod
    def create_encoder(
        cls,
        name: str,
        in_chans: int = 3,
        out_indices: Optional[tuple] = None,
        output_stride: Optional[int] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Build an encoder from a name and some keyword arguments.

        Args:
            name: The name of the encoder.
            in_chans: The number of input channels.
            out_indices: The indices of the feature maps to return.
            output_stride: The output stride of the encoder.
            kwargs: Keyword arguments for the encoder.
        """
        Encoder = get_encoder(name)
        encoder = Encoder(
            in_chans=in_chans,
            out_indices=out_indices,
            output_stride=output_stride,
            **kwargs,
        )
        cls.out_channels = encoder.out_channels
        cls.out_reductions = encoder.out_reductions
        return encoder

    @classmethod
    def create_decoder(
        cls,
        name: str,
        out_channels: Optional[int] = None,
        out_reductions: Optional[int] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Build a decoder from a name and some keyword arguments.

        Args:
            name: The name of the decoder.
            out_channels: The number of channels of the input tensors of the forward pass.
            out_reductions: The reduction factor of the input tensors of the forward pass.
        """
        Decoder = get_decoder(name)
        components = build_components(kwargs)

        kwargs.update(components)
        out_channels = out_channels or cls.out_channels
        out_reductions = out_reductions or cls.out_reductions
        return Decoder(out_channels, out_reductions, **kwargs)


create_encoder = Factory.create_encoder
create_decoder = Factory.create_decoder


def create_model(
    encoder_name: str,
    decoder_name: str,
    classes: int,
    task: str = "segmentation",
    encoder_cfg: dict = None,
    decoder_cfg: dict = None,
):
    """
    Build a model from an encoder and a decoder.

    Args:
        encoder_name: The name of the encoder.
        decoder_name: The name of the decoder.
        classes: The number of classes.
        task: The task of the model.
        encoder_cfg: Keyword arguments for the encoder.
        decoder_cfg: Keyword arguments for the decoder.
    """

    encoder_cfg = encoder_cfg or {}
    decoder_cfg = decoder_cfg or {}

    encoder = create_encoder(encoder_name, **encoder_cfg)
    decoder = create_decoder(decoder_name, **decoder_cfg)
    head = heads_builder[task](decoder.out_classes, classes)
    return nn.Sequential(encoder, decoder, head)
