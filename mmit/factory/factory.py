from typing import List, Optional

import torch.nn as nn

from ..models import MmitModel
from .components import build_components
from .registry import get_decoder_class, get_encoder_class, get_head

__all__ = ["create_encoder", "create_decoder", "create_model"]


class Factory:
    encoder_channels: List[int]
    encoder_reductions: List[int]
    decoder_channels: int

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
            kwargs: Keyword arguments for the encoder. Take a look at the specific encoder docs for more info!
        """
        Encoder = get_encoder_class(name)

        kwargs["in_chans"] = in_chans
        if output_stride is not None:
            kwargs["output_stride"] = output_stride
        if out_indices is not None:
            kwargs["out_indices"] = out_indices

        encoder = Encoder(**kwargs)
        cls.encoder_channels = encoder.out_channels
        cls.encoder_reductions = encoder.out_reductions
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
            kwargs: Keyword arguments for the decoder. Take a look at the specific decoder docs for more info!
        """
        Decoder = get_decoder_class(name)
        components = build_components(kwargs)

        kwargs.update(components)
        out_channels = out_channels or cls.encoder_channels
        out_reductions = out_reductions or cls.encoder_reductions

        decoder = Decoder(out_channels, out_reductions, **kwargs)
        cls.decoder_channels = decoder.out_classes
        return decoder

    @classmethod
    def create_head(
        cls,
        task: str,
        classes: int,
        in_channels: Optional[int] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Build a head from a name and some keyword arguments.

        Args:
            name: The name of the head.
            in_channels: The number of channels of the input tensors of the forward pass.
            out_classes: The number of classes of the output tensors of the forward pass.
            kwargs: Keyword arguments for the head. Take a look at the specific head docs for more info!
        """
        Head = get_head(task)
        in_channels = in_channels or cls.decoder_channels

        head = Head(in_channels, classes, **kwargs)
        return head

    @classmethod
    def create_model(
        cls,
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
            encoder_cfg: Keyword arguments for the encoder. Check the specific encoder docs for more info.
            decoder_cfg: Keyword arguments for the decoder. Check the specific decoder docs for more info.
        """

        encoder_cfg = encoder_cfg or {}
        decoder_cfg = decoder_cfg or {}

        encoder = cls.create_encoder(encoder_name, **encoder_cfg)
        decoder = cls.create_decoder(decoder_name, **decoder_cfg)
        head = cls.create_head(task, classes)

        return MmitModel(encoder, decoder, head)


create_encoder = Factory.create_encoder
create_decoder = Factory.create_decoder
create_model = Factory.create_model
