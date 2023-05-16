import pytest
import torch
from conftest import DECODERS, TIMM_ENCODERS

import mmit


@pytest.mark.parametrize("encoder_name", TIMM_ENCODERS)
@pytest.mark.parametrize("decoder_name", DECODERS)
def test_timm_encoder_decoder(encoder_name, decoder_name):
    """Test that the timm encoder and decoder work together."""
    encoder = mmit.encoders.TimmEncoder(encoder_name, pretrained=False)
    out_ch, out_red = encoder.out_channels, encoder.out_reductions

    decoder = mmit.decoders.create_decoder(decoder_name, out_ch, out_red)

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        features = encoder(x)
        for f in features:
            print(f.shape)
        out = decoder(*features)

    assert out.shape[-2:] == x.shape[-2:]


@pytest.mark.parametrize("encoder_name", TIMM_ENCODERS)
@pytest.mark.parametrize("decoder_name", DECODERS)
@pytest.mark.parametrize("input_shape", [(151, 210), (87, 141)])
def test_timm_encoder_decoder_awful_shape(encoder_name, decoder_name, input_shape):
    """Test that the timm encoder and decoder work together."""
    encoder = mmit.encoders.TimmEncoder(encoder_name, pretrained=False)
    out_ch, out_red = encoder.out_channels, encoder.out_reductions

    decoder = mmit.decoders.create_decoder(decoder_name, out_ch, out_red)

    x = torch.randn(2, 3, *input_shape)
    with torch.no_grad():
        features = encoder(x)
        for f in features:
            print(f.shape)
        out = decoder(*features)

    assert out.shape[-2:] == x.shape[-2:]


@pytest.mark.parametrize("encoder_name", TIMM_ENCODERS)
@pytest.mark.parametrize("decoder_name", DECODERS)
def test_timm_encoder_layers_stride_decoder(encoder_name, decoder_name):
    """Test that the timm encoder and decoder work together."""
    encoder = mmit.encoders.TimmEncoder(
        encoder_name, pretrained=False, layers=(0, 3, 4), output_stride=8
    )
    out_ch, out_red = encoder.out_channels, encoder.out_reductions

    decoder = mmit.decoders.create_decoder(decoder_name, out_ch, out_red)

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        features = encoder(x)
        for f in features:
            print(f.shape)
        out = decoder(*features)

    assert out.shape[-2:] == x.shape[-2:]
