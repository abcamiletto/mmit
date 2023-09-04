import pytest
import torch
from conftest import DECODERS, TEST_ENCODERS

import mmit


@pytest.mark.parametrize("encoder_name", TEST_ENCODERS)
@pytest.mark.parametrize("decoder_name", DECODERS)
def test_timm_encoder_decoder(encoder_name, decoder_name):
    """Test that the timm encoder and decoder work together."""
    encoder = mmit.create_encoder(encoder_name, pretrained=False)
    decoder = mmit.create_decoder(decoder_name)

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        features = encoder(x)
        out = decoder(*features)

    assert out.shape[-2:] == x.shape[-2:]
    assert out.shape[1] == decoder.out_classes


@pytest.mark.parametrize("encoder_name", TEST_ENCODERS)
@pytest.mark.parametrize("decoder_name", DECODERS)
@pytest.mark.parametrize("input_shape", [(277, 289), (271, 333)])
def test_timm_encoder_decoder_awful_shape(encoder_name, decoder_name, input_shape):
    """Test that the timm encoder and decoder work together."""
    encoder = mmit.create_encoder(encoder_name, pretrained=False)
    decoder = mmit.create_decoder(decoder_name)

    x = torch.randn(2, 3, *input_shape)
    with torch.no_grad():
        features = encoder(x)
        out = decoder(*features)

    assert out.shape[-2:] == x.shape[-2:]
    assert out.shape[1] == decoder.out_classes


@pytest.mark.parametrize("encoder_name", TEST_ENCODERS)
@pytest.mark.parametrize("decoder_name", DECODERS)
def test_timm_encoder_layers_stride_decoder(encoder_name, decoder_name):
    """Test that the timm encoder and decoder work together."""
    encoder = mmit.create_encoder(
        encoder_name, pretrained=False, out_indices=(0, 3, 4), output_stride=8
    )
    decoder = mmit.create_decoder(decoder_name)

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        features = encoder(x)
        out = decoder(*features)

    assert out.shape[-2:] == x.shape[-2:]
    assert out.shape[1] == decoder.out_classes


@pytest.mark.parametrize("encoder_name", TEST_ENCODERS)
@pytest.mark.parametrize("decoder_name", DECODERS)
def test_return_features(encoder_name, decoder_name):
    """Test that the timm encoder and decoder work together."""
    encoder = mmit.create_encoder(encoder_name, pretrained=False)
    decoder = mmit.create_decoder(decoder_name, return_features=True)

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        features = encoder(x)
        features = decoder(*features)

    assert isinstance(features, list)
    for item in features:
        assert isinstance(item, torch.Tensor)
        assert item.ndim == 4
