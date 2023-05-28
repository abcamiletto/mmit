import pytest
import torch

import mmit


@pytest.mark.parametrize("activ_name", ["relu", "leakyrelu", "elu", "selu", "none"])
def test_activ_layers(activ_name):
    """Test that the timm decoder layers work."""
    placeholder_encoder = mmit.create_encoder("classicnet")
    decoder = mmit.create_decoder("unet", activation_layer=activ_name)

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        features = placeholder_encoder(x)
        out = decoder(*features)

    assert out.shape[-2:] == x.shape[-2:]
    assert out.shape[1] == decoder.out_classes


@pytest.mark.parametrize("norm_name", ["batch", "instance", "none"])
def test_norm_layers(norm_name):
    """Test that the timm decoder layers work."""
    placeholder_encoder = mmit.create_encoder("classicnet")
    decoder = mmit.create_decoder("unet", norm_layer=norm_name)

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        features = placeholder_encoder(x)
        out = decoder(*features)

    assert out.shape[-2:] == x.shape[-2:]
    assert out.shape[1] == decoder.out_classes


@pytest.mark.parametrize("extra_layer", ["none"])
def test_extra_layers(extra_layer):
    """Test that the timm decoder layers work."""
    placeholder_encoder = mmit.create_encoder("classicnet")
    decoder = mmit.create_decoder("unet", extra_layer=extra_layer)

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        features = placeholder_encoder(x)
        out = decoder(*features)

    assert out.shape[-2:] == x.shape[-2:]
    assert out.shape[1] == decoder.out_classes


@pytest.mark.parametrize("mismatch_layer", ["pad", "interpolate"])
def test_mismatch_layers(mismatch_layer):
    """Test that the timm decoder layers work."""
    placeholder_encoder = mmit.create_encoder("classicnet")
    decoder = mmit.create_decoder("unet", mismatch_layer=mismatch_layer)

    x = torch.randn(2, 3, 179, 111)
    with torch.no_grad():
        features = placeholder_encoder(x)
        out = decoder(*features)

    assert out.shape[-2:] == x.shape[-2:]
    assert out.shape[1] == decoder.out_classes


@pytest.mark.parametrize("upsample_layer", ["convtransposed", "interpolate"])
def test_upsample_layers(upsample_layer):
    """Test that the timm decoder layers work."""
    placeholder_encoder = mmit.create_encoder("classicnet")
    decoder = mmit.create_decoder("unet", upsample_layer=upsample_layer)

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        features = placeholder_encoder(x)
        out = decoder(*features)

    assert out.shape[-2:] == x.shape[-2:]
    assert out.shape[1] == decoder.out_classes
