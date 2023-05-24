import pytest
import torch
from conftest import TEST_DECODERS, TEST_ENCODERS

import mmit


@pytest.mark.parametrize("encoder_name", TEST_ENCODERS)
@pytest.mark.parametrize("decoder_name", TEST_DECODERS)
def test_end2end(encoder_name, decoder_name):
    model = mmit.create_model(encoder_name, decoder_name, 3)

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out.shape[-2:] == x.shape[-2:]
    assert out.shape[1] == 3
