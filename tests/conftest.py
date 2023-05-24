from demo_blocks import ClassicDecoder, ClassicNet, CursedNet

import mmit

DECODERS = mmit.list_decoders()
ENCODERS = mmit.list_encoders()

mmit.register_encoder(CursedNet)
mmit.register_encoder(ClassicNet)
mmit.register_decoder(ClassicDecoder)

TEST_ENCODERS = ["classicnet", "cursednet"]
TEST_DECODERS = ["classicdecoder"]
