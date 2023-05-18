from demo_encoders import ClassicNet, CursedNet

import mmit

mmit.register_encoder(CursedNet)
mmit.register_encoder(ClassicNet)

TEST_ENCODERS = ["classicnet", "cursednet"]
DECODERS = mmit.list_decoders()
