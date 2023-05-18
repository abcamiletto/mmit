from cursed_net import CursedNet

import mmit

mmit.register_encoder(CursedNet)

TIMM_ENCODERS = ["resnet18", "cursednet"]
DECODERS = mmit.list_decoders()
