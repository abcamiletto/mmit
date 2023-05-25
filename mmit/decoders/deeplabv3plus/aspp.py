from functools import partial

from ..deeplabv3.aspp import ASPP
from .parts import ASPPSeparableConv

ASPP = partial(ASPP, asppconv_layer=ASPPSeparableConv)
