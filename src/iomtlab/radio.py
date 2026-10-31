"""IEEE 802.15.4 frames: one convention for every lab, the same as the theory.

A frame carries at most 127 bytes (PSDU). With 11 bytes of MAC header and
footer, at most 116 bytes are useful; with the 6 bytes of synchronisation
header and PHY header on the air, every frame costs 17 bytes on top of its
payload. 250 kbit/s, 240 nJ per bit (iomtlab.config.ENERGY).
"""
from __future__ import annotations

import math

from iomtlab.config import ENERGY

__all__ = ["PSDU_B", "PAYLOAD_B", "OVERHEAD_B", "PHY_B", "RATE_BPS", "frames", "air_bytes",
           "air_time_s", "air_energy_J"]

PSDU_B = 127          #: byte di una trama 802.15.4
PAYLOAD_B = 116       #: byte utili per trama, tolte le intestazioni MAC
OVERHEAD_B = 17       #: byte in più per trama sull'aria: MAC (11) e fisico (6)
PHY_B = 6             #: preambolo, delimitatore e intestazione fisica, fuori dai 127
RATE_BPS = ENERGY["rate_lowpower_bps"]


def frames(nbytes: int, payload_B: int = PAYLOAD_B) -> int:
    """Frames needed for nbytes of payload, payload_B useful bytes each; 0 for nothing."""
    return int(math.ceil(nbytes / payload_B)) if nbytes > 0 else 0


def air_bytes(nbytes: int) -> int:
    """Bytes on the air for nbytes of payload: the payload plus OVERHEAD_B per frame."""
    return int(nbytes + OVERHEAD_B * frames(nbytes))


def air_time_s(nbytes: int, rate_bps: float = RATE_BPS) -> float:
    """Seconds of radio for nbytes of payload, frames included, without acknowledgements."""
    return air_bytes(nbytes) * 8 / rate_bps


def air_energy_J(nbytes: int, e_bit_J: float = ENERGY["e_bit_lowpower_J"]) -> float:
    """Energy of the radio for nbytes of payload, frames included."""
    return air_bytes(nbytes) * 8 * e_bit_J
