"""Reading MQTT control packets out of a byte stream.

Every MQTT control packet starts with a fixed header (OASIS MQTT 3.1.1,
section 2.2; unchanged in 5.0):

    byte 1       bits 7-4 packet type, bits 3-0 flags
    bytes 2..    Remaining Length: bytes still to come in this packet,
                 as a Variable Byte Integer of 1 to 4 bytes

A Variable Byte Integer carries seven bits of value per byte, least
significant group first; bit 7 says whether another byte follows. Once the
Remaining Length is decoded the receiver knows where the packet ends. This is
MQTT's own framing on top of the TCP stream.
"""
from __future__ import annotations

from dataclasses import dataclass


__all__ = ["PACKET_NAMES", "Packet", "encode_remaining_length",
           "remaining_length", "split_packets", "publish_size"]

PACKET_NAMES = {
    1: "CONNECT", 2: "CONNACK", 3: "PUBLISH", 4: "PUBACK", 5: "PUBREC",
    6: "PUBREL", 7: "PUBCOMP", 8: "SUBSCRIBE", 9: "SUBACK", 10: "UNSUBSCRIBE",
    11: "UNSUBACK", 12: "PINGREQ", 13: "PINGRESP", 14: "DISCONNECT", 15: "AUTH",
}


def encode_remaining_length(x: int) -> bytes:
    """Encode x as a Variable Byte Integer (MQTT 3.1.1, section 2.2.3).

    Raises ValueError outside 0 .. 268435455, the largest value four bytes hold.
    """
    if not 0 <= x <= 268_435_455:
        raise ValueError(f"{x} è fuori da 0 .. 268 435 455")
    out = bytearray()
    while True:
        b, x = x % 128, x // 128                 # stacco i 7 bit meno significativi
        # se resta qualcosa accendo il bit 7: segue un altro byte
        out.append(b | 0x80 if x else b)
        if not x:
            return bytes(out)


def remaining_length(buf: bytes, i: int = 1) -> tuple[int, int]:
    """Decode the Variable Byte Integer that starts at buf[i].

    Returns (value, n), n being the bytes it takes (1 to 4). Raises
    ValueError if buf ends before the last byte, or if a fifth byte would be
    needed (malformed packet).
    """
    value, mult = 0, 1
    for n in range(1, 5):                        # al massimo quattro byte
        if i + n - 1 >= len(buf):
            raise ValueError("il buffer finisce dentro la Remaining Length")
        b = buf[i + n - 1]
        value += (b & 0x7F) * mult               # i 7 bit bassi valgono mult: 1, 128, 128², ...
        if not b & 0x80:                         # bit 7 spento: questo era l'ultimo byte
            return value, n
        mult *= 128
    raise ValueError("Remaining Length malformata: più di quattro byte")


@dataclass(frozen=True)
class Packet:
    """One MQTT control packet found in a stream."""
    type: int        # da 1 a 15
    flags: int       # i bit 3-0 del primo byte
    rl_bytes: int    # byte occupati dalla Remaining Length
    size: int        # il pacchetto intero, intestazione fissa compresa

    @property
    def name(self) -> str:
        """CONNECT, PUBLISH, PUBACK, ..."""
        return PACKET_NAMES.get(self.type, f"type {self.type}")

    @property
    def qos(self) -> int | None:
        """QoS of a PUBLISH (bits 2-1 of the flags), None for other packets."""
        return (self.flags >> 1) & 3 if self.type == 3 else None


def split_packets(stream: bytes) -> list[Packet]:
    """Cut a byte stream into MQTT control packets and return the list.

    An incomplete packet at the end of the stream is dropped.
    """
    out, i = [], 0
    while i < len(stream):
        try:
            # la Remaining Length parte dal secondo byte del pacchetto
            rl, n = remaining_length(stream, i + 1)
        except ValueError:
            break
        size = 1 + n + rl                        # primo byte, byte della lunghezza, resto del pacchetto
        if i + size > len(stream):               # pacchetto troncato in fondo: lo lascio fuori
            break
        # il tipo sta nei 4 bit alti del primo byte, i flag nei 4 bassi
        out.append(Packet(stream[i] >> 4, stream[i] & 0x0F, n, size))
        i += size                                # salto al pacchetto successivo
    return out


def publish_size(topic_len: int, payload_len: int, qos: int = 0) -> int:
    """Size in bytes of an MQTT 3.1.1 PUBLISH, computed from the specification.

    fixed header (1 + Remaining Length) + topic (2 + topic_len)
    + packet identifier (2, only if qos > 0) + payload.
    """
    # tutto quello che segue l'intestazione fissa: topic con i suoi 2 byte di lunghezza,
    # packet identifier solo con QoS 1 e 2, e il payload
    rest = 2 + topic_len + (2 if qos else 0) + payload_len
    return 1 + len(encode_remaining_length(rest)) + rest
