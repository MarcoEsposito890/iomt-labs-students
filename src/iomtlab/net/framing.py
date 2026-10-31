"""Length-prefixed framing on top of TCP.

TCP is a byte stream: two send() of 11 bytes can arrive as one recv() of 22,
or as 3 + 19. The receiver cannot tell where a message ends unless the
sender marks it. Here each payload is preceded by a 2-byte length
(big-endian, as in MQTT), so a payload is at most 65535 bytes.

    +---------------------+---------------------------+
    | length, uint16, BE  | payload, `length` bytes   |
    +---------------------+---------------------------+
"""
from __future__ import annotations

import struct


__all__ = ["HEADER", "MAX_PAYLOAD", "frame", "read_frame"]

#: The length prefix: unsigned 16-bit integer, big-endian.
HEADER = struct.Struct("!H")
MAX_PAYLOAD = 2 ** (8 * HEADER.size) - 1


def frame(payload: bytes) -> bytes:
    """Return payload preceded by its length as a 2-byte big-endian integer.

    Raises ValueError if the payload is longer than MAX_PAYLOAD bytes.
    """
    if len(payload) > MAX_PAYLOAD:           # con 2 byte di lunghezza non posso annunciare più di 65535 byte
        raise ValueError(f"payload di {len(payload)} byte: con 2 byte di lunghezza "
                         f"il massimo è {MAX_PAYLOAD}")
    return HEADER.pack(len(payload)) + bytes(payload)    # scrivo la lunghezza e poi il payload


def read_frame(sock) -> bytes | None:
    """Read one frame from sock and return its payload.

    Returns None if the peer closed between two frames; raises
    ConnectionError if it closed in the middle of one. sock only needs
    recv(n), which may return fewer than n bytes.
    """
    def exactly(n: int, at_boundary: bool) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:                          # continuo a leggere finché non ho n byte
            chunk = sock.recv(n - len(buf))          # recv può restituire meno di quanto chiedo
            if not chunk:                            # b"" vuol dire che l'altro lato ha chiuso
                # se ha chiuso fra due frame è una chiusura pulita, altrimenti è un errore
                if at_boundary and not buf:
                    return None
                raise ConnectionError(f"flusso chiuso dopo {len(buf)} byte "
                                      f"dei {n} attesi")
            buf += chunk
        return bytes(buf)

    head = exactly(HEADER.size, at_boundary=True)    # prima leggo i 2 byte di lunghezza
    if head is None:
        return None
    (n,) = HEADER.unpack(head)                       # li decodifico come intero big-endian
    return exactly(n, at_boundary=False) if n else b""   # poi leggo esattamente n byte di payload
