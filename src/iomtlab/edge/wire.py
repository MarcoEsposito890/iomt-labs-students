"""Serialising a tensor for the link as int8, with scale and zero point.

If the scale is lost on the way, the model downstream still predicts
something, and the results are wrong without any error. The round trip
unpack_int8(pack_int8(t)) gives t back within one quantisation step.

Format, little-endian, 10 + 4 * ndim bytes of header followed by the data:

    magic   4 bytes  b'IOMT'
    ndim    1 byte
    shape   ndim * 4 bytes, uint32
    scale   4 bytes, float32
    zero    1 byte, int8
    data    prod(shape) bytes, int8
"""
from __future__ import annotations

import struct

import numpy as np

__all__ = ["pack_int8", "unpack_int8"]

_MAGIC = b"IOMT"


def pack_int8(t) -> bytes:
    """Quantise t to int8 (asymmetric, one scale per tensor) and return the bytes.

    t is a torch tensor or anything numpy can convert. A constant tensor gets
    scale 1, zero point 0 and all codes 0.
    """
    # porto il tensore in numpy, in float32
    a = np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t,
                   dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi == lo:
        # tensore costante: non ho un intervallo da dividere, mando tutti zeri
        scale, zero = 1.0, 0
        q = np.zeros(a.shape, dtype=np.int8)
    else:
        scale = (hi - lo) / 255.0            # divido l'intervallo [lo, hi] in 255 passi
        # lo zero point è il codice tale che lo vada in -128
        zero = int(np.clip(round(-128 - lo / scale), -128, 127))
        q = np.clip(np.round(a / scale) + zero, -128, 127).astype(np.int8)
    # intestazione: magic, numero di dimensioni, dimensioni, scala e zero point
    head = _MAGIC + struct.pack("<B", a.ndim)
    head += b"".join(struct.pack("<I", d) for d in a.shape)
    head += struct.pack("<fb", scale, zero)
    return head + q.tobytes()


def unpack_int8(b: bytes) -> np.ndarray:
    """Inverse of pack_int8: return the float32 numpy array.

    Raises ValueError if the magic is missing or the data are truncated.
    """
    if b[:4] != _MAGIC:
        raise ValueError("manca l'intestazione IOMT: non è un payload di pack_int8")
    i = 4                                    # i è la posizione a cui leggo nel buffer
    ndim = struct.unpack_from("<B", b, i)[0]
    i += 1
    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack_from("<I", b, i)[0])
        i += 4
    scale, zero = struct.unpack_from("<fb", b, i)   # 4 byte di scala e 1 di zero point
    i += 5
    n = int(np.prod(shape)) if shape else 1         # numero di valori attesi
    q = np.frombuffer(b, dtype=np.int8, count=n, offset=i)
    if len(q) != n:
        raise ValueError(f"payload troncato: {len(q)} valori su {n} attesi")
    # tolgo lo zero point e moltiplico per la scala per tornare ai valori reali
    return ((q.astype(np.float32) - zero) * scale).reshape(shape)
