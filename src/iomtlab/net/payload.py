"""A binary format for one window of the course pipeline.

MQTT and TCP carry bytes and define nothing about them, so the two ends agree
on a format beforehand. This is the one used in the lab:

    offset  size  field   type     meaning
    0       4     seq     uint32   window counter, from 0
    4       4     t0_ms   uint32   time of the first sample, ms from the start
    8       2     n       uint16   samples per channel
    10      1     ch      uint8    channels
    11      2*n*ch        int16    samples, sample-major: s0c0 s0c1 .. s0c5 s1c0 ..

All fields are big-endian. The samples are the six channels of the pipeline
(3 accelerometer, 3 gyroscope) quantised to int16 against the full scale of
the device (config.FULL_SCALE):

    code = round(value / lsb),   lsb = full_scale / 32767

Values beyond the full scale saturate at +-32767, as the sensor itself does.
A wrapping int16 cast would turn +9 g into a large negative value. The full
scale is not sent: both ends know it, like the format.
"""
from __future__ import annotations

import json
import struct

import numpy as np

from iomtlab.config import FULL_SCALE, N_CHANNELS

__all__ = ["WINDOW_HEADER", "lsb", "pack_window", "unpack_window",
           "encode_json", "encode_float32", "synthetic_window",
           "DECISION", "pack_decision", "unpack_decision"]

#: The 11-byte header: seq, t0_ms, n, ch.
WINDOW_HEADER = struct.Struct("!IIHB")
INT16_MAX = 32767


def lsb(device: str = "sp") -> np.ndarray:
    """Quantisation step of each of the six channels, in m/s^2 and rad/s."""
    fs = FULL_SCALE[device]
    # tre canali di accelerometro e tre di giroscopio, ognuno con il suo fondo scala diviso 32767
    return np.array([fs["acc"]] * 3 + [fs["gyro"]] * 3) / INT16_MAX


def pack_window(seq: int, t0_ms: int, X: np.ndarray, device: str = "sp") -> bytes:
    """Encode one window X of shape (n, 6) in the format of this module.

    Each channel is quantised with its own lsb and saturated at +-32767.
    Returns the 11-byte header followed by the samples, all big-endian.
    Raises ValueError if X is not (n, 6).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != N_CHANNELS:
        raise ValueError(f"attesa una finestra (n, {N_CHANNELS}), arrivata {X.shape}")
    # divido per il passo di quantizzazione, arrotondo e saturo a ±32767 prima di passare a int16
    q = np.clip(np.round(X / lsb(device)), -INT16_MAX, INT16_MAX).astype(">i2")
    # l'intestazione: contatore, istante iniziale in ms, numero di campioni e di canali
    head = WINDOW_HEADER.pack(int(seq), int(t0_ms), X.shape[0], X.shape[1])
    # ordine C: prima i sei canali del campione 0, poi quelli del campione 1, e così via
    return head + q.tobytes(order="C")


def unpack_window(b: bytes, device: str = "sp") -> tuple[int, int, np.ndarray]:
    """Inverse of pack_window: return (seq, t0_ms, X), X in physical units.

    Raises ValueError if the body length does not match the header.
    """
    seq, t0, n, ch = WINDOW_HEADER.unpack_from(b, 0)     # leggo gli 11 byte di intestazione
    body = b[WINDOW_HEADER.size:]
    # controllo che il corpo abbia proprio 2 byte per ogni campione annunciato
    if len(body) != 2 * n * ch:
        raise ValueError(f"l'intestazione annuncia {n}x{ch} campioni ({2 * n * ch} byte), "
                         f"il corpo ne ha {len(body)}")
    q = np.frombuffer(body, dtype=">i2").reshape(n, ch)  # rileggo gli int16 come matrice n x ch
    # e torno alle unità fisiche moltiplicando per il passo di quantizzazione
    return seq, t0, q.astype(np.float64) * lsb(device)[:ch]


def encode_json(seq: int, t0_ms: int, X: np.ndarray, decimals: int | None = None) -> bytes:
    """The same window as compact JSON bytes: {"seq":..,"t0":..,"x":[[..6..],..]}.

    With decimals=None each float is written with all the digits of Python's
    shortest round-trip repr, the json.dumps default.
    """
    X = np.asarray(X, dtype=np.float64)
    # se chiedo dei decimali arrotondo prima, altrimenti json scrive tutte le cifre del numero
    rows = X.tolist() if decimals is None else np.round(X, decimals).tolist()
    # separatori senza spazi: è il JSON più compatto che posso mandare
    return json.dumps({"seq": int(seq), "t0": int(t0_ms), "x": rows},
                      separators=(",", ":")).encode()


def encode_float32(X: np.ndarray) -> bytes:
    """The samples alone as big-endian float32, no header: 4 bytes per value."""
    return np.asarray(X, dtype=">f4").tobytes(order="C")


def synthetic_window(n: int = 50, fs: float = 50.0, seed: int = 0) -> tuple[int, np.ndarray]:
    """One synthetic window of the course pipeline: (t0_ms, X) with X of shape (n, 6).

    Lab 1 measures bytes and does not need the dataset. The values have the
    magnitudes of a phone in a pocket during walking: gravity on z plus a step
    component on the accelerometer [m/s^2], a few tenths of rad/s on the
    gyroscope, and noise. Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs                     # istanti dei campioni, in s
    w = 2 * np.pi * 1.8                       # pulsazione del passo: 1.8 passi al secondo
    # accelerometro: due componenti al ritmo del passo, e su z la gravità più un'oscillazione a 2w
    acc = np.c_[0.8 * np.sin(w * t), 1.2 * np.sin(w * t + 0.6),
                9.81 + 2.0 * np.sin(2 * w * t)]
    # giroscopio: qualche decimo di rad/s
    gyr = np.c_[0.4 * np.sin(w * t + 1.0), 0.2 * np.sin(w * t + 2.0),
                0.6 * np.sin(0.5 * w * t)]
    # aggiungo rumore gaussiano, più alto sull'accelerometro
    X = np.c_[acc + rng.normal(0, 0.05, acc.shape), gyr + rng.normal(0, 0.01, gyr.shape)]
    return 2000, X                            # t0 = 2000 ms: un istante qualsiasi, fisso


#: One decision of the node: t_ms uint32, class uint8, confidence in % uint8.
DECISION = struct.Struct("!IBB")


def pack_decision(t_ms: int, label: int, confidence: float) -> bytes:
    """One decision in 6 bytes, big-endian: time in ms (uint32, modulo 2**32),
    class (uint8) and confidence rounded to whole percent (uint8, 0-100).

    The reference solution of the Lab 1 «Scrivi tu» exercise.
    """
    pct = int(round(100 * min(max(float(confidence), 0.0), 1.0)))   # confidenza in percento, fra 0 e 100
    return DECISION.pack(int(t_ms) % 2**32, int(label), pct)


def unpack_decision(b: bytes) -> tuple[int, int, float]:
    """Inverse of pack_decision: (t_ms, label, confidence in [0, 1])."""
    t, c, pct = DECISION.unpack(b)
    return t, c, pct / 100
