"""Resampling onto a uniform grid, and a zero-phase low-pass filter (Lab 2).

The raw data arrive on a non-uniform time grid (about 102.5 Hz on the phone,
104.3 Hz on the watch, with jitter). Windowing, convolution and FFT
downstream all assume uniform sampling, so the pipeline resamples first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iomtlab.config import SENSOR_CHANNELS, TARGET_HZ


def resample_to_hz(df: pd.DataFrame, target_hz: float = TARGET_HZ
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample one recording onto a uniform grid at target_hz.

    Sensor channels are interpolated linearly. Labels take the value of the
    last raw sample not after each grid time (previous-neighbour), since a
    categorical variable can't be interpolated. The grid starts at 0 s and
    never goes past the last raw sample.

    Returns (t, X, y): t (N,) in s from the first sample, X (N, 6) float32,
    y (N,) labels as strings.
    """
    t_raw = df["timestamp"].to_numpy(dtype=np.float64) / 1000.0   # i timestamp sono in ms: passo ai secondi
    t_raw = t_raw - t_raw[0]                       # porto l'origine dei tempi al primo campione
    t_new = np.arange(0.0, t_raw[-1], 1.0 / target_hz)   # griglia uniforme con passo 1/target_hz

    X_raw = df[SENSOR_CHANNELS].to_numpy(dtype=np.float64)
    # interpolo linearmente ogni canale sui nuovi istanti, uno alla volta
    X_new = np.stack(
        [np.interp(t_new, t_raw, X_raw[:, c]) for c in range(X_raw.shape[1])],
        axis=1,
    ).astype(np.float32)

    y_raw = df["label"].to_numpy()
    # per ogni istante nuovo prendo l'etichetta dell'ultimo campione grezzo che non lo supera
    idx = np.clip(np.searchsorted(t_raw, t_new, side="right") - 1, 0, len(y_raw) - 1)
    return t_new, X_new, y_raw[idx]


def lowpass_ma(X: np.ndarray, cutoff_hz: float, fs: float = TARGET_HZ) -> np.ndarray:
    """Moving-average low-pass applied forward and backward (zero phase); returns float32.

    X is (N, C) sampled at fs [Hz]; the average spans round(fs / cutoff_hz)
    samples. It separates gravity from the dynamic acceleration.
    """
    n = max(1, int(round(fs / cutoff_hz)))          # lunghezza della media mobile in campioni
    kernel = np.ones(n) / n
    out = np.empty_like(X, dtype=np.float32)
    for c in range(X.shape[1]):
        fwd = np.convolve(X[:, c], kernel, mode="same")                # filtro in avanti
        # poi filtro di nuovo il segnale ribaltato, così il ritardo della prima passata si annulla
        out[:, c] = np.convolve(fwd[::-1], kernel, mode="same")[::-1]
    return out


def on_grid(df: pd.DataFrame, grid_ms: np.ndarray) -> pd.DataFrame:
    """One recording evaluated at the times grid_ms (UNIX ms).

    Channels are interpolated linearly; the label is the one of the last raw
    sample not after each grid time. Returns a DataFrame with timestamp, the
    six channels and label. Raises ValueError if the grid is empty or goes
    outside the data: we don't extrapolate.
    """
    t = df["timestamp"].to_numpy(dtype=np.float64)
    g = np.asarray(grid_ms, dtype=np.float64)
    # se la griglia esce dall'intervallo dei dati mi fermo: fuori dai dati non ho valori
    if len(g) == 0 or g[0] < t[0] or g[-1] > t[-1]:
        raise ValueError("la griglia esce dall'intervallo dei dati: niente estrapolazione")
    out = {"timestamp": g}
    for c in SENSOR_CHANNELS:                # interpolo linearmente, un canale alla volta
        out[c] = np.interp(g, t, df[c].to_numpy(dtype=np.float64))
    # l'etichetta è quella dell'ultimo campione non successivo all'istante della griglia
    idx = np.clip(np.searchsorted(t, g, side="right") - 1, 0, len(t) - 1)
    out["label"] = df["label"].to_numpy()[idx]
    return pd.DataFrame(out)
