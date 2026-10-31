"""Windowing, labels and normalisation.

A window must hold one gait cycle and be short enough not to merge two
activities. At 50 Hz, 50 samples are 1.0 s.
"""

from __future__ import annotations

import numpy as np

from iomtlab.config import FULL_SCALE, LABEL_MAP, WINDOW_SHIFT, WINDOW_SIZE


def normalize_full_scale(X: np.ndarray, device: str) -> np.ndarray:
    """Scale X (N, 6) to [-1, 1] by the declared full scale of the sensor.

    We use the full scale instead of statistics of the data: it is known on
    the device, so the same transform applies in deployment and leaks nothing
    from the dataset. Returns float32. Raises ValueError for an unknown device.
    """
    if device not in FULL_SCALE:
        raise ValueError(f"dispositivo sconosciuto: {device}")
    acc = FULL_SCALE[device]["acc"]
    gyro = FULL_SCALE[device]["gyro"]
    out = np.asarray(X, dtype=np.float32).copy()
    out[:, 0:3] = np.clip(out[:, 0:3] / acc, -1.0, 1.0)      # accelerometro diviso per il suo fondo scala
    out[:, 3:6] = np.clip(out[:, 3:6] / gyro, -1.0, 1.0)     # giroscopio diviso per il suo
    return out


def encode_labels(y: np.ndarray) -> np.ndarray:
    """Text labels to int64 class indices, following config.LABEL_MAP; KeyError on an unknown label."""
    return np.array([LABEL_MAP[str(v)] for v in y], dtype=np.int64)


def make_windows(X: np.ndarray, y: np.ndarray,
                 size: int = WINDOW_SIZE, shift: int = WINDOW_SHIFT
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut X (N, C) and integer labels y (N,) into overlapping windows.

    Returns (Xw, yw, purity):
      Xw     (W, size, C) float32;
      yw     (W,) majority label; on a tie the smaller label wins;
      purity (W,) share of samples in the window that carry yw. purity < 1
             marks windows across a transition.
    With fewer than size samples the three arrays are empty.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    if n < size:
        return (np.empty((0, size, X.shape[1]), np.float32),
                np.empty((0,), np.int64),
                np.empty((0,), np.float32))

    n_win = (n - size) // shift + 1                # quante finestre intere ci stanno con questo passo
    Xw = np.empty((n_win, size, X.shape[1]), dtype=np.float32)
    yw = np.empty(n_win, dtype=np.int64)
    purity = np.empty(n_win, dtype=np.float32)

    for i in range(n_win):
        s = i * shift                              # inizio della finestra i
        seg_y = y[s:s + size]
        counts = np.bincount(seg_y)                # conto i campioni di ogni classe nella finestra
        Xw[i] = X[s:s + size]
        # argmax prende il primo massimo: a parità vince l'indice minore
        yw[i] = int(counts.argmax())
        # frazione di campioni che hanno l'etichetta scelta
        purity[i] = counts.max() / size

    return Xw, yw, purity


def label_window(y_seg, min_purity: float = 0.8) -> tuple[int, bool]:
    """Majority label of one window, and whether the window is pure.

    Returns (label, purity >= min_purity). A tie goes to the smaller label,
    so the result does not depend on the order of the samples.
    """
    vals, counts = np.unique(np.asarray(y_seg), return_counts=True)   # np.unique restituisce i valori ordinati
    k = int(np.argmax(counts))               # argmax prende il primo massimo, cioè l'etichetta minore
    # la finestra è pura se l'etichetta di maggioranza copre almeno min_purity dei campioni
    return int(vals[k]), bool(counts[k] / counts.sum() >= min_purity)
