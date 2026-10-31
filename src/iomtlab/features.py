"""Hand-crafted features for the classical baseline (Labs 1, 2, 3).

Each feature is a physical quantity with a unit. They come in blocks, so a
lab can measure what each block adds. Every block takes windows X of shape
(W, T, C) and returns (features (W, F), names).

The spectral block is kept for the theory; the labs use time-domain
features only.
"""

from __future__ import annotations

import numpy as np

from iomtlab.config import TARGET_HZ

#: Bands in Hz, wide compared with the 1 Hz resolution of a 1 s window.
BANDS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 25.0))

#: Axis pairs for the correlation: accelerometer first, then gyroscope.
AXIS_PAIRS = ((0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5))


def _channel_names(n_ch: int) -> list[str]:
    """Short names of the n_ch channels of the active dataset (ax ... gz for TUG); c0, c1, ... if they differ."""
    from iomtlab.spec import active
    ch = active().short_names()
    return ch[:n_ch] if len(ch) >= n_ch else [f"c{i}" for i in range(n_ch)]


def _names(prefix: str, n_ch: int) -> list[str]:
    """Feature names prefix_ax, prefix_ay, ... for the first n_ch channels."""
    return [f"{prefix}_{c}" for c in _channel_names(n_ch)]


def amplitude(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Level and orientation: mean, min, max, RMS per channel."""
    # le quattro statistiche le calcolo lungo il tempo, che è l'asse 1, canale per canale
    f = [X.mean(1), X.min(1), X.max(1), np.sqrt((X ** 2).mean(1))]
    names = sum([_names(p, X.shape[2]) for p in ("mean", "min", "max", "rms")], [])
    return np.concatenate(f, 1), names


def dispersion(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """How much the signal moves: standard deviation (ddof=0) and interquartile range."""
    iqr = np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1)   # distanza fra terzo e primo quartile
    names = sum([_names(p, X.shape[2]) for p in ("std", "iqr")], [])
    return np.concatenate([X.std(1), iqr], 1), names


def dynamics(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """How fast the signal changes: mean jerk and zero-crossing rate.

    Jerk is the mean absolute difference between consecutive samples (per
    sample, not per second); zero crossings are counted around the mean.
    """
    # la media, in modulo, delle differenze fra campioni consecutivi
    jerk = np.abs(np.diff(X, axis=1)).mean(1)
    c = X - X.mean(1, keepdims=True)                   # tolgo la media della finestra
    zcr = (np.diff(np.sign(c), axis=1) != 0).mean(1)   # frazione di passi in cui il segno cambia
    names = sum([_names(p, X.shape[2]) for p in ("jerk", "zcr")], [])
    return np.concatenate([jerk, zcr], 1), names


def spectral(X: np.ndarray, fs: float | None = None,
             bands=BANDS) -> tuple[np.ndarray, list[str]]:
    """Periodicity and energy distribution: dominant frequency (Hz), spectral entropy, band energies.

    The mean is removed and a Hann window applied before the transform:
    without them the DC component fills the first bin and the rectangular
    cut leaks energy into the nearby bins.
    """
    if fs is None:
        from iomtlab.spec import active
        fs = active().fs_hz
    T = X.shape[1]
    w = np.hanning(T)[None, :, None]         # finestra di Hann lungo il tempo
    # tolgo la media, finestro e prendo il modulo al quadrato della rfft
    S = np.abs(np.fft.rfft((X - X.mean(1, keepdims=True)) * w, axis=1)) ** 2
    f = np.fft.rfftfreq(T, 1.0 / fs)         # frequenza di ogni bin in Hz
    P = S / (S.sum(1, keepdims=True) + 1e-12)   # normalizzo lo spettro a somma 1, come una distribuzione
    # frequenza del bin più alto ed entropia spettrale
    out = [f[S.argmax(1)], -(P * np.log(P + 1e-12)).sum(1)]
    names = sum([_names(p, X.shape[2]) for p in ("fdom", "sent")], [])
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        out.append(S[:, m, :].sum(1))        # energia nella banda [lo, hi)
        names += _names(f"b{lo:g}_{hi:g}", X.shape[2])
    return np.concatenate(out, 1), names


def cross_axis(X: np.ndarray, pairs=None) -> tuple[np.ndarray, list[str]]:
    """Direction of motion: Pearson correlation between the pairs of axes in pairs.

    pairs defaults to the pairs inside each triad of the active dataset
    (AXIS_PAIRS for TUG); a dataset without triads gets no columns.
    """
    from iomtlab.spec import active
    if pairs is None:
        pairs = [(t[i], t[j]) for t in active().triads for i, j in ((0, 1), (0, 2), (1, 2))]
    lab = _channel_names(X.shape[2])
    if not pairs:
        return np.zeros((len(X), 0)), []
    cols, names = [], []
    for a, b in pairs:
        xa, xb = X[:, :, a], X[:, :, b]
        # covarianza fra i due assi dentro la finestra
        num = ((xa - xa.mean(1, keepdims=True))
               * (xb - xb.mean(1, keepdims=True))).mean(1)
        # divido per il prodotto delle deviazioni standard: è la correlazione di Pearson
        cols.append((num / (xa.std(1) * xb.std(1) + 1e-12))[:, None])
        names.append(f"corr_{lab[a]}{lab[b]}")
    return np.concatenate(cols, 1), names


def norm(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Mean and std of the norm of each three-axis sensor (triads of the active dataset).

    They do not depend on the orientation of the device (block 1.1, section
    4), which matters because the phone sits in a pocket with no fixed
    orientation. The accelerometer norm still contains gravity. For TUG the
    triads are the accelerometer (a) and the gyroscope (w): mean_anorm,
    std_anorm, mean_wnorm, std_wnorm. A dataset without triads gets no columns.
    """
    from iomtlab.spec import active
    S = active()
    f, names = [], []
    for t, nm in zip(S.triads, S.triad_names):
        v = np.linalg.norm(X[:, :, list(t)], axis=2)   # la norma della terna, campione per campione
        f += [v.mean(1), v.std(1)]
        names += [f"mean_{nm}norm", f"std_{nm}norm"]
    if not f:
        return np.zeros((len(X), 0)), []
    # per TUG l'ordine resta quello di prima: media e std di a, poi di w
    return np.stack(f, 1), names


def sma(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Signal magnitude area of each triad: mean over the window of |x| + |y| + |z|.

    One feature per triad of the active dataset (sma_a, sma_w for TUG). Not one of
    the default blocks: it is the reference solution of the Lab 3 «Scrivi tu»
    exercise, registered by the notebook as BLOCKS["sma"].
    """
    from iomtlab.spec import active
    sp = active()
    f, names = [], []
    for tri, nm in zip(sp.triads, sp.triad_names):
        f.append(np.abs(X[:, :, list(tri)]).sum(2).mean(1))   # |x|+|y|+|z| a ogni campione, poi la media
        names.append(f"sma_{nm}")
    if not f:
        return np.zeros((len(X), 0)), []
    return np.stack(f, 1), names


#: The blocks, in order of how much domain knowledge they encode.
BLOCKS = {
    "amplitude": amplitude,
    "dispersion": dispersion,
    "dynamics": dynamics,
    "cross_axis": cross_axis,
    "norm": norm,
    "spectral": spectral,
}


def extract(X: np.ndarray, blocks=("amplitude", "dispersion"),
            fs: float | None = None) -> tuple[np.ndarray, list[str]]:
    """Extract the requested blocks from windows X (W, T, C).

    Returns (features (W, F), names), NaN replaced by 0. The default is the
    amplitude and dispersion blocks, 36 features on six channels. Raises
    ValueError on an unknown block name.
    """
    cols, names = [], []
    for b in blocks:
        if b not in BLOCKS:
            raise ValueError(f"blocco sconosciuto: {b}; disponibili {list(BLOCKS)}")
        fn = BLOCKS[b]
        v, nm = fn(X, fs) if b == "spectral" else fn(X)   # solo il blocco spettrale ha bisogno di fs
        cols.append(v)
        names += nm
    # affianco i blocchi colonna per colonna e sostituisco eventuali NaN con zero
    return np.nan_to_num(np.concatenate(cols, 1)), names
