"""Frequency analysis, for the theory of block 1.1 (the labs use time-domain features).

Three quantities decide whether a spectrum can be read:

  resolution   Delta_f = fs / N = 1 / T_obs. It depends only on the observed
               duration; zero padding does not change it.
  leakage      a sinusoid that does not fall on a bin spreads over the
               neighbouring bins. A tapering window lowers the side lobes and
               widens the main lobe.
  cost         a radix-2 FFT takes (N/2) log2(N) butterflies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iomtlab.config import TARGET_HZ


# ------------------------------------------------------------------ windows
WINDOWS = {
    "rect": np.ones,
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
}


@dataclass
class WindowProperties:
    """Properties of a tapering window.

    coherent_gain       gain on a sinusoid centred on a bin; divide by it to
                        read an absolute amplitude.
    enbw                equivalent noise bandwidth, in bins: width of the
                        rectangular window that passes the same white-noise
                        power.
    scalloping_loss_db  largest loss for a sinusoid halfway between two bins, dB.
    """

    name: str
    coherent_gain: float
    enbw: float
    scalloping_loss_db: float


def window_properties(name: str, n: int = 1024) -> WindowProperties:
    """Coherent gain, ENBW (bins) and scalloping loss (dB) of WINDOWS[name] on n samples."""
    w = WINDOWS[name](n)
    cg = w.sum() / n                               # il guadagno coerente è la media dei pesi
    enbw = n * (w ** 2).sum() / (w.sum() ** 2)     # banda equivalente di rumore, in bin
    # per lo scalloping calcolo la risposta a metà bin, f = 0.5 * Delta_f,
    # e la confronto in dB con quella al centro del bin
    k = np.arange(n)
    resp = np.abs(np.sum(w * np.exp(-2j * np.pi * 0.5 * k / n)))
    scallop = -20.0 * np.log10(resp / w.sum())
    return WindowProperties(name, float(cg), float(enbw), float(scallop))


# ------------------------------------------------------------------ spectrum
def rfft_magnitude(x: np.ndarray, fs: float = TARGET_HZ, window: str = "hann",
                   nfft: int | None = None, correct_gain: bool = True):
    """Single-sided amplitude spectrum of x sampled at fs [Hz], as (freqs in Hz, amplitude).

    The mean is removed before the window. nfft > len(x) zero-pads: the bins
    get denser, the resolution stays 1 / T_obs. With correct_gain the
    amplitude of a sinusoid on a bin reads as its true amplitude.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    xw = (x - x.mean()) * WINDOWS[window](n)      # tolgo la media e applico la finestra
    m = nfft or n                                  # con nfft > n numpy aggiunge zeri in coda
    X = np.fft.rfft(xw, n=m)
    # ampiezza a singolo lato: raddoppio e divido per n
    mag = np.abs(X) * 2.0 / n
    if correct_gain:
        # divido per il guadagno coerente, così una sinusoide di ampiezza A si legge A
        mag = mag / window_properties(window, n).coherent_gain
    return np.fft.rfftfreq(m, d=1.0 / fs), mag


# ------------------------------------------------------------------ cost
def fft_cost(n: int, real_input: bool = False) -> dict:
    """Cost of an n-point radix-2 FFT in real operations, as a dict.

    A radix-2 butterfly does one complex multiply (4 real mult + 2 add) and
    two complex adds (4 real add): 4 mult and 6 add. With real input the
    Hermitian symmetry roughly halves the cost. memory_bytes counts data and
    twiddles in float32. Raises ValueError if n is not a power of two.
    """
    if n & (n - 1):                                # una potenza di due ha un solo bit acceso
        raise ValueError("n deve essere una potenza di due")
    butterflies = (n // 2) * int(np.log2(n))       # n/2 farfalle per ognuno dei log2(n) stadi
    mult, add = 4 * butterflies, 6 * butterflies
    if real_input:
        mult, add, butterflies = mult // 2, add // 2, butterflies // 2
    return {
        "n": n,
        "butterflies": butterflies,
        "real_mult": mult,
        "real_add": add,
        "total_ops": mult + add,
        "memory_bytes": 2 * n * 4 + (n // 2) * 2 * 4,   # dati complessi più i twiddle, in fp32
    }
