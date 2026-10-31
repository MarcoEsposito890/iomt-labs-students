"""Tests of the frequency analysis used in the block 1.1 theory."""
import numpy as np
import pytest

from iomtlab.signal.spectral import fft_cost, rfft_magnitude, window_properties


def test_window_properties_match_reference():
    """Window figures against Harris (1978)."""
    r = window_properties("rect")
    assert r.enbw == pytest.approx(1.00, abs=0.01)
    assert r.scalloping_loss_db == pytest.approx(3.92, abs=0.05)
    h = window_properties("hann")
    assert h.enbw == pytest.approx(1.50, abs=0.01)
    assert h.scalloping_loss_db == pytest.approx(1.42, abs=0.05)


def test_rfft_recovers_amplitude_on_bin():
    """A sine centred on a bin: the amplitude is recovered."""
    fs, n, f0, a0 = 50.0, 250, 5.0, 2.0          # 5 Hz falls on a bin
    t = np.arange(n) / fs
    f, m = rfft_magnitude(a0 * np.sin(2 * np.pi * f0 * t), fs, "hann")
    assert m[np.argmax(m)] == pytest.approx(a0, rel=0.02)
    assert f[np.argmax(m)] == pytest.approx(f0, abs=fs / n)


def test_zero_padding_does_not_add_resolution():
    """Zero padding makes the bins denser but does not resolve two close lines."""
    fs, n = 50.0, 50
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 5.0 * t) + np.sin(2 * np.pi * 5.4 * t)
    _, m1 = rfft_magnitude(x, fs, "hann")
    _, m8 = rfft_magnitude(x, fs, "hann", nfft=8 * n)
    peaks = lambda v: np.sum((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:])
                             & (v[1:-1] > 0.5 * v.max()))
    assert peaks(m1) == 1 and peaks(m8) == 1   # still a single peak


def test_fft_cost_scaling():
    c256, c512 = fft_cost(256), fft_cost(512)
    assert c256["butterflies"] == 128 * 8
    assert c512["total_ops"] > 2 * c256["total_ops"]     # grows faster than linear
    assert fft_cost(256, real_input=True)["total_ops"] == c256["total_ops"] // 2
    with pytest.raises(ValueError):
        fft_cost(100)
