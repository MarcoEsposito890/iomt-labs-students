"""Tests of the Lab 2 additions: the clock model, label agreement, the windows file, the figures."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from iomtlab.config import SENSOR_CHANNELS
from iomtlab.signal import label_agreement, offset_after, shift_clock


def _rec(n=500, period_ms=10.0, t0=1_000_000.0, cut=250):
    """A recording with two phases, the boundary at sample cut."""
    t = t0 + period_ms * np.arange(n)
    df = pd.DataFrame({"timestamp": t, **{c: np.zeros(n) for c in SENSOR_CHANNELS}})
    df["label"] = np.where(np.arange(n) < cut, "SEATED", "WALKING")
    return df


def test_offset_after_matches_the_theory():
    assert offset_after(40, 60) == pytest.approx(2.4)            # 2,4 ms al minuto
    assert offset_after(40, 3600) == pytest.approx(144.0)
    assert offset_after(50, 86400) / 1000 == pytest.approx(4.32)  # 50 ppm: 4,3 s al giorno


def test_shift_clock_model():
    df = _rec()
    s = shift_clock(df, offset_ms=7.0, skew_ppm=100)
    t, t2 = df["timestamp"].to_numpy(), s["timestamp"].to_numpy()
    assert t2[0] == t[0] + 7.0
    assert t2[-1] - t2[0] == pytest.approx((t[-1] - t[0]) * (1 + 100e-6))
    assert df["timestamp"].iloc[0] == t[0]                          # l'originale non cambia


def test_label_agreement_drops_with_the_offset():
    a, b = _rec(), _rec()
    assert label_agreement(a, b) == 1.0
    # 100 ms di errore spostano un confine di 5 istanti su una griglia a 50 Hz
    acc = label_agreement(a, b, offset_ms=100.0)
    n_grid = len(np.arange(a.timestamp.iloc[0] + 100, a.timestamp.iloc[-1], 20.0))
    assert acc == pytest.approx(1 - 5 / n_grid, abs=1.5 / n_grid)
    assert label_agreement(a, b, offset_ms=200.0) < acc


def test_save_windows_round_trip(tmp_path):
    from iomtlab.io import save_windows
    from iomtlab.train import load_reference_windows
    W = {"X": np.random.default_rng(0).normal(size=(4, 50, 6)), "y": np.arange(4),
         "subject": np.array(["s01", "s01", "s02", "s02"], dtype=object), "purity": np.ones(4)}
    p = save_windows(tmp_path / "w.npz", W)
    d = load_reference_windows(p)
    assert d["X"].dtype == np.float32 and d["X"].shape == (4, 50, 6)
    assert list(d["subject"]) == ["s01", "s01", "s02", "s02"] and list(d["y"]) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        save_windows(tmp_path / "x.npz", {**W, "y": np.arange(3)})


def test_lab2_figures_draw():
    from iomtlab.plots import plot_agreement, plot_intervals, plot_phases, plot_window_purity
    from iomtlab.tug import phase_table
    a, b = _rec(), shift_clock(_rec(), 30.0)
    plot_phases(phase_table(a), phase_table(b))
    plot_intervals(a, b)
    plot_agreement([0, 10, 20], [1.0, 0.99, 0.98], mark=10.0)
    plot_window_purity(np.array([0, 0, 1, 2]), np.array([1.0, 0.5, 0.9, 0.7]))
