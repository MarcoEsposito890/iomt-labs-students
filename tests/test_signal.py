"""Tests of the signal pipeline, run in CI."""

import numpy as np
import pytest

from iomtlab.config import TARGET_HZ, WINDOW_SHIFT, WINDOW_SIZE
from iomtlab.signal.loading import load_execution
from scipy.signal import welch

from iomtlab.signal.resample import resample_to_hz
from iomtlab.signal.timing import timing_report
from iomtlab.signal.windowing import (encode_labels, make_windows,
                                      normalize_full_scale)


@pytest.fixture(scope="module")
def df(data_dir):
    return load_execution("s16", 1, "sp", data_dir)


def test_raw_is_not_uniform(df):
    """Raw sampling is not uniform."""
    r = timing_report(df)
    assert r.std_dt_ms > 1.0
    assert 95 < r.mean_hz < 115


def test_resample_grid_is_uniform(df):
    t, X, y = resample_to_hz(df)
    assert np.allclose(np.diff(t), 1.0 / TARGET_HZ)
    assert X.shape[1] == 6
    assert len(t) == len(X) == len(y)


def test_walking_energy_below_nyquist(df):
    """Why 50 Hz: nearly all the energy is below 25 Hz."""
    _, X, y = resample_to_hz(df)
    mask = y == "WALKING"
    if mask.sum() < 128:
        pytest.skip("finestra WALKING troppo corta")
    f, p = welch(X[mask, 1], fs=TARGET_HZ, nperseg=128)
    assert p[f <= 20.0].sum() / p.sum() > 0.95


def test_normalization_bounded(df):
    _, X, _ = resample_to_hz(df)
    Xn = normalize_full_scale(X, "sp")
    assert Xn.min() >= -1.0 and Xn.max() <= 1.0
    assert Xn.dtype == np.float32


def test_window_shapes_and_purity(df):
    _, X, y = resample_to_hz(df)
    Xn = normalize_full_scale(X, "sp")
    Xw, yw, pur = make_windows(Xn, encode_labels(y))
    n_expected = (len(Xn) - WINDOW_SIZE) // WINDOW_SHIFT + 1
    assert Xw.shape == (n_expected, WINDOW_SIZE, 6)
    assert len(yw) == len(pur) == n_expected
    assert np.all((pur > 0.0) & (pur <= 1.0))


def test_windows_shorter_than_size_are_empty():
    X = np.zeros((10, 6), dtype=np.float32)
    y = np.zeros(10, dtype=np.int64)
    Xw, yw, pur = make_windows(X, y)
    assert len(Xw) == len(yw) == len(pur) == 0


def test_on_grid_does_not_extrapolate():
    import pandas as pd
    from iomtlab.config import SENSOR_CHANNELS
    from iomtlab.signal import on_grid
    df = pd.DataFrame({"timestamp": [0.0, 10.0, 20.0], "label": ["A", "A", "B"],
                       **{c: [0.0, 1.0, 2.0] for c in SENSOR_CHANNELS}})
    out = on_grid(df, np.array([0.0, 5.0, 15.0]))
    assert np.allclose(out[SENSOR_CHANNELS[0]], [0.0, 0.5, 1.5])
    assert list(out["label"]) == ["A", "A", "A"]
    with pytest.raises(ValueError):
        on_grid(df, np.array([0.0, 25.0]))
