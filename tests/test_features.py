"""Tests of the hand-crafted features."""
import numpy as np
import pytest

from iomtlab.config import N_CHANNELS, TARGET_HZ, WINDOW_SIZE
from iomtlab.features import BLOCKS, extract

W = 32


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.normal(size=(W, WINDOW_SIZE, N_CHANNELS)).astype(np.float32)


def test_each_block_has_consistent_names(X):
    for b in BLOCKS:
        v, names = extract(X, blocks=(b,))
        assert v.shape == (W, len(names))
        assert len(set(names)) == len(names)


def test_default_is_minimal(X):
    v, names = extract(X)
    assert v.shape[1] == 6 * N_CHANNELS      # 4 amplitude + 2 dispersion per channel


def test_all_blocks_composable(X):
    v, names = extract(X, blocks=tuple(BLOCKS))
    assert v.shape == (W, len(names))
    assert np.isfinite(v).all()


def test_dominant_frequency_is_found():
    """A known sine shows up as the dominant frequency."""
    f0 = 5.0
    t = np.arange(WINDOW_SIZE) / TARGET_HZ
    x = np.zeros((1, WINDOW_SIZE, N_CHANNELS), dtype=np.float32)
    x[0, :, 0] = np.sin(2 * np.pi * f0 * t)
    v, names = extract(x, blocks=("spectral",))
    assert v[0, names.index("fdom_ax")] == pytest.approx(f0, abs=TARGET_HZ / WINDOW_SIZE)


def test_correlation_is_bounded(X):
    v, names = extract(X, blocks=("cross_axis",))
    assert np.all(np.abs(v) <= 1.0 + 1e-6)


def test_unknown_block_raises(X):
    with pytest.raises(ValueError):
        extract(X, blocks=("inesistente",))


def test_sma_reference_block():
    """The Lab 3 «Scrivi tu» reference: mean of |x|+|y|+|z| per triad, not a default block."""
    import numpy as np
    from iomtlab import features as FE
    X = np.ones((4, 50, 6)) * np.array([1, -2, 3, 0.5, -0.5, 1])
    F, names = FE.sma(X)
    assert names == ["sma_a", "sma_w"] and np.allclose(F, [[6, 2]] * 4)
    assert "sma" not in FE.BLOCKS
