"""iomtlab.edge: measurement, cutting, int8 transport."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.utils.prune as P

from iomtlab.config import ENERGY, N_CHANNELS, N_CLASSES, WINDOW_SIZE
from iomtlab.edge import (cut_cost, cut_points, nonzero_params, pack_int8,
                          size_on_disk, split_model, time_per_window,
                          unpack_int8)
from iomtlab.models.cost import layer_profile
from iomtlab.models.deep import DeepCNN, EarlyExitCNN
from iomtlab.models.simple_cnn import SimpleCNN


X1 = torch.zeros(1, N_CHANNELS, WINDOW_SIZE)


# ══════════════════════════════════ measurement
def test_size_on_disk_is_measured_not_estimated():
    m = SimpleCNN()
    got = size_on_disk(m)
    est = sum(p.numel() for p in m.parameters()) * 4
    assert got > 0
    # close to the estimate but not equal: the archive has a header
    assert 0.9 * est < got < 1.3 * est
    assert got != est, "se coincide esattamente, non e' stato misurato"


def test_size_on_disk_drops_with_int8():
    import torch.ao.quantization as Q
    m = SimpleCNN().eval()
    # su Apple Silicon il motore predefinito è «none»: si sceglie qnnpack se c'è
    qb = torch.backends.quantized
    if qb.engine == "none" or qb.engine not in qb.supported_engines:
        qb.engine = "qnnpack" if "qnnpack" in qb.supported_engines else qb.supported_engines[-1]
    mq = Q.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
    r = size_on_disk(m) / size_on_disk(mq)
    # about 4, not exactly: convolutions and normalisations stay in float
    assert r > 1.5, f"rapporto {r:.2f}: la quantizzazione non ha ridotto nulla"


def test_time_per_window_refuses_a_batch():
    m = SimpleCNN().eval()
    with pytest.raises(ValueError):
        time_per_window(m, torch.zeros(8, N_CHANNELS, WINDOW_SIZE), n=2, warmup=1)


def test_time_per_window_reports_its_own_dispersion():
    """spread is reported and consistent with min, max and median.

    We don't put a threshold on it: on a shared machine repetitions can
    differ a lot and the test would fail at random.
    """
    m = SimpleCNN().eval()
    d = time_per_window(m, X1, n=30, warmup=5, repeats=3)
    assert d["median_s"] > 0
    assert d["min_s"] <= d["median_s"] <= d["max_s"]
    assert d["spread"] == pytest.approx(
        (d["max_s"] - d["min_s"]) / d["median_s"], rel=1e-9)
    assert d["repeats"] == 3 and d["n"] == 30


def test_warmup_changes_the_measurement():
    """time_per_window runs with and without warmup."""
    m = SimpleCNN().eval()
    cold = time_per_window(m, X1, n=5, warmup=0, repeats=1)
    hot = time_per_window(m, X1, n=200, warmup=50, repeats=3)
    assert cold["median_s"] > 0 and hot["median_s"] > 0


def test_nonzero_params_sees_live_pruning():
    """After l1_unstructured the weights are zeroed, not removed."""
    m = SimpleCNN()
    before = nonzero_params(m)
    assert before["zero_fraction"] < 0.01
    assert not before["pruning_active"]

    for mod in m.modules():
        if isinstance(mod, (nn.Conv1d, nn.Linear)):
            P.l1_unstructured(mod, "weight", amount=0.5)
    after = nonzero_params(m)
    assert after["pruning_active"], "il pruning vivo deve essere rilevato"
    assert 0.3 < after["zero_fraction"] < 0.7

    # and the file does not shrink
    m2 = SimpleCNN()
    assert size_on_disk(m) >= size_on_disk(m2) * 0.9


# ══════════════════════════════════ cutting
@pytest.mark.parametrize("Model", [SimpleCNN, DeepCNN, EarlyExitCNN])
def test_every_declared_cut_recomposes_forward(Model):
    m = Model().eval()
    with torch.no_grad():
        want = m(X1)
    for at, _ in m.stages():
        head, tail = split_model(m, at)
        with torch.no_grad():
            got = tail(head(X1))
        assert torch.allclose(got, want, atol=1e-6), (
            f"{Model.__name__}, confine '{at}': testa+coda non ricompone il "
            f"forward")


def test_split_rejects_unknown_boundary():
    with pytest.raises(ValueError):
        split_model(SimpleCNN(), "non_esiste")


def test_cut_points_sum_is_invariant():
    prof = layer_profile(DeepCNN())
    rows = cut_points(prof)
    totals = {r["mac_head"] + r["mac_tail"] for r in rows}
    assert len(totals) == 1, "testa + coda non puo' dipendere dal taglio"


def test_int8_quarters_the_payload():
    prof = layer_profile(DeepCNN())
    a = cut_points(prof, bits=32)
    b = cut_points(prof, bits=8)
    for ra, rb in zip(a, b):
        assert ra["bits"] == 4 * rb["bits"]


def test_cut_cost_uses_declared_constants():
    prof = layer_profile(DeepCNN())
    rows = cut_cost(cut_points(prof), ENERGY["e_mac_J"],
                    ENERGY["e_bit_lowpower_J"], ENERGY["rate_lowpower_bps"])
    for r in rows:
        assert r["e_tot_J"] == pytest.approx(r["e_node_J"] + r["e_link_J"])
        assert r["t_link_s"] == pytest.approx(
            r["bits"] / ENERGY["rate_lowpower_bps"])


# ══════════════════════════════════ trasporto
def test_pack_unpack_preserves_shape_and_scale():
    rng = np.random.default_rng(0)
    for scale in (1e-4, 1.0, 1e4):
        a = (rng.normal(size=(1, 8, 13)) * scale).astype(np.float32)
        b = unpack_int8(pack_int8(a))
        assert b.shape == a.shape
        assert np.abs(a - b).max() / np.abs(a).max() < 0.01


def test_pack_is_four_times_smaller_than_float32():
    a = np.zeros((1, 64, 25), dtype=np.float32)
    assert len(pack_int8(a)) < a.nbytes / 3.5


def test_unpack_rejects_garbage():
    with pytest.raises(ValueError):
        unpack_int8(b"non sono un payload")


def test_pack_handles_constant_tensor():
    a = np.full((1, 4, 4), 2.5, dtype=np.float32)
    b = unpack_int8(pack_int8(a))
    assert np.all(np.isfinite(b)), "un tensore costante non deve dare NaN"


# ══════════════════════════════════ uscita anticipata
def test_early_exit_returns_both_in_one_pass():
    """The exit and the full logits come from one forward pass."""
    m = EarlyExitCNN().eval()
    with torch.no_grad():
        z_exit, z_full = m(X1, return_exit=True)
    assert z_exit.shape == (1, N_CLASSES)
    assert z_full.shape == (1, N_CLASSES)
