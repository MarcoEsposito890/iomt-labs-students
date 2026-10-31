"""Tests of the Lab 6 additions: cuts, delay, hops, the real split, the exit.

The numbers are those of the theory (block 4.1, «Energy and delay at every
cut», «Designing the cut instead of finding it», «An exit placed at the cut»)
and of GUIDA_LAB_INIZIO: they depend only on the architecture and on the
constants of config.ENERGY, not on the data.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import torch

from iomtlab import radio
from iomtlab.config import ENERGY
from iomtlab.edge import cut_cost, pack_int8
from iomtlab.edge.cuts import cut_delay, cut_profile, exit_curve, run_split, train_exit
from iomtlab.models import BottleneckCNN, ExampleCNN, layer_profile

E_MAC, E154, R154 = ENERGY["e_mac_J"], ENERGY["e_bit_lowpower_J"], ENERGY["rate_lowpower_bps"]
NODO, SERVER = 40e6, 2.4e9


def _rows(model, hops=1):
    return {r["name"]: r for r in cut_delay(cut_cost(cut_profile(model), E_MAC, E154, R154, hops),
                                            NODO, SERVER, R154, hops)}


# ── the profile ────────────────────────────────────────────────────────────
def test_profile_of_the_example_network():
    r = cut_profile(ExampleCNN())
    assert [x["name"] for x in r] == ["finestra", "conv1", "pool1", "conv2", "pool2", "conv3",
                                      "pool3", "dense", "classe"]
    d = {x["name"]: x for x in r}
    assert d["finestra"]["bits"] == 4800 and d["classe"]["bits"] == 3
    assert d["conv1"]["values"] == 1600 and d["pool2"]["values"] == 768
    assert d["pool3"]["bits"] == 3072 and d["dense"]["bits"] == 512
    assert all(x["mac_head"] + x["mac_tail"] == 574_656 for x in r)
    assert cut_profile(ExampleCNN(), act_bits=32)[1]["bits"] == 51_200     # 10,7 volte la finestra


@pytest.mark.parametrize("cls", [ExampleCNN, BottleneckCNN])
def test_profile_macs_agree_with_layer_profile(cls):
    m = cls()
    assert cut_profile(m)[-1]["mac_head"] == layer_profile(m).macs


def test_profile_needs_stages():
    with pytest.raises(NotImplementedError):
        cut_profile(torch.nn.Linear(3, 2))


# ── energy and delay: the numbers of the theory ────────────────────────────
def test_energy_at_the_ends_and_best_internal_cut():
    c = _rows(ExampleCNN())
    assert c["classe"]["e_tot_J"] * 1e3 == pytest.approx(0.3455, abs=5e-5)     # 0,346 mJ
    assert c["finestra"]["e_tot_J"] * 1e3 == pytest.approx(1.152)
    assert c["dense"]["e_tot_J"] * 1e3 == pytest.approx(0.4675, abs=5e-5)      # 0,467 mJ
    interni = [c[n] for n in ("conv1", "pool1", "conv2", "pool2", "conv3", "pool3", "dense")]
    assert min(interni, key=lambda r: r["e_tot_J"])["name"] == "dense"
    assert min(c.values(), key=lambda r: r["e_tot_J"])["name"] == "classe"


def test_bottleneck_cut_wins_without_headers():
    c = _rows(BottleneckCNN())
    co = c["bottleneck"]
    assert co["values"] == 48 and co["bits"] == 384 and co["mac_head"] == 307_072
    assert co["e_tot_J"] * 1e3 == pytest.approx(0.2764, abs=5e-5)              # 0,276 mJ
    assert co["e_tot_J"] < _rows(ExampleCNN())["classe"]["e_tot_J"]


def test_break_even_and_r_star():
    r = cut_profile(ExampleCNN())
    assert r[-1]["mac_head"] * E_MAC / r[0]["bits"] * 1e9 == pytest.approx(71.8, abs=0.05)
    assert r[0]["bits"] / (r[-1]["mac_head"] / NODO) / 1e3 == pytest.approx(334.1, abs=0.1)


def test_hops_multiply_link_energy_and_time_only():
    m = ExampleCNN()
    one, three = _rows(m, 1), _rows(m, 3)
    for n in one:
        assert three[n]["e_link_J"] == pytest.approx(3 * one[n]["e_link_J"])
        assert three[n]["e_link_node_J"] == pytest.approx(one[n]["e_link_J"])
        assert three[n]["t_link_s"] == pytest.approx(3 * one[n]["t_link_s"])
        assert three[n]["e_node_J"] == one[n]["e_node_J"]
    with pytest.raises(ValueError):
        cut_cost(cut_profile(m), E_MAC, E154, R154, hops=0)


def test_hops_at_50_nj_per_bit_bring_back_the_class():
    r = cut_profile(ExampleCNN())
    best = lambda h: min(cut_cost(r, E_MAC, 50e-9, R154, h), key=lambda x: x["e_tot_J"])["name"]  # noqa: E731
    assert best(1) == "finestra" and best(2) == "classe"     # pari a 71,8 / 50 = 1,44 salti


# ── the real split: packets and frames ─────────────────────────────────────
def _windows(n=6, seed=0):
    return np.random.default_rng(seed).normal(0, 5, (n, 50, 6)).astype(np.float32)


@pytest.mark.parametrize("cls,at,nbytes,nframes,ms", [(ExampleCNN, "pool2", 790, 7, 36.8),
                                                      (BottleneckCNN, "bottleneck", 70, 1, 10.6)])
def test_run_split_packets_frames_and_delay(cls, at, nbytes, nframes, ms):
    torch.manual_seed(0)
    m = cls().eval()
    s = run_split(m, at, _windows())
    assert {len(p) for p in s["packets"]} == {nbytes} and s["header_B"] == 22
    assert radio.frames(nbytes) == nframes
    r = _rows(m)[at]
    t = r["mac_head"] / NODO + radio.air_time_s(nbytes, R154) + r["mac_tail"] / SERVER
    assert 1e3 * t == pytest.approx(ms, abs=0.05)
    assert np.mean(s["logits_split"].argmax(1) == s["logits_full"].argmax(1)) >= 5 / 6


def test_bottleneck_with_headers_is_about_all_on_node():
    co = _rows(BottleneckCNN())["bottleneck"]
    e = co["e_node_J"] + radio.air_energy_J(70, E154)
    assert e * 1e3 == pytest.approx(0.351, abs=5e-4)                          # circa 0,35 mJ
    assert len(pack_int8(torch.zeros(1, 4, 12))) == 70


# ── the exit ───────────────────────────────────────────────────────────────
def test_exit_thresholds_of_the_theory():
    c = _rows(ExampleCNN())
    p2 = c["pool2"]
    e_testa = (p2["mac_head"] + 768 * 5) * E_MAC
    e_link = p2["bits"] * E154
    assert 1e3 * e_testa == pytest.approx(0.185, abs=5e-4) and 1e3 * e_link == pytest.approx(1.475, abs=5e-4)
    assert 1 - (c["classe"]["e_tot_J"] - e_testa) / e_link == pytest.approx(0.89, abs=0.005)
    assert 1 - (c["finestra"]["e_tot_J"] - e_testa) / e_link == pytest.approx(0.34, abs=0.005)


def test_exit_curve_ends_and_energy_line():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 200)
    z_exit, z_tail = rng.normal(size=(200, 5)), np.eye(5)[y] * 10
    rows = exit_curve(z_exit, z_tail, y, [0.0, 0.5, 1.01], 1.0, 2.0)
    assert rows[0]["q"] == 1.0 and rows[0]["energy_J"] == 1.0
    assert rows[-1]["q"] == 0.0 and rows[-1]["energy_J"] == 3.0
    assert rows[-1]["accuracy"] == 1.0 and rows[-1]["acc_local"] is None
    for r in rows:
        assert r["energy_J"] == pytest.approx(1.0 + (1 - r["q"]) * 2.0)


def test_train_exit_keeps_the_head_frozen():
    torch.manual_seed(0)
    m = ExampleCNN().eval()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    X, y = _windows(40), np.arange(40) % 5
    ex = train_exit(m, "pool2", X, y, epochs=2)
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    lin = [l for l in ex.modules() if isinstance(l, torch.nn.Linear)][0]
    assert (lin.in_features, lin.out_features) == (768, 5)                    # 3 840 MAC


def test_train_windows_is_the_recipe_of_fit_reference():
    from iomtlab import train
    X, y = _windows(64), np.arange(64) % 5
    out = []
    for _ in range(2):
        train.set_seed(train.RECIPE["seed"])
        m = BottleneckCNN()
        train.train_windows(m, X, y, epochs=1)
        out.append(m.state_dict())
    assert all(torch.equal(out[0][k], out[1][k]) for k in out[0])
