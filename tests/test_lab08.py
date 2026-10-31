"""Tests of the Lab 8 additions: round cost, FedAvg and DSGD loops, consensus, figures.

The fixed numbers depend only on the example network and on the constants of
the course (config.ENERGY, iomtlab.radio): 56 805 parameters, 227 220 B in
float32, 1 959 frames per model, 16.7 s of radio per round in float32 and
10.4 s with the update in int8 (the 2.8 and 1.7 min of GUIDA_LAB_INIZIO are
10 rounds of these); the consensus numbers are those of Xiao and Boyd.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import torch

from iomtlab import radio
from iomtlab.fl import (bytes_to_reach, grid_graph, metropolis_weights, ring_graph, round_cost,
                        run_centralised, run_consensus, run_dsgd, run_fedavg, second_eigenvalue,
                        disagreement, iterations_to, uniform_weights, update_bytes)
from iomtlab.models import ExampleCNN


# ── bytes and the cost of a round ──────────────────────────────────────────
def test_update_bytes_and_frames():
    m = ExampleCNN()
    assert update_bytes(m) == 227_220
    assert update_bytes(m, 8) == 56_805 + 4 * 10           # un byte per valore, una scala per tensore
    assert radio.frames(update_bytes(m)) == 1959 and radio.frames(update_bytes(m, 8)) == 491
    with pytest.raises(ValueError):
        update_bytes(m, 16)


def test_round_cost_float32_and_int8():
    m = ExampleCNN()
    f, i = round_cost(m, 190), round_cost(m, 190, up_bits=8)
    assert f["frames"] == 2 * 1959 and i["frames"] == 1959 + 491
    assert f["radio_s"] == pytest.approx(16.67, abs=0.01) and i["radio_s"] == pytest.approx(10.42, abs=0.01)
    assert 10 * f["radio_s"] / 60 == pytest.approx(2.8, abs=0.05)      # 2,8 min in float32
    assert 10 * i["radio_s"] / 60 == pytest.approx(1.7, abs=0.05)      # 1,7 min in int8
    assert f["radio_J"] == pytest.approx(f["air_B"] * 8 * 240e-9)
    assert f["train_MAC"] == 3 * 574_656 * 190


def test_round_cost_theory_case_compute_dominates():
    c = round_cost(ExampleCNN(), 500, epochs=5, up_bits=8)
    assert c["train_MAC"] == 3 * 574_656 * 500 * 5 and c["train_J"] > c["radio_J"]


# ── the training loops ─────────────────────────────────────────────────────
def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 5
    X = rng.normal(0, 1, (n, 50, 6)).astype(np.float32)
    X[:, :, 0] += 20 * y[:, None]                         # una classe per livello: separabili
    return X, y


def test_fedavg_history_and_bytes():
    torch.manual_seed(0)
    m0 = ExampleCNN()
    X, y = _data()
    clients = [np.arange(0, 30), np.arange(30, 60)]
    before = {k: v.clone() for k, v in m0.state_dict().items()}
    h = run_fedavg(m0, X, y, clients, X, y, rounds=2)
    assert [r for r, _, _ in h] == [0, 1, 2] and h[0][2] == 0
    assert h[2][2] == 2 * 2 * 2 * 227_220                  # 2 round, 2 client, giù e su
    assert all(torch.equal(before[k], v) for k, v in m0.state_dict().items())   # m0 non cambia
    h8 = run_fedavg(m0, X, y, clients, X, y, rounds=1, bits=8)
    assert h8[1][2] == 2 * (227_220 + 56_845)


def test_fedavg_is_deterministic():
    torch.manual_seed(0)
    m0 = ExampleCNN()
    X, y = _data()
    clients = {"a": np.arange(30), "b": np.arange(30, 60)}
    assert run_fedavg(m0, X, y, clients, X, y, rounds=1) == run_fedavg(m0, X, y, clients, X, y, rounds=1)


def test_centralised_and_dsgd():
    torch.manual_seed(0)
    m0 = ExampleCNN()
    X, y = _data()
    hc = run_centralised(m0, X, y, X, y, epochs=2, raw_bytes=600 * len(y))
    assert len(hc) == 3 and all(b == 600 * len(y) for _, _, b in hc)
    W = metropolis_weights(ring_graph(3))
    hd = run_dsgd(m0, X, y, [np.arange(i, 60, 3) for i in range(3)], W, X, y, rounds=1)
    assert hd[1][2] == 3 * 227_220                         # ogni nodo trasmette una volta
    with pytest.raises(ValueError):
        run_dsgd(m0, X, y, [np.arange(30), np.arange(30, 60)], W, X, y, rounds=1)


def test_bytes_to_reach():
    h = [(0, 0.1, 0), (1, 0.5, 10), (2, 0.7, 20)]
    assert bytes_to_reach(h, 0.6) == 20 and bytes_to_reach(h, 0.9) is None


# ── consensus ──────────────────────────────────────────────────────────────
def test_graphs_and_weights():
    A = grid_graph(4, 4)
    assert A.sum(1).min() == 2 and A.sum(1).max() == 4 and (A == A.T).all()
    assert ring_graph(16).sum(1).tolist() == [2] * 16
    Wm = metropolis_weights(A)
    assert np.allclose(Wm, Wm.T) and np.allclose(Wm.sum(0), 1) and np.allclose(Wm.sum(1), 1)
    Wu = uniform_weights(A)
    assert np.allclose(Wu.sum(1), 1) and not np.allclose(Wu.sum(0), 1)


def test_second_eigenvalue_of_the_ring():
    # anello con pesi 1/3: autovalori (1 + 2 cos(2πk/n)) / 3
    assert second_eigenvalue(metropolis_weights(ring_graph(16))) == pytest.approx(
        (1 + 2 * np.cos(2 * np.pi / 16)) / 3)
    assert second_eigenvalue(metropolis_weights(ring_graph(6))) == pytest.approx(2 / 3)


def test_consensus_reaches_the_mean_only_with_doubly_stochastic_weights():
    x0 = np.random.default_rng(0).normal(20, 2, 16)
    t = run_consensus(metropolis_weights(grid_graph(4, 4)), x0, 300)
    assert t.shape == (301, 16) and abs(t[-1].mean() - x0.mean()) < 1e-9
    e = disagreement(t)
    assert e[0] == pytest.approx(1.0) and iterations_to(e, 0.01) is not None
    eu = disagreement(run_consensus(uniform_weights(grid_graph(4, 4)), x0, 300))
    assert iterations_to(eu, 0.01) is None and eu[-1] > 0.01      # d'accordo, ma non sulla media


def test_ring_iterations_grow_with_nodes():
    k = {}
    for n in (16, 32):
        x = np.random.default_rng(1).normal(20, 2, n)
        k[n] = iterations_to(disagreement(run_consensus(metropolis_weights(ring_graph(n)), x, 2000)))
    assert k[32] > 2 * k[16]


def test_figures_draw():
    from iomtlab.fl import dirichlet_partition, subject_clients
    from iomtlab.fl.figures import plot_client_mix, plot_consensus, plot_round_cost
    y = np.arange(100) % 5
    plot_client_mix(subject_clients(np.repeat(["a", "b"], 50)), y)
    plot_client_mix(dirichlet_partition(y, 4, 0.1)[0], y, legend=False)
    plot_round_cost({"f32": round_cost(ExampleCNN(), 100)})
    plot_consensus({"r": np.geomspace(1, 1e-3, 50)}, {"r": 0.9})
