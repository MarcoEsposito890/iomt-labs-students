"""Tests of the Lab 7 additions: routing on conformal sets, energy, channel, figures.

The fixed numbers are those of the theory (deck «Affidabilità», «Split
conformal prediction»: n = 2000, alpha = 0.10, k = 1801) and of Lab 5 (520
nodes per channel deciding on the node, 13 sending the raw window).
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from iomtlab.uncertainty import (apply_temperature, conformal_quantile, conformal_sets,
                                 coverage_by_subject, energy_constants, energy_per_decision,
                                 link_load, nonconformity, routing_table)


def _probs(n, seed=0, K=5, sharp=2.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    z = rng.normal(size=(n, K))
    z[np.arange(n), y] += sharp                      # la classe vera ha un logit più alto
    return apply_temperature(z, 1.0), y


def test_k_of_the_theory():
    assert int(np.ceil((2000 + 1) * (1 - 0.10))) == 1801
    s = np.arange(1, 2001) / 2001.0
    assert conformal_quantile(s, 0.10) == pytest.approx(1801 / 2001)


def test_nonconformity_is_one_minus_true_probability():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    assert np.allclose(nonconformity(p, [0, 1]), [0.3, 0.9])


def test_coverage_holds_on_exchangeable_data():
    cov = []
    for seed in range(40):
        p, y = _probs(600, seed)
        S = conformal_sets(p[:300], y[:300], p[300:], 0.1)
        cov.append(S[np.arange(300), y[300:]].mean())
    assert np.mean(cov) >= 0.89                      # in media almeno 1 - α, a meno del rumore


def test_routing_table_counts_singletons_and_forwards_the_rest():
    p_cal, y_cal = _probs(500, 1)
    p_te, y_te = _probs(400, 2)
    server = y_te.copy()                             # un server che non sbaglia mai
    t = routing_table(p_cal, y_cal, p_te, y_te, [0.05, 0.1, 0.4], server_pred=server)
    assert list(t["alpha"]) == [0.05, 0.1, 0.4]
    for _, r in t.iterrows():
        S = (1 - p_te) <= r.qhat
        single = S.sum(1) == 1
        assert r.q == pytest.approx(single.mean())
        assert r["empty"] == pytest.approx((S.sum(1) == 0).mean())
        assert r.acc_server == 1.0
        assert r.acc_system == pytest.approx(r.q * r.acc_local + (1 - r.q))
    assert t["empty"].iloc[-1] > 0                   # con α grande compaiono gli insiemi vuoti
    assert (np.diff(t["mean_size"]) <= 0).all()      # α più grande, insiemi più piccoli


def test_coverage_by_subject():
    S = np.array([[1, 0], [0, 1], [1, 0], [1, 1]], bool)
    c = coverage_by_subject(S, [0, 0, 0, 1], ["b", "b", "a", "a"])
    assert c.to_dict() == {"a": 1.0, "b": 0.5}


def test_energy_constants_without_and_with_curve():
    E = energy_constants()
    assert 1e3 * E["e_full_J"] == pytest.approx(0.3448, abs=1e-4)
    assert 1e3 * E["e_raw_J"] == pytest.approx(1.152)
    assert 1e3 * E["e_head_J"] == pytest.approx(0.1824) and 1e3 * E["e_link_J"] == pytest.approx(1.47456)
    E2 = energy_constants({"e_head_J": 1e-4, "e_link_J": 2e-4})
    assert E2["e_head_J"] == 1e-4 and E2["e_link_J"] == 2e-4 and "curve" in E2["source"]


def test_energy_per_decision_scalar_and_array():
    assert energy_per_decision(0.5, 1.0, 2.0) == 2.0
    assert np.allclose(energy_per_decision([0, 1], 1.0, 2.0), [3.0, 1.0])


def test_link_load_ends_match_lab5():
    assert link_load(1.0)["nodes"] == 520 and link_load(0.0)["nodes"] == 13
    assert link_load(1.0)["bit_s"] == 2 * 8 * 18                     # 1 B in una trama
    assert link_load(1.0, decisions_per_s=4)["nodes"] == 260
    a = link_load(np.array([0.0, 1.0]))
    assert list(a["nodes"]) == [13.0, 520.0]


def test_figures_draw():
    import pandas as pd
    from iomtlab.uncertainty.figures import plot_calibration_pair, plot_coverage, plot_routing
    p_cal, y_cal = _probs(300, 3)
    p_te, y_te = _probs(200, 4)
    t = routing_table(p_cal, y_cal, p_te, y_te, [0.05, 0.1, 0.2], server_pred=y_te)
    assert isinstance(t, pd.DataFrame)
    plot_calibration_pair(p_te, p_te, y_te)
    plot_coverage(t)
    plot_routing(t, {"A": energy_per_decision(t["q"], 1e-3, 1e-3), "_all_local": 1e-3})
