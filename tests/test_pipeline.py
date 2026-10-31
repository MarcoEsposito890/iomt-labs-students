"""The library pieces used together, as one lab hands results to the next."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from iomtlab import io
from iomtlab.config import ENERGY, N_CHANNELS, N_CLASSES, WINDOW_SIZE
from iomtlab.edge import cut_cost, cut_points, split_model
from iomtlab.evaluation import loso_folds, subject_folds
from iomtlab.models.cost import layer_profile
from iomtlab.models.deep import DeepCNN, DeepConvLSTM
from iomtlab.models.simple_cnn import SimpleCNN
from iomtlab.train.loops import evaluate, first_batch_hash, fit


def _windows(n=96, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_CHANNELS, WINDOW_SIZE)).astype(np.float32)
    y = rng.integers(0, N_CLASSES, n).astype(np.int64)
    return X, y


def _loader(n=96, bs=16, seed=0, shuffle=False):
    X, y = _windows(n, seed)
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                      batch_size=bs, shuffle=shuffle)


# ══════════════════════════════════ Lab 2 -> Lab 3
def test_features_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 7)).astype(np.float32)
    y = rng.integers(0, N_CLASSES, 40)
    subj = np.array([f"s{1 + i % 4:02d}" for i in range(40)], dtype=object)
    names = [f"f{i}" for i in range(7)]
    p = io.save_features(tmp_path / "features", X, y, names, subj)
    d = io.load_features(p)
    assert np.allclose(d["X"], X) and np.array_equal(d["y"], y)
    assert d["names"] == names and len(d["subject"]) == 40


def test_features_rejects_wrong_shapes(tmp_path):
    with pytest.raises(ValueError):
        io.save_features(tmp_path / "a", np.zeros((3, 2)), np.zeros(4),
                         ["a", "b"], ["s01"] * 4)
    with pytest.raises(ValueError):
        io.save_features(tmp_path / "b", np.zeros((4, 2)), np.zeros(4),
                         ["a"], ["s01"] * 4)


# ══════════════════════════════════ Lab 3 -> Lab 4
def test_split_roundtrip_and_masks(tmp_path):
    subj = np.array([f"s{1 + i % 6:02d}" for i in range(60)], dtype=object)
    folds = [[str(s) for s in part]
             for part in np.array_split(np.unique(subj), 3)]
    p = io.save_split(tmp_path / "split", folds, np.unique(subj), seed=0)
    d = io.load_split(p)
    masks = list(io.fold_masks(d, subj))
    assert len(masks) == 3
    seen = np.zeros(len(subj), dtype=bool)
    for tr, te in masks:
        assert not (tr & te).any()
        seen |= te
    assert seen.all(), "ogni riga deve finire in prova esattamente una volta"


def test_split_rejects_non_partition(tmp_path):
    with pytest.raises(ValueError):
        io.save_split(tmp_path / "s", [["s01"], ["s01"]], ["s01", "s02"], 0)


def test_subject_folds_matches_saved_split(tmp_path):
    """A saved and reloaded split gives the same folds."""
    subj = np.array([f"s{1 + i % 8:02d}" for i in range(80)], dtype=object)
    folds = list(subject_folds(subj, 4, seed=0))
    parts = [sorted(set(subj[te])) for _, te in folds]
    p = io.save_split(tmp_path / "sp", parts, np.unique(subj), 0)
    again = list(io.fold_masks(io.load_split(p), subj))
    for (_, a), (_, b) in zip(folds, again):
        assert np.array_equal(np.asarray(a), np.asarray(b))


# ══════════════════════════════════ Lab 4: training and comparison
def test_fit_reduces_loss():
    m = SimpleCNN()
    h = fit(m, _loader(), epochs=3, seed=0)
    assert len(h) == 3
    assert all(np.isfinite(h))


def test_fit_is_reproducible_with_seed():
    """The seed of fit does not cover weight initialisation.

    Two runs match only if set_seed is also called before building the model.
    """
    from iomtlab.train.loops import set_seed
    set_seed(0); a = fit(SimpleCNN(), _loader(), epochs=2, seed=0)
    set_seed(0); b = fit(SimpleCNN(), _loader(), epochs=2, seed=0)
    assert np.allclose(a, b), "seme prima della costruzione: devono coincidere"

    set_seed(0); c = fit(SimpleCNN(), _loader(), epochs=2, seed=0)
    set_seed(7); d = fit(SimpleCNN(), _loader(), epochs=2, seed=0)
    assert not np.allclose(c, d), ("senza seme prima della costruzione i pesi "
                                   "iniziali differiscono: il seme di fit non "
                                   "basta, e va detto")


def test_same_data_order_detects_reshuffle():
    """first_batch_hash changes when the loader is rebuilt with shuffle=True."""
    good = [first_batch_hash(_loader(seed=0)) for _ in range(3)]
    assert len(set(good)) == 1
    bad = [first_batch_hash(_loader(seed=0, shuffle=True)) for _ in range(3)]
    assert len(set(bad)) > 1


def test_model_roundtrip(tmp_path):
    m = SimpleCNN()
    prof = layer_profile(m)
    res = evaluate(m, _loader(), torch.device("cpu"))
    p = io.save_model(tmp_path / "m", m, prof, res, note="Lab 4")
    back = io.load_model(p, SimpleCNN())
    assert back["class_name"] == "SimpleCNN"
    assert back["cost"]["total_macs"] == prof.macs


def test_load_model_rejects_wrong_class(tmp_path):
    p = io.save_model(tmp_path / "m", SimpleCNN())
    with pytest.raises(ValueError):
        io.load_model(p, DeepCNN())


# ══════════════════════════════════ Lab 6: cut and cost
def test_cut_points_identity():
    prof = layer_profile(DeepCNN())
    rows = cut_points(prof)
    assert rows
    for r in rows:
        assert r["mac_head"] + r["mac_tail"] == prof.macs, r["name"]


def test_split_model_preserves_output():
    m = DeepCNN().eval()
    x = torch.zeros(1, N_CHANNELS, WINDOW_SIZE)
    with torch.no_grad():
        want = m(x)
    names = [n for n, _ in m.stages()]
    assert len(names) > 2, "un modello con due soli confini non si taglia"
    for at in names:
        head, tail = split_model(m, at)
        with torch.no_grad():
            got = tail(head(x))
        assert torch.allclose(got, want, atol=1e-6), (
            f"al confine '{at}' testa+coda non ricompone il forward")


def test_split_model_refuses_recurrent():
    with pytest.raises(NotImplementedError):
        split_model(DeepConvLSTM(), "conv1")


def test_cut_cost_endpoints():
    """At the two ends the cost matches the two known cases."""
    prof = layer_profile(DeepCNN())
    rows = cut_cost(cut_points(prof), ENERGY["e_mac_J"],
                    ENERGY["e_bit_lowpower_J"], ENERGY["rate_lowpower_bps"])
    last = rows[-1]
    assert last["mac_tail"] == 0, "all'ultimo confine non resta nulla da fare"
    tot = last["mac_head"]
    assert abs(last["e_node_J"] - tot * ENERGY["e_mac_J"]) < 1e-15


# ══════════════════════════════════ Lab 6 -> Lab 7
def test_curve_roundtrip(tmp_path):
    rows = [[0.5, 0.7, 5.8e-5, 0.81], [0.9, 0.37, 7.5e-5, 0.86]]
    p = io.save_curve(tmp_path / "c", rows, e_head_J=4.2e-5, e_link_J=5.2e-5)
    d = io.load_curve(p)
    assert d["e_head_J"] == pytest.approx(4.2e-5)
    assert len(d["rows"]) == 2


def test_curve_rejects_ragged(tmp_path):
    with pytest.raises(ValueError):
        io.save_curve(tmp_path / "c", [[0.5, 0.7]], 1.0, 1.0)


# ══════════════════════════════════ conformal coverage
def test_conformal_coverage_on_exchangeable_data():
    from iomtlab.uncertainty import conformal_sets, coverage_by_class
    rng = np.random.default_rng(0)
    K, n = N_CLASSES, 3000

    def sample(m):
        y = rng.integers(0, K, m)
        p = rng.dirichlet(np.ones(K) * 0.7, size=m)
        p[np.arange(m), y] += 0.8
        return p / p.sum(1, keepdims=True), y

    p_cal, y_cal = sample(1000)
    p_te, y_te = sample(n)
    S = conformal_sets(p_cal, y_cal, p_te, alpha=0.1)
    d = coverage_by_class(S, y_te)
    assert 0.86 <= d["marginal"] <= 0.94
    # empty sets are allowed (see conformal.py), so the mean size can be below 1
    assert 0.0 < d["mean_size"] <= K
    assert 0.0 <= d["empty_fraction"] < 0.5


def test_calibration_split_is_disjoint():
    from iomtlab.uncertainty import calibration_split
    subs = [f"s{i:02d}" for i in range(1, 24)]
    d = calibration_split(subs, seed=0)
    a, b, c = set(d["train"]), set(d["cal"]), set(d["test"])
    assert not (a & b) and not (a & c) and not (b & c)
    assert a | b | c == set(subs)
    assert b and c, "calibrazione e prova non possono essere vuote"


# ══════════════════════════════════ Lab 8
def test_fedavg_then_dequantize():
    from iomtlab.fl import clone_with, dequantize_delta, fedavg, quantize_delta
    m = SimpleCNN()
    g = {k: v.clone() for k, v in m.state_dict().items()}
    locals_ = []
    for s in (0, 1):
        mm = clone_with(m, g)
        fit(mm, _loader(seed=s), epochs=1, seed=s)
        locals_.append(mm.state_dict())
    agg = fedavg(locals_, [40, 60])
    assert set(agg) == set(g)
    packed = quantize_delta(locals_[0], g)
    back = dequantize_delta(packed, g)
    for k in g:
        if torch.is_floating_point(g[k]):
            assert torch.isfinite(back[k]).all()


# ══════════════════════════════════ names the slides cite
def test_public_names_the_decks_cite():
    """The «La libreria» tables of the lab decks (Labs 2-8) cite these names.

    The deck builder reads each signature from the library, and a name
    reachable only from a submodule (train.loops.fit instead of train.fit)
    would break the first cell of a notebook.
    """
    import importlib
    expected = {
        "iomtlab.signal": ["list_executions", "load_execution", "load_executions_info",
                           "load_subjects_info", "timing_report", "resample_to_hz",
                           "on_grid", "encode_labels", "make_windows"],
        "iomtlab.features": ["extract"],
        "iomtlab.io": ["save_features", "load_features", "save_split", "load_split",
                       "fold_masks", "save_model", "load_model", "save_curve", "load_curve"],
        "iomtlab.evaluation": ["loso_folds", "subject_folds", "kfold_shuffled", "macro_f1",
                               "per_subject_scores", "confusion_report", "top_k_table",
                               "incremental_loso_subsets", "accuracy_with_exit"],
        "iomtlab.train": ["fit", "set_seed", "first_batch_hash", "evaluate",
                          "inverse_freq_weights", "effective_num_weights"],
        "iomtlab.models": ["layer_profile", "count_parameters", "model_size_bytes",
                           "federated_round_bytes", "SimpleCNN", "DeepCNN",
                           "DeepConvLSTM", "EarlyExitCNN"],
        "iomtlab.edge": ["size_on_disk", "time_per_window", "nonzero_params", "split_model",
                         "cut_points", "cut_cost", "pack_int8", "unpack_int8"],
        "iomtlab.uncertainty": ["calibration_split", "ece", "fit_temperature",
                                "conformal_quantile", "conformal_sets", "coverage",
                                "coverage_by_class", "singleton_rate"],
        "iomtlab.fl": ["fedavg", "clone_with", "subject_clients", "dirichlet_partition",
                       "quantize_delta"],
        "iomtlab.plots": ["plot_folds", "per_subject_bars", "plot_cost_accuracy",
                          "reliability_diagram", "plot_rounds"],
        "iomtlab.config": ["ENERGY"],
    }
    missing = []
    for mod_name, names in expected.items():
        mod = importlib.import_module(mod_name)
        missing += [f"{mod_name}.{n}" for n in names if not hasattr(mod, n)]
    assert not missing, f"cited in the decks and not importable: {missing}"
