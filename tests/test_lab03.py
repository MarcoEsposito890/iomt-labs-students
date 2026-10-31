"""Tests of the Lab 3 additions: build_windows, forests, evaluation figures, reference split."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from iomtlab.evaluation import confusion_report, kfold_shuffled, subject_folds
from iomtlab.features import extract
from iomtlab.forests import (BYTES_PER_NODE, cross_predict, evaluate_forest, features_cost,
                             forest_cost, mdi_orders)

BLOCCHI = ("amplitude", "dispersion", "dynamics", "norm")


def _data(n=300, d=8, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, n)
    X = rng.normal(size=(n, d)) + y[:, None] * 0.8        # classi separabili
    g = np.array([f"s{i % 6:02d}" for i in range(n)], dtype=object)
    names = [f"mean_a{c}" for c in "xyz"] + [f"std_g{c}" for c in "xyz"] + ["mean_anorm", "std_anorm"]
    return X, y, g, names[:d]


def _rf():
    return RandomForestClassifier(n_estimators=5, random_state=0)


def test_cross_predict_predicts_every_row_once():
    X, y, g, _ = _data()
    pred, sc = cross_predict(_rf, X, y, subject_folds(g, 3, seed=0))
    assert pred.shape == y.shape and (pred >= 0).all()
    assert len(sc) == 3 and ((0 <= sc) & (sc <= 1)).all()


def test_cross_predict_rejects_incomplete_folds():
    X, y, g, _ = _data()
    folds = list(subject_folds(g, 3, seed=0))[:2]          # una piega in meno: righe mai in prova
    with pytest.raises(ValueError):
        cross_predict(_rf, X, y, folds)


def test_forest_cost_counts_nodes_and_comparisons():
    X, y, _, _ = _data()
    rf = _rf().fit(X, y)
    c = forest_cost(rf, X)
    assert c["nodes"] == sum(e.tree_.node_count for e in rf.estimators_)
    assert c["bytes"] == c["nodes"] * BYTES_PER_NODE
    assert c["leaves"] < c["nodes"]
    # al più un confronto per livello e per albero
    assert 0 < c["mean_comparisons"] <= c["trees"] * c["max_depth"]


def test_features_cost_counts_the_norm_once():
    one = features_cost(["mean_anorm"])
    both = features_cost(["mean_anorm", "std_anorm"])
    assert both - one < one                     # la norma non si paga due volte
    assert features_cost(["mean_ax", "std_ax"]) > features_cost(["mean_ax"])
    with pytest.raises(ValueError):
        features_cost(["fdom_ax"])


def test_features_cost_of_the_lab2_features():
    _, names = extract(np.zeros((1, 50, 6)), blocks=BLOCCHI)
    assert len(names) == 52
    assert features_cost(names) == 7410             # il numero della teoria (make_b2_method_figs)


def test_evaluate_forest_respects_the_configuration():
    X, y, g, names = _data()
    P = list(subject_folds(g, 3, seed=0))
    ords = mdi_orders(X, y, P, n_trees=5)
    assert all(sorted(o) == list(range(X.shape[1])) for o in ords)
    r = evaluate_forest(X, y, names, P, 4, 3, 2, orders=ords)
    assert r["k"] == 2 and r["n_trees"] == 4
    assert r["mean_comparisons"] <= 4 * 3
    full = evaluate_forest(X, y, names, P, 4, 3, None)
    assert full["k"] == len(names) and full["feature_ops"] > r["feature_ops"]


def test_reference_split_partitions_the_subjects(tmp_path):
    from iomtlab.train.compare import save_reference_split
    g = np.array([f"s{i % 7:02d}" for i in range(70)], dtype=object)
    sp = save_reference_split(g, path=tmp_path / "split.json")
    flat = sorted(s for f in sp["folds"] for s in f)
    assert flat == sorted(set(g)) and len(sp["folds"]) == 5 and sp["seed"] == 0
    # stesse pieghe di subject_folds con lo stesso seme
    ref = [sorted(set(g[te])) for _, te in subject_folds(g, 5, seed=0)]
    assert sp["folds"] == ref


def test_evaluation_figures_return_a_figure():
    from iomtlab.evaluation_figures import plot_confusion, plot_validation
    X, y, g, _ = _data()
    assert plot_validation([("a", [0.8, 0.9]), ("b", [0.7, 0.75])]) is not None
    pred, _ = cross_predict(_rf, X, y, kfold_shuffled(len(y), 3))
    for norm in (None, "row", "col"):
        assert plot_confusion(confusion_report(y, pred, normalize=norm)) is not None


def test_build_windows_matches_the_lab2_construction(data_dir):
    from iomtlab.signal import (build_windows, encode_labels, list_executions,
                                load_execution, make_windows, resample_to_hz)
    W = build_windows("sp", 50, 25)
    n = len(W["y"])
    assert W["X"].shape == (n, 50, 6) and len(W["purity"]) == n == len(W["subject"])
    s = W["subject"][0]
    e = list_executions(s, "sp")[0]
    _, Xs, ys = resample_to_hz(load_execution(s, e, "sp"), 50.0)
    Xw, yw, _ = make_windows(Xs, encode_labels(ys), 50, 25)
    assert np.allclose(W["X"][:len(yw)], Xw) and (W["y"][:len(yw)] == yw).all()
