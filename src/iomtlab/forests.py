"""Random forests in Lab 3: cross-validated predictions and the cost on a node.

A forest is stored on a microcontroller as arrays, one entry per node:
feature index, threshold, left and right child. Tabanelli, Tagliavini and
Benini (IEEE TCAD 41(11), 2022, Table V) measure 16 bytes per node in this
form, against 24 bytes per node with the trees unrolled into if-else code.
A decision walks one path per tree, one comparison per internal node.

Every function takes the folds as boolean (train_mask, test_mask) pairs, as
iomtlab.evaluation yields them.
"""
from __future__ import annotations

import math

import numpy as np

from iomtlab.config import N_CLASSES, WINDOW_SIZE
from iomtlab.evaluation import macro_f1

__all__ = ["BYTES_PER_NODE", "cross_predict", "forest_cost", "features_cost",
           "mdi_orders", "evaluate_forest"]

#: Bytes per node of a forest stored as arrays (Tabanelli et al., 2022).
BYTES_PER_NODE = 16


def cross_predict(make_model, X, y, folds):
    """Cross-validated predictions: a new model per fold, every row predicted once.

    make_model() returns an unfitted scikit-learn classifier; folds yields
    boolean (train_mask, test_mask) pairs whose test masks cover the rows.
    Returns (pred, scores): pred (n,) the prediction of each row by the
    model that did not train on it, scores (k,) the macro-F1 of each fold.
    Raises ValueError if a row is never in test.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    pred = np.full(len(y), -1, dtype=np.int64)
    scores = []
    for tr, te in folds:
        m = make_model().fit(X[tr], y[tr])      # un modello nuovo, addestrato solo sulla piega
        pred[te] = m.predict(X[te])
        scores.append(macro_f1(y[te], pred[te]))
    if (pred < 0).any():
        raise ValueError(f"{int((pred < 0).sum())} righe mai in prova: le pieghe non coprono i dati")
    return pred, np.array(scores)


def forest_cost(forest, X=None) -> dict:
    """What a fitted forest costs on a node.

    Returns a dict with trees, nodes (all nodes, leaves included), leaves,
    max_depth (deepest tree), bytes (nodes x BYTES_PER_NODE) and, if X is
    given, mean_comparisons: comparisons per decision averaged over the
    rows of X, one per internal node on the path of every tree.
    """
    trees = [e.tree_ for e in forest.estimators_]
    nodes = int(sum(t.node_count for t in trees))
    leaves = int(sum(t.n_leaves for t in trees))
    out = {"trees": len(trees), "nodes": nodes, "leaves": leaves,
           "max_depth": int(max(t.max_depth for t in trees)),
           "bytes": nodes * BYTES_PER_NODE}
    if X is not None:
        path, _ = forest.decision_path(np.asarray(X, dtype=np.float32))
        visited = np.asarray(path.sum(1)).ravel()          # nodi visitati, foglie comprese
        out["mean_comparisons"] = float((visited - len(trees)).mean())   # una foglia per albero non confronta
    return out


def _ops_per_feature(T: int) -> dict:
    """Operations of one feature on one channel of T samples, by prefix.

    The convention of the theory (block 1.1; make_b2_method_figs.feature_cost):
    an addition, a multiplication, a comparison or a multiply-add counts 1, a
    sort of n values counts floor(n log2 n). Each feature is counted alone.
    """
    return {
        "mean": T + 1,                        # T somme e una divisione
        "min": T, "max": T,                   # un confronto per campione
        "rms": 2 * T + 2,                     # quadrati e somme, divisione e radice
        "std": 3 * T + 2,                     # scarti, quadrati, somme, divisione e radice
        "iqr": int(T * math.log2(T)),         # l'ordinamento
        "jerk": 3 * (T - 1) + 1,              # differenze, moduli, somme, divisione
        "zcr": 2 * T + 1 + T + (T - 1) + (T - 1),   # media e scarti, segni, cambi, conteggio
    }


def features_cost(names, T: int | None = None) -> int:
    """Operations per window of the features in names, on windows of T samples.

    names are those of iomtlab.features (mean_ax, std_wnorm, ...). The norm
    of a sensor, 5 operations per sample, is counted once even when both its
    mean and its std are requested. The 52 features of Lab 2 cost 7410
    operations on 50 samples, as in the theory. Raises ValueError on a name
    with an unknown prefix.
    """
    if T is None:
        from iomtlab.spec import active
        T = active().window
    per = _ops_per_feature(T)
    total, norms = 0, set()
    for n in names:
        prefix = n.split("_")[0]
        if n.endswith("norm"):                # mean_anorm, std_anorm, mean_wnorm, std_wnorm
            sensor = n.split("_")[1][0]
            if sensor not in norms:
                norms.add(sensor)
                total += 5 * T                # la norma, campione per campione
            total += per[prefix]              # media o deviazione standard della norma
            continue
        if prefix not in per:
            raise ValueError(f"feature sconosciuta: {n}")
        total += per[prefix]
    return int(total)


def _forest(n_trees, max_depth, seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                  random_state=seed, n_jobs=2)


def mdi_orders(X, y, folds, n_trees: int = 50, seed: int = 0) -> list:
    """Features ordered by impurity importance (MDI), computed on the training part of each fold.

    Choosing features on the test part would leak it into the model. Returns
    one array of column indices per fold, most important first.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    out = []
    for tr, _ in folds:
        rf = _forest(n_trees, None, seed).fit(X[tr], y[tr])
        out.append(np.argsort(rf.feature_importances_)[::-1])   # dalla più importante
    return out


def evaluate_forest(X, y, names, folds, n_trees: int, max_depth=None, k=None,
                    orders=None, seed: int = 0) -> dict:
    """One forest configuration on every fold: score and cost on the node.

    n_trees trees of depth at most max_depth (None: grown to pure leaves),
    on the k most important features of each fold (None: all of them).
    folds must be a list, since it is read more than once; orders are the
    per-fold feature orders of mdi_orders, computed here if k is set and
    orders is None.

    Returns a dict with n_trees, max_depth, k, macro_f1 and f1_std (over
    folds), and the means over folds of nodes, bytes, mean_comparisons and
    feature_ops (features_cost of the features kept).
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    folds = list(folds)
    if k is not None and orders is None:
        orders = mdi_orders(X, y, folds, seed=seed)
    f1, cost, ops = [], [], []
    for i, (tr, te) in enumerate(folds):
        cols = np.arange(X.shape[1]) if k is None else np.sort(orders[i][:k])
        rf = _forest(n_trees, max_depth, seed).fit(X[tr][:, cols], y[tr])
        f1.append(macro_f1(y[te], rf.predict(X[te][:, cols])))
        cost.append(forest_cost(rf, X[te][:, cols]))
        ops.append(features_cost([names[c] for c in cols]))
    return {"n_trees": n_trees, "max_depth": max_depth, "k": len(names) if k is None else k,
            "macro_f1": float(np.mean(f1)), "f1_std": float(np.std(f1)),
            "nodes": float(np.mean([c["nodes"] for c in cost])),
            "bytes": float(np.mean([c["bytes"] for c in cost])),
            "mean_comparisons": float(np.mean([c["mean_comparisons"] for c in cost])),
            "feature_ops": float(np.mean(ops))}
