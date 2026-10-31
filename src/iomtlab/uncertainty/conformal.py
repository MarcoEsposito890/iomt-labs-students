"""Split conformal prediction and the coverage it guarantees.

Angelopoulos and Bates, "A Gentle Introduction to Conformal Prediction and
Distribution-Free Uncertainty Quantification", 2021, section 1. Two details
matter for the finite-sample guarantee:

  - the +1 in ceil((n+1)(1-alpha)); without it the guarantee holds only
    asymptotically (with n = 9 and alpha = 0.1 the difference is visible);
  - the quantile is the k-th smallest score, index k-1 of the sorted
    scores. np.quantile with its default interpolation gives another value.

Probabilities are (n, K) arrays, labels integer class indices.
"""
from __future__ import annotations

import numpy as np

from iomtlab.config import LABEL_NAMES, N_CLASSES

__all__ = ["conformal_quantile", "conformal_sets", "coverage",
           "coverage_by_class", "singleton_rate"]


def conformal_quantile(scores, alpha: float = 0.1) -> float:
    """The conformal quantile: the ceil((n+1)(1-alpha))-th smallest score, as a float.

    Returns inf if that index exceeds n (alpha too small for n calibration
    points). Raises ValueError on an empty score array.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))   # ordino i punteggi di calibrazione
    n = len(s)
    if n == 0:
        raise ValueError("nessun punteggio di calibrazione")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))   # l'indice con il +1 che dà la garanzia a campione finito
    if k > n:
        # alpha è troppo piccolo per questo n: solo l'insieme con tutte le classi garantisce la copertura
        return float("inf")
    return float(s[k - 1])                   # il k-esimo punteggio sta all'indice k-1


def conformal_sets(p_cal, y_cal, p_test, alpha: float = 0.1, score_fn=None):
    """Prediction sets with marginal coverage >= 1 - alpha, as a boolean (n_test, K) matrix.

    Non-conformity score: 1 - p(true class) on the calibration data. A test
    set holds the classes whose score does not exceed the conformal
    quantile. A set can be empty; it is left empty, and what to do with it
    is up to the caller.

    score_fn(p, y) -> (n,) replaces the default score 1 - p[y] (a project
    can plug its own, e.g. one that treats a rare class differently). It is
    called on the calibration data with their labels, and on the test data
    once per class k with y = k for every row; the guarantee holds for any
    score computed this way.
    """
    p_cal = np.asarray(p_cal, dtype=np.float64)
    p_test = np.asarray(p_test, dtype=np.float64)
    y_cal = np.asarray(y_cal, dtype=np.int64)
    if score_fn is None:
        # punteggio di non conformità: 1 meno la probabilità della classe vera
        s = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
        qh = conformal_quantile(s, alpha)        # la soglia stimata sulla calibrazione
        # in ogni insieme metto le classi il cui punteggio non supera la soglia
        return (1.0 - p_test) <= qh
    qh = conformal_quantile(np.asarray(score_fn(p_cal, y_cal), dtype=np.float64), alpha)
    n, K = p_test.shape
    sc = np.stack([np.asarray(score_fn(p_test, np.full(n, k, dtype=np.int64)), dtype=np.float64)
                   for k in range(K)], axis=1)   # il punteggio di ogni classe candidata
    return sc <= qh


def coverage(sets, y_true) -> float:
    """Fraction of sets that contain the true label, as a float."""
    sets = np.asarray(sets, dtype=bool)
    y_true = np.asarray(y_true, dtype=np.int64)
    # per ogni riga guardo se la colonna della classe vera è accesa
    return float(sets[np.arange(len(y_true)), y_true].mean())


def coverage_by_class(sets, y_true, names=None) -> dict:
    """Marginal and per-class coverage, mean set size and fraction of empty sets, as a dict.

    The guarantee is marginal: it can hold while a transient class is well
    below 1 - alpha. A class absent from y_true gets NaN.
    """
    from iomtlab.spec import active
    names = list(names or active().classes)
    sets = np.asarray(sets, dtype=bool)
    y_true = np.asarray(y_true, dtype=np.int64)
    hit = sets[np.arange(len(y_true)), y_true]   # vero dove l'insieme contiene la classe vera
    per = {}
    for k in range(len(names)):
        m = y_true == k                      # i campioni della classe k
        per[names[k]] = float(hit[m].mean()) if m.any() else float("nan")
    return {"marginal": float(hit.mean()),
            "by_class": per,
            "mean_size": float(sets.sum(1).mean()),
            "empty_fraction": float((sets.sum(1) == 0).mean())}


def singleton_rate(sets) -> float:
    """q: the fraction of sets with exactly one class, as a float.

    These are the windows the node can decide on its own, with the coverage
    of the sets behind them.
    """
    sets = np.asarray(sets, dtype=bool)
    # conto le classi in ogni insieme e tengo quelli con una sola
    return float((sets.sum(1) == 1).mean())
