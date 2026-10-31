"""Splits, metrics and tables shared by the labs.

Split functions yield boolean (train_mask, test_mask) pairs over the rows,
built from the subject id of each row (groups).
"""
from __future__ import annotations

import numpy as np

from iomtlab.config import LABEL_NAMES, N_CLASSES

__all__ = [
    "loso_folds", "incremental_loso_subsets",
    "subject_folds", "kfold_shuffled",
    "macro_f1", "confusion_report", "top_k_table", "accuracy_with_exit",
    "per_subject_scores",
]


# ══════════════════════════════════════════════════════════ splits
def loso_folds(groups):
    """Leave-One-Subject-Out: one fold per subject, subjects in sorted order.

    The source system was evaluated this way (Matey-Sanz, Casteleyn, Granell;
    libs/chapter3/pipeline/03_incremental-loso.py in the thesis repository),
    so LOSO results compare with the published ones. No seed; one score per
    subject; one training per subject. For networks, subject_folds is the
    cheaper approximation.

    Yields boolean (train_mask, test_mask) pairs over groups.
    """
    groups = np.asarray(groups, dtype=object).astype(str)   # "s01" e 1 devono confrontarsi allo stesso modo
    for s in np.unique(groups):
        test = groups == s                   # in prova metto tutte le righe del soggetto s
        yield ~test, test                    # e addestro su tutte le altre


def subject_folds(groups, k: int = 5, seed: int = 0):
    """k folds over subjects, never splitting a subject: the cheap version of LOSO.

    Every subject is tested exactly once. Fold sizes are not balanced, since
    balancing would split a subject. The spread across folds is between
    groups of subjects, narrower than the spread across subjects, and the
    result depends on seed.

    Yields boolean (train_mask, test_mask) pairs over groups. Raises
    ValueError if k exceeds the number of subjects.
    """
    groups = np.asarray(groups, dtype=object).astype(str)
    subs = np.unique(groups)
    if k > len(subs):
        raise ValueError(f"{k} pieghe per {len(subs)} soggetti: impossibile")
    rng = np.random.default_rng(seed)
    order = subs.copy()
    rng.shuffle(order)                       # mescolo i soggetti, non le righe
    for part in np.array_split(order, k):    # e li divido in k gruppi
        test = np.isin(groups, part)         # in prova vanno tutte le righe dei soggetti del gruppo
        yield ~test, test


def incremental_loso_subsets(groups, test_subject, sizes=None, repeats: int = 3,
                             seed: int = 0):
    """Incremental LOSO: how many training subjects are needed.

    test_subject is held out; the training set is n of the other subjects,
    drawn at random repeats times for each n in sizes (default 1, 2, 4, 8,
    all). The score against n shows whether more subjects would still help.
    The source study runs n = 1 .. 22 with 10 repetitions.

    Yields (n, repetition, train_mask, test_mask). Raises ValueError if
    test_subject is missing or is the only subject.
    """
    groups = np.asarray(groups, dtype=object).astype(str)
    test_subject = str(test_subject)
    others = [s for s in np.unique(groups) if s != test_subject]   # i candidati per l'addestramento
    if not others:
        raise ValueError(f"{test_subject} non compare in groups, oppure è il "
                         f"solo soggetto presente")
    sizes = sizes if sizes is not None else [1, 2, 4, 8, len(others)]
    sizes = [n for n in sizes if 1 <= n <= len(others)]   # tengo solo le numerosità possibili
    test = groups == test_subject            # il soggetto in prova è sempre lo stesso
    rng = np.random.default_rng(seed)
    for n in sizes:
        r = 1 if n == len(others) else repeats   # con tutti gli altri soggetti l'estrazione è una sola
        for i in range(r):
            train_subjects = rng.choice(others, size=n, replace=False)   # n soggetti senza ripetizione
            yield n, i, np.isin(groups, train_subjects), test


def kfold_shuffled(n: int, k: int = 5, seed: int = 0):
    """Random k-fold over n rows, ignoring the subject: only a term of comparison.

    With overlapping windows the same subject, and nearly the same window,
    end up in both train and test, so the score is optimistic. Yields
    boolean (train_mask, test_mask) pairs.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)                 # qui mescolo le righe, senza guardare i soggetti
    for part in np.array_split(idx, k):
        test = np.zeros(n, dtype=bool)
        test[part] = True
        yield ~test, test


# ══════════════════════════════════════════════════════════ metrics
def macro_f1(y_true, y_pred, K: int | None = None) -> float:
    """Mean F1 over all K classes, as a float.

    A class absent from both y_true and y_pred counts as F1 = 0 and lowers
    the mean: with this convention the value is comparable across
    partitions. Same as sklearn f1_score(labels=range(K), average="macro",
    zero_division=0). K defaults to the classes of the active dataset.
    """
    if K is None:
        from iomtlab.spec import active
        K = active().n_classes
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    C = np.zeros((K, K), dtype=np.int64)
    np.add.at(C, (y_true, y_pred), 1)        # matrice di confusione: righe vere, colonne predette
    tp = np.diag(C).astype(np.float64)       # i veri positivi stanno sulla diagonale
    prec = tp / np.maximum(C.sum(0), 1)      # precisione: divido per il totale di colonna
    rec = tp / np.maximum(C.sum(1), 1)       # richiamo: divido per il totale di riga
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(f1.mean())                  # media semplice su tutte le K classi


def confusion_report(y_true, y_pred, names=None, normalize: str | None = None):
    """Confusion matrix and per-class scores.

    normalize is None, "row" (rows sum to 1: recall per class) or "col"
    (columns sum to 1: precision). Returns a dict with matrix (counts),
    normalized, names, precision, recall, f1, macro_f1, accuracy, normalize.
    Raises ValueError on any other normalize.
    """
    if names is None:
        from iomtlab.spec import active
        names = active().classes
    names = list(names)
    K = len(names)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    C = np.zeros((K, K), dtype=np.int64)
    np.add.at(C, (y_true, y_pred), 1)        # conto le coppie (vera, predetta)
    M = C.astype(np.float64)
    if normalize == "row":
        # per riga: quota di ogni classe vera finita in ciascuna colonna
        M = M / np.maximum(C.sum(1, keepdims=True), 1)
    elif normalize == "col":
        # per colonna: da quali classi vere vengono le predizioni
        M = M / np.maximum(C.sum(0, keepdims=True), 1)
    elif normalize is not None:
        raise ValueError("normalize deve essere None, 'row' o 'col'")
    tp = np.diag(C).astype(np.float64)
    prec = tp / np.maximum(C.sum(0), 1)      # stesse formule di macro_f1
    rec = tp / np.maximum(C.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return {"matrix": C, "normalized": M, "names": names,
            "precision": prec, "recall": rec, "f1": f1,
            "macro_f1": float(f1.mean()),
            "accuracy": float(tp.sum() / max(C.sum(), 1)),
            "normalize": normalize}


def top_k_table(importance, names, k: int = 5):
    """The k most important features as [index, name, importance], largest first."""
    importance = np.asarray(importance, dtype=np.float64)
    order = np.argsort(importance)[::-1][:k]   # ordino in modo decrescente e tengo le prime k
    return [[int(i), str(names[i]), float(importance[i])] for i in order]


# ══════════════════════════════════════════════════════════ early exit
def accuracy_with_exit(z_exit, z_full, take_exit, y) -> float:
    """Accuracy of a system with an early exit, as a float.

    z_exit and z_full are the (N, K) logits of the exit and of the full
    model; where take_exit is true the exit decides, elsewhere the full
    model. Report it next to the saving: the saving alone is maximised by
    always taking the exit.
    """
    z_exit = np.asarray(z_exit)
    z_full = np.asarray(z_full)
    take = np.asarray(take_exit, dtype=bool)
    y = np.asarray(y, dtype=np.int64)
    # dove accetto l'uscita prendo la sua classe, altrove quella del modello intero
    pred = np.where(take, z_exit.argmax(1), z_full.argmax(1))
    return float((pred == y).mean())


def per_subject_scores(groups, y_true, y_pred, metric=None) -> "pd.DataFrame":
    """The metric computed separately for each subject.

    metric(y_true, y_pred) defaults to macro_f1. Returns a DataFrame with
    subject, n (rows) and score, sorted by subject.
    """
    import pandas as pd
    metric = metric or macro_f1
    groups = np.asarray(groups, dtype=object).astype(str)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for s in np.unique(groups):
        m = groups == s                      # la metrica la calcolo solo sulle righe del soggetto s
        rows.append({"subject": s, "n": int(m.sum()),
                     "score": float(metric(y_true[m], y_pred[m]))})
    return pd.DataFrame(rows)                # np.unique ha già ordinato i soggetti
