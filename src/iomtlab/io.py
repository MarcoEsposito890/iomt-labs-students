"""Files passed from one lab to the next.

A later lab loads what an earlier one saved instead of recomputing it, so
two labs use the same features and the same subject split. Four save/load
pairs:

    features   feature matrix, labels, names, subject of each row (.npz)   Lab 2 -> 3
    windows    raw windows, labels, subject, purity (.npz)                Lab 2 -> 4-8
    split      subject folds and their seed (.json)                       Lab 3 -> 4
    model      weights, cost profile and results (.pt)                    Lab 4 -> 5, 6
    curve      rows of a measured curve and two energy constants (.json)  Lab 6 -> 7

Every save_* returns the path written; every load_* raises ValueError when
the content is inconsistent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from iomtlab.config import N_CLASSES

__all__ = [
    "save_features", "load_features", "save_windows",
    "save_split", "load_split",
    "save_model", "load_model",
    "save_curve", "load_curve", "fold_masks",
]


def _p(path) -> Path:
    """path as a Path, with its parent folder created."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ══════════════════════════════════════════════════ features
def save_features(path, X: np.ndarray, y: np.ndarray, names, subject) -> Path:
    """Save a feature matrix to path (.npz) and return the path.

    X is (n, d), stored as float32; y (n,) class indices, stored as int64;
    names d strings; subject (n,) ids such as "s01". Raises ValueError on
    inconsistent shapes.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    names = np.asarray(names, dtype=object)
    subject = np.asarray(subject, dtype=object)
    if X.ndim != 2:
        raise ValueError(f"X deve essere (n, d), non {X.shape}")
    if not (len(X) == len(y) == len(subject)):
        raise ValueError(
            f"lunghezze diverse: X {len(X)}, y {len(y)}, subject {len(subject)}")
    if len(names) != X.shape[1]:
        raise ValueError(f"{len(names)} nomi per {X.shape[1]} colonne")
    p = _p(path).with_suffix(".npz")
    np.savez_compressed(p, X=X, y=y, names=names, subject=subject)   # un solo file con i quattro array
    return p


def save_windows(path, W: dict) -> Path:
    """Save the windows of signal.build_windows to path (.npz) and return the path.

    Keeps X (W, T, C) float32 in physical units, y int64, subject as strings
    and purity: the file train.load_reference_windows reads (work/windows.npz).
    Raises ValueError on inconsistent lengths.
    """
    X = np.asarray(W["X"], dtype=np.float32)
    y = np.asarray(W["y"], dtype=np.int64)
    sub = np.asarray(W["subject"]).astype(str)
    pur = np.asarray(W["purity"], dtype=np.float32)
    if X.ndim != 3 or not (len(X) == len(y) == len(sub) == len(pur)):
        raise ValueError(f"forme incoerenti: X {X.shape}, y {y.shape}, subject {sub.shape}, "
                         f"purity {pur.shape}")
    p = _p(path).with_suffix(".npz")
    np.savez_compressed(p, X=X, y=y, subject=sub, purity=pur)
    return p


def load_features(path) -> dict:
    """Load a file written by save_features.

    Returns a dict with X, y, names (list) and subject. Raises ValueError if
    a key is missing, the shapes disagree, or a label is out of range.
    """
    # allow_pickle serve perché names e subject sono array di oggetti, stringhe Python
    d = np.load(Path(path).with_suffix(".npz"), allow_pickle=True)
    for k in ("X", "y", "names", "subject"):
        if k not in d:
            raise ValueError(f"{path}: manca '{k}', non è un file scritto da save_features")
    X, y = d["X"], d["y"]
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError(f"{path}: forme incoerenti, X {X.shape}, y {y.shape}")
    from iomtlab.spec import active
    K = active().n_classes
    if y.min() < 0 or y.max() >= K:             # controllo che le etichette siano indici di classe
        raise ValueError(f"{path}: etichette fuori da [0, {K})")
    return {"X": X, "y": y, "names": list(d["names"]),
            "subject": np.asarray(d["subject"], dtype=object)}


# ══════════════════════════════════════════════════ split
def save_split(path, folds, subjects, seed: int) -> Path:
    """Save subject folds to path (.json) and return the path.

    folds is a list of lists of subject ids, one per fold. Subjects are saved
    instead of row indices because row indices depend on the feature file,
    and a later lab may use other windows. Raises ValueError unless the folds
    partition subjects.
    """
    folds = [list(map(str, f)) for f in folds]
    subs = sorted(map(str, subjects))
    seen = [s for f in folds for s in f]
    if sorted(seen) != subs:                 # ogni soggetto deve stare in una e una sola piega
        raise ValueError(
            "le pieghe non partizionano i soggetti: "
            f"{len(seen)} assegnazioni per {len(subs)} soggetti")
    p = _p(path).with_suffix(".json")
    p.write_text(json.dumps({"folds": folds, "subjects": subs, "seed": seed},
                            indent=1))
    return p


def load_split(path) -> dict:
    """Load a file written by save_split: a dict with folds, subjects, seed.

    Raises ValueError if a subject appears in two folds.
    """
    d = json.loads(Path(path).with_suffix(".json").read_text())
    folds = d["folds"]
    flat = [s for f in folds for s in f]
    if len(flat) != len(set(flat)):          # se ci sono doppioni, un soggetto sta in due pieghe
        raise ValueError(f"{path}: un soggetto compare in due pieghe")
    return d


def fold_masks(split: dict, subject: np.ndarray):
    """Boolean masks over an array of subject ids, from a split loaded by load_split.

    Yields (train_mask, test_mask), as subject_folds does.
    """
    subject = np.asarray(subject, dtype=object).astype(str)
    for part in split["folds"]:
        test = np.isin(subject, np.asarray(part, dtype=str))   # in prova le righe dei soggetti della piega
        yield ~test, test


# ══════════════════════════════════════════════════ model
def save_model(path, model, cost=None, results=None, note: str = "") -> Path:
    """Save weights, cost profile and results in one file (.pt) and return the path.

    Kept together, the profile always describes the saved weights. cost is a
    models.ModelCost or a dict; numpy arrays in results are stored as lists.
    """
    import torch
    p = _p(path).with_suffix(".pt")
    payload = {
        "state_dict": model.state_dict(),
        "class_name": type(model).__name__,          # lo ricontrollo in load_model
        "cost": cost.to_dict() if hasattr(cost, "to_dict") else cost,
        "results": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in (results or {}).items()},
        "note": note,
    }
    torch.save(payload, p)
    return p


def load_model(path, model=None):
    """Load a file written by save_model and return its dict.

    If model is given, the weights are loaded into it and it is returned
    under "model". Raises ValueError if the file holds another model class.
    """
    import torch
    p = Path(path).with_suffix(".pt")
    # weights_only=False perché nel file ci sono anche dizionari e stringhe, non solo tensori
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if model is not None:
        want = type(model).__name__              # prima di caricare i pesi controllo la classe
        got = payload.get("class_name")
        if got and got != want:
            raise ValueError(f"{p.name} contiene un {got}; il modello passato è un {want}")
        model.load_state_dict(payload["state_dict"])
        payload["model"] = model
    return payload


# ══════════════════════════════════════════════════ curve
def save_curve(path, rows, e_head_J: float, e_link_J: float,
               columns=("threshold", "q", "energy_J", "accuracy")) -> Path:
    """Save a curve to path (.json) and return the path.

    rows have one value per entry of columns, by default (threshold, q,
    energy_J, accuracy). The two energy constants e_head_J and e_link_J (J)
    are saved with them, so a later lab can redo the computation for another
    q. Raises ValueError if a row has the wrong length.
    """
    rows = [list(map(float, r)) for r in rows]
    if any(len(r) != len(columns) for r in rows):
        raise ValueError(f"ogni riga deve avere {len(columns)} valori")
    p = _p(path).with_suffix(".json")
    p.write_text(json.dumps({"columns": list(columns), "rows": rows,
                             "e_head_J": float(e_head_J),
                             "e_link_J": float(e_link_J)}, indent=1))
    return p


def load_curve(path) -> dict:
    """Load a file written by save_curve as a dict. Raises ValueError if a key is missing."""
    d = json.loads(Path(path).with_suffix(".json").read_text())
    for k in ("columns", "rows", "e_head_J", "e_link_J"):
        if k not in d:
            raise ValueError(f"{path}: manca '{k}', non è un file scritto da save_curve")
    return d
