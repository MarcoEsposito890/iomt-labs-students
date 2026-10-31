"""The data of Labs 4 to 8: the windows of Lab 2, the folds of Lab 3, the network input.

Every lab from 4 on trains on the subjects of folds 1-4 of work/split.json
and tests on the five subjects of fold 0. The files are rebuilt when they are
missing, so a lab can run without the earlier ones.
"""
from __future__ import annotations

import numpy as np
import torch

from iomtlab.config import REPO_ROOT

__all__ = ["WINDOWS_FILE", "load_reference_windows", "reference_split", "train_test_masks",
           "to_input", "logits_of"]

#: The windows of the phone, 50 samples with step 25, in physical units.
WINDOWS_FILE = REPO_ROOT / "work" / "windows.npz"


def load_reference_windows(path=None, device: str = "sp") -> dict:
    """The windows of work/windows.npz, built with signal.build_windows if the file is missing.

    Returns a dict with X (W, 50, 6) float32 in physical units, y (W,)
    int64, subject (W,) strings and purity (W,). With a project dataset
    active (spec.use), the file is windows.npz in its work folder, written by
    the project loader; it is not rebuilt here.
    """
    from pathlib import Path
    from iomtlab.spec import TUG, active
    S = active()
    p = Path(path or S.work() / "windows.npz")
    if not p.exists() and S is not TUG:
        raise FileNotFoundError(f"{p} non c'è: esegui prima il loader del progetto «{S.name}» "
                                f"(projects/.../loader.py), che scrive le finestre con io.save_windows")
    if not p.exists():
        from iomtlab.io import save_windows
        from iomtlab.signal import build_windows
        p.parent.mkdir(parents=True, exist_ok=True)
        save_windows(p, build_windows(device, 50, 25))
    d = np.load(p, allow_pickle=False)
    return {"X": d["X"], "y": d["y"], "subject": d["subject"].astype(object),
            "purity": d["purity"]}


def reference_split(path=None, subjects=None) -> dict:
    """The folds of work/split.json; written with train.compare.save_reference_split if missing.

    subjects, the subject of each window, is needed only to write the file;
    by default they are read from the reference windows.
    """
    from pathlib import Path
    from iomtlab.io import load_split
    from iomtlab.spec import active
    from iomtlab.train.compare import save_reference_split
    p = Path(path or active().work() / "split.json")
    if p.exists():
        return load_split(p)
    if subjects is None:
        subjects = load_reference_windows()["subject"]
    return save_reference_split(subjects, path=p)


def train_test_masks(subject, split=None, test_fold: int = 0):
    """(train_mask, test_mask) over subject: fold test_fold in test, the other folds in training."""
    split = split or reference_split()
    subject = np.asarray(subject, dtype=object).astype(str)
    te = np.isin(subject, np.asarray(split["folds"][test_fold], dtype=str))
    return ~te, te


def to_input(X, device: str | None = None) -> torch.Tensor:
    """Windows (W, T, C) in physical units to the network input (W, C, T), float32.

    Each channel is divided by the declared full scale of the sensor
    (8 g, 1000 deg/s on the phone) and clipped to [-1, 1]: a constant known
    on the device, not a statistic of the dataset, so the node can apply it
    too. The full scale is that of the active dataset (spec.active().scale),
    for device if it has more than one ("sp" by default for TUG).
    """
    from iomtlab.spec import active
    X = np.asarray(X, dtype=np.float32)
    fs = np.asarray(active().scale(device), dtype=np.float32)
    if X.shape[-1] != len(fs):
        raise ValueError(f"le finestre hanno {X.shape[-1]} canali, il dataset attivo "
                         f"«{active().name}» ne dichiara {len(fs)}")
    flat = np.clip(X / fs, -1.0, 1.0)
    return torch.from_numpy(np.ascontiguousarray(flat.transpose(0, 2, 1)))


@torch.no_grad()
def logits_of(model, X, batch: int = 512, device: str | None = None) -> np.ndarray:
    """Logits (W, K) of model on windows X: (W, T, C) in physical units, or a tensor (W, C, T).

    device chooses the full scale for windows in physical units ("sw" for the watch of TUG).
    """
    x = X if isinstance(X, torch.Tensor) else to_input(X, device)
    model.eval()
    return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)]).numpy()
