"""Training and evaluation loops, seeds and class weights.

evaluate computes the course macro-F1 with sklearn: the mean F1 over all five
classes, with F1 = 0 for a class absent from both labels and predictions
(labels passed explicitly, zero_division=0). Without labels, sklearn averages
only over the classes present and the value changes across partitions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, f1_score

from iomtlab.config import N_CLASSES


def set_seed(seed: int) -> None:
    """Seed Python's random, numpy and torch with the same seed."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def inverse_freq_weights(y: np.ndarray, n_classes: int | None = None) -> torch.Tensor:
    """Class weights inversely proportional to class frequency, mean 1, as a float32 tensor.

    n_classes defaults to the classes of the active dataset.
    """
    from iomtlab.spec import active
    n_classes = n_classes or active().n_classes
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)   # campioni per classe
    w = counts.sum() / (counts + 1e-8)       # ogni classe pesa l'inverso della sua frequenza
    # normalizzo a media 1, così la loss resta sulla scala che avrebbe senza pesi
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def effective_num_weights(y: np.ndarray, n_classes: int | None = None,
                          beta: float = 0.999) -> torch.Tensor:
    """Class weights from the effective number of samples (Cui et al., CVPR 2019).

    E_n = (1 - beta**n) / (1 - beta), weight 1 / E_n, normalised to sum to
    n_classes; returned as a float32 tensor. beta -> 0 gives uniform
    weights, beta -> 1 inverse frequency.
    """
    from iomtlab.spec import active
    n_classes = n_classes or active().n_classes
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)   # numero effettivo di campioni per classe
    w = 1.0 / np.maximum(eff, 1e-8)
    # normalizzo perché i pesi sommino al numero di classi
    return torch.tensor(w / w.sum() * n_classes, dtype=torch.float32)


def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer, criterion,
                device: torch.device) -> float:
    """One training epoch; returns the mean loss per batch as a float.

    criterion is any callable(logits, y) -> scalar tensor, for example
    nn.CrossEntropyLoss() or a distillation loss.
    """
    model.train()                            # modalità di addestramento: il dropout è attivo
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()                # azzero i gradienti del lotto precedente
        loss = criterion(model(x), y)        # passaggio in avanti e loss sul lotto
        loss.backward()                      # calcolo i gradienti
        optimizer.step()                     # e aggiorno i pesi
        running += loss.item()
    return running / max(1, len(loader))     # la loss media sui lotti dell'epoca


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device
            ) -> tuple[np.ndarray, np.ndarray]:
    """(logits, true labels) over the whole loader, as numpy arrays, without gradient."""
    model.eval()                             # modalità di valutazione: niente dropout
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())   # raccolgo i logit lotto per lotto
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Dict with accuracy, macro_f1, per_class_f1 and confusion (matrix) over the loader."""
    logits, y_true = predict(model, loader, device)
    y_pred = logits.argmax(1)                # la classe predetta è quella con il logit più alto
    from iomtlab.spec import active
    labels = list(range(active().n_classes))
    # passo labels in modo esplicito: così una classe assente conta come F1 = 0
    per_class = f1_score(y_true, y_pred, labels=labels, average=None,
                         zero_division=0)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(per_class.mean()),
        "per_class_f1": per_class,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }


def fit(model: nn.Module, loader: DataLoader, epochs: int = 15,
        weights: torch.Tensor | None = None, lr: float = 1e-3,
        loss_fn=None, device: torch.device | None = None,
        seed: int | None = None, verbose: bool = False) -> list[float]:
    """Train model for epochs epochs with Adam; returns the mean loss of each epoch.

    Everything that changes the result is an argument, so two runs differ
    only in what is passed.

    weights: class weights (inverse_freq_weights or effective_num_weights)
        for the cross-entropy.
    loss_fn: callable(logits, y) -> scalar that replaces the cross-entropy,
        for example a distillation loss.
    seed: if given, seeds everything before the optimiser is built. It does
        not cover weight initialisation, which happens when the model is
        constructed: for a reproducible comparison call set_seed before
        Model() too.
    """
    if seed is not None:
        set_seed(seed)
    device = device or torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_fn is None:
        # senza una loss data uso la cross-entropy, pesata per classe se ho i pesi
        loss_fn = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))
    history = []
    for ep in range(epochs):
        history.append(train_epoch(model, loader, opt, loss_fn, device))   # un'epoca, e ne tengo la loss media
        if verbose:
            print(f"  epoca {ep + 1}/{epochs}  loss {history[-1]:.4f}")
    return history


def first_batch_hash(loader) -> str:
    """First 16 hex digits of the SHA-1 of the first batch; "" for an empty loader.

    Two loaders with the same hash give the same first batch in the same
    order. A DataLoader rebuilt with shuffle=True changes it.
    """
    import hashlib
    for x, y in loader:
        # faccio l'hash dei byte di ingressi ed etichette del primo lotto, e mi fermo lì
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(x.detach().cpu().numpy()).tobytes())
        h.update(np.ascontiguousarray(y.detach().cpu().numpy()).tobytes())
        return h.hexdigest()[:16]
    return ""
