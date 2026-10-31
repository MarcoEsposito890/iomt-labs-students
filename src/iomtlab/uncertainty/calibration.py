"""Calibration: how far the confidence the model reports can be trusted."""
from __future__ import annotations

import numpy as np


__all__ = ["calibration_split", "ece", "fit_temperature", "apply_temperature"]


def calibration_split(subjects, seed: int = 0,
                      fractions=(0.6, 0.2, 0.2)) -> dict:
    """Three disjoint groups of subjects: train, calibration, test.

    If calibration sees training subjects, the temperature is fitted on
    over-confident scores and the reported ECE is too low. fractions are the
    shares of subjects; each group gets at least one. Returns a dict with
    train, cal, test (lists of ids) and seed. Raises ValueError if fractions
    do not sum to 1 or there are fewer than 3 subjects.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"le frazioni devono sommare a 1, non {sum(fractions)}")
    subs = np.unique(np.asarray(subjects, dtype=object).astype(str))
    if len(subs) < 3:
        raise ValueError(f"servono almeno 3 soggetti, ce ne sono {len(subs)}")
    rng = np.random.default_rng(seed)
    order = subs.copy()
    rng.shuffle(order)                       # mescolo i soggetti
    n = len(order)
    a = max(1, int(round(fractions[0] * n)))           # fine del gruppo di addestramento
    b = max(a + 1, a + int(round(fractions[1] * n)))   # fine della calibrazione, almeno un soggetto
    b = min(b, n - 1)                        # e lascio almeno un soggetto per la prova
    return {"train": list(order[:a]), "cal": list(order[a:b]),
            "test": list(order[b:]), "seed": seed}


def ece(conf, correct, n_bins: int = 15) -> float:
    """Expected calibration error (Guo et al., ICML 2017) with equal-width bins, as a float.

    conf is the confidence of each prediction, correct 1 where it was right.
    Each bin weighs by its fraction of samples: uniform weights would let the
    nearly empty high-confidence bins dominate. The value depends on n_bins,
    so report it with the number of bins.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)   # bin di uguale larghezza fra 0 e 1
    idx = np.digitize(conf, edges[1:-1])     # indice del bin di ogni campione
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue                         # i bin vuoti non contribuiscono
        # scarto fra accuratezza e confidenza media nel bin, pesato con la quota di campioni del bin
        out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def fit_temperature(logits, y, max_iter: int = 50) -> float:
    """Temperature scaling (Guo et al., ICML 2017): the T that divides the logits, as a float.

    Fitted on calibration logits and labels by minimising the negative log
    likelihood with LBFGS. Dividing all logits by the same T leaves the
    argmax, and so the accuracy, unchanged; only the confidence moves.
    """
    import torch
    import torch.nn.functional as F
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    t = torch.as_tensor(np.asarray(y), dtype=torch.long)
    # ottimizzo log T, così T = exp(logT) resta positiva; parto da T = 1
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        # la cross-entropy dei logit divisi per T è la log-verosimiglianza negativa da minimizzare
        loss = F.cross_entropy(z / torch.exp(logT), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(logT).item())


def apply_temperature(logits, T: float):
    """Softmax probabilities of logits / T, as an (N, K) numpy array."""
    z = np.asarray(logits, dtype=np.float64) / float(T)
    # tolgo il massimo di ogni riga: la softmax non cambia e l'esponenziale non va in overflow
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
