"""Measurements on a model: bytes on disk, time per window, non-zero weights.

None of them is estimated from the parameter count: the labs compare them
with the estimates of models.cost.
"""
from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

__all__ = ["size_on_disk", "time_per_window", "nonzero_params"]


def size_on_disk(model: nn.Module) -> int:
    """Size in bytes of the file torch.save writes for the model's state_dict.

    A dynamically quantised model is about a quarter of the float one, not
    exactly: some layers stay in floating point and the archive header does
    not scale.
    """
    # salvo lo state_dict in un file temporaneo e ne leggo la dimensione
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save(model.state_dict(), tmp)
        return tmp.stat().st_size
    finally:
        tmp.unlink(missing_ok=True)          # il file lo cancello in ogni caso


@torch.no_grad()
def time_per_window(model: nn.Module, x: torch.Tensor, n: int = 200,
                    warmup: int = 20, repeats: int = 5) -> dict:
    """Seconds per inference on one window, as a dict.

    x is one window, (C, T) or (1, C, T). warmup calls run first (the first
    ones pay for allocation and caches), then n calls are timed, repeats
    times. Returns median_s, min_s, max_s (seconds per call over the
    repetitions), spread ((max - min) / median), n and repeats. Raises
    ValueError on a batch larger than one: on the node windows arrive one at
    a time.

    The value does not transfer to a node. On a laptop it depends on the CPU
    frequency: compare models on the same machine, minutes apart at most.
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)                   # aggiungo la dimensione del lotto
    if x.shape[0] != 1:                      # accetto una finestra per volta: un lotto nasconde il costo per finestra
        raise ValueError(f"serve una finestra sola, x ha un lotto di {x.shape[0]}")
    model.eval()
    for _ in range(warmup):                  # chiamate di riscaldamento, non le misuro
        model(x)
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
        runs.append((time.perf_counter() - t0) / n)   # tempo medio per chiamata in questa ripetizione
    # riporto la mediana delle ripetizioni: una singola interruzione del sistema operativo non la sposta
    return {"median_s": statistics.median(runs),
            "min_s": min(runs), "max_s": max(runs),
            "spread": (max(runs) - min(runs)) / max(statistics.median(runs), 1e-12),
            "n": n, "repeats": repeats}


def nonzero_params(model: nn.Module) -> dict:
    """Total and non-zero parameters, as a dict.

    Unstructured pruning sets weights to zero without removing them, so the
    non-zero count drops while the file size stays the same. If pruning is
    still attached (weight_orig and weight_mask present), model.parameters()
    returns the unmasked originals; then orig * mask is counted instead and
    pruning_active is True. Returns total, nonzero, zero_fraction and
    pruning_active.
    """
    total = nz = 0
    active = False
    # controllo se c'è un pruning ancora attaccato a qualche modulo
    active = any(hasattr(mod, "weight_mask") for mod in model.modules())
    if not active:
        for p in model.parameters():
            total += p.numel()
            nz += int(torch.count_nonzero(p).item())      # conto i pesi diversi da zero
    else:
        # con il pruning attaccato conto i pesi effettivi, cioè orig * mask
        for mod in model.modules():
            if hasattr(mod, "weight_mask"):
                w = mod.weight_orig * mod.weight_mask
                total += w.numel()
                nz += int(torch.count_nonzero(w).item())
            else:
                for p in mod.parameters(recurse=False):
                    total += p.numel()
                    nz += int(torch.count_nonzero(p).item())
    return {"total": total, "nonzero": nz,
            "zero_fraction": 1.0 - nz / max(total, 1),
            "pruning_active": active}
