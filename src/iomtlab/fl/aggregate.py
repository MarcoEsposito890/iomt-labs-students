"""Federated aggregation, and quantised updates to send it with fewer bytes.

States are PyTorch state_dicts (name -> tensor).
"""
from __future__ import annotations

import copy

import torch


__all__ = ["fedavg", "clone_with", "quantize_delta", "dequantize_delta"]


def fedavg(states, n_samples):
    """FedAvg (McMahan et al., 2017): mean of the client states weighted by their samples.

    states is a list of state_dicts, n_samples the samples of each client.
    Integer buffers (num_batches_tracked of the normalisation layers) are not
    averaged, since a fractional weight would truncate them to zero: they
    come from the client with most samples. Returns the averaged state.
    Raises ValueError if the lengths differ or the samples sum to zero.
    """
    if len(states) != len(n_samples):
        raise ValueError(f"{len(states)} stati per {len(n_samples)} conteggi")
    tot = float(sum(n_samples))
    if tot <= 0:
        raise ValueError("somma dei campioni nulla")
    big = int(max(range(len(n_samples)), key=lambda i: n_samples[i]))   # il client con più campioni
    out = {}
    for k in states[0]:
        ref = states[0][k]
        if not torch.is_floating_point(ref):
            # i buffer interi non li medio: copio quelli del client più grande
            out[k] = states[big][k].clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float64)   # accumulo in float64 per non perdere cifre
        for s, n in zip(states, n_samples):
            # ogni client pesa quanto la sua quota di campioni
            acc += s[k].to(torch.float64) * (float(n) / tot)
        out[k] = acc.to(ref.dtype)           # torno al tipo originale del parametro
    return out


def clone_with(model, state):
    """A copy of model with the given state; the original is left unchanged."""
    m = copy.deepcopy(model)                 # copia profonda: il modello di partenza resta com'è
    m.load_state_dict(state)
    return m


def quantize_delta(local, glob, bits: int = 8):
    """Quantise the difference between the local and the global state.

    Symmetric, one scale per tensor: scale = max|delta| / (2**(bits-1) - 1).
    Deltas are small and centred, so they quantise better than the weights;
    the error adds up over rounds. An all-zero delta (buffers in the first
    round) gets scale 1 instead of 0, which would give NaN. Returns
    {name: (int8 tensor, scale)}; bits above 8 do not fit the int8 codes.
    """
    qmax = 2 ** (bits - 1) - 1               # codice massimo: 127 con 8 bit
    out = {}
    for k in glob:
        d = (local[k] - glob[k]).to(torch.float32)   # la differenza rispetto al modello globale
        m = float(d.abs().max())
        if m == 0.0:
            # differenza tutta nulla: scala 1 e codici a zero, altrimenti dividerei per zero
            out[k] = (torch.zeros_like(d, dtype=torch.int8), 1.0)
            continue
        s = m / qmax                         # scala simmetrica: il valore più grande va a qmax
        out[k] = (torch.round(d / s).clamp(-qmax - 1, qmax).to(torch.int8), s)
    return out


def dequantize_delta(packed, glob):
    """Rebuild the local state as the global state plus the dequantised delta."""
    # riporto ogni codice alla sua scala e lo sommo al parametro globale
    return {k: glob[k] + (q.to(torch.float32) * s).to(glob[k].dtype)
            for k, (q, s) in packed.items()}
