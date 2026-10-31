"""Federated rounds (Lab 8): what a round costs on 802.15.4, and three ways to train.

  update_bytes     bytes of a model or of an update: float32, or the int8 delta
                   of fl.quantize_delta (one byte per value and a float32 scale
                   per tensor).
  round_cost       one client, one round: bytes down and up, frames, seconds
                   and joules of radio, MACs and joules of local training.
  run_fedavg       FedAvg (McMahan et al., 2017, Algorithm 1) with every client
                   at every round; history (round, macro-F1, cumulative bytes).
  run_centralised  the same network trained in one place, one point per epoch;
                   the bytes are those of the raw windows, moved once.
  run_dsgd         decentralised SGD: local epoch, then a weighted average with
                   the neighbours (Lian et al., NeurIPS 2017).
  bytes_to_reach   cumulative bytes when a history first reaches a target.

A history is a list of rows (round, macro-F1 on the test windows, cumulative
bytes); row 0 is the starting model. Bytes are payload bytes, without frame
headers: frames and headers are in round_cost, through iomtlab.radio.

Choices of the course, declared: local training is Adam 1e-3, a new optimiser
at every round, plain cross-entropy (no class weights: a client may miss a
class); the order of the batches depends only on seed, round and client. The
radio counts reception at the same energy per bit as transmission.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from iomtlab.config import ENERGY

__all__ = ["update_bytes", "round_cost", "run_fedavg", "run_centralised", "run_dsgd",
           "bytes_to_reach"]


def update_bytes(model, bits: int = 32) -> int:
    """Bytes of the state of model: 4 per value in float32; with bits=8 the int8 delta,
    one byte per value plus a 4-byte scale per tensor (fl.quantize_delta)."""
    tensors = [t for t in model.state_dict().values() if torch.is_floating_point(t)]
    n = sum(t.numel() for t in tensors)
    if bits == 32:
        return 4 * n
    if bits == 8:
        return n + 4 * len(tensors)
    raise ValueError("bits deve essere 32 oppure 8")


def round_cost(model, n_windows: int, epochs: int = 1, up_bits: int = 32, down_bits: int = 32,
               e_bit_J: float = ENERGY["e_bit_lowpower_J"], e_mac_J: float = ENERGY["e_mac_J"],
               rate_bps: float = ENERGY["rate_lowpower_bps"]) -> dict:
    """The cost of one round for one client with n_windows windows and epochs local epochs.

    Down the global model (down_bits), up the local state or its int8 delta
    (up_bits), in 802.15.4 frames (116 useful bytes, 17 of headers). Training
    costs about 3 times the forward MACs per window (forward and backward), as
    in the theory. Returns down_B, up_B, frames, air_B, radio_s, radio_J,
    train_MAC, train_J, total_J.
    """
    from iomtlab import radio
    from iomtlab.models import layer_profile
    down, up = update_bytes(model, down_bits), update_bytes(model, up_bits)
    frames = radio.frames(down) + radio.frames(up)
    air = radio.air_bytes(down) + radio.air_bytes(up)
    mac = 3 * layer_profile(model).macs * n_windows * epochs
    out = {"down_B": down, "up_B": up, "frames": frames, "air_B": air,
           "radio_s": air * 8 / rate_bps, "radio_J": air * 8 * e_bit_J,
           "train_MAC": mac, "train_J": mac * e_mac_J}
    out["total_J"] = out["radio_J"] + out["train_J"]
    return out


def _f1(model, X_te, y_te) -> float:
    from iomtlab.evaluation import macro_f1
    from iomtlab.train.data import logits_of
    return macro_f1(y_te, logits_of(model, X_te).argmax(1))


def _local_epochs(model, x, y, epochs, batch, lr, seed):
    """epochs epochs of Adam on (x, y) tensors; a new optimiser, batches in an order fixed by seed."""
    from iomtlab.train.compare import make_loader
    from iomtlab.train.loops import train_epoch
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = make_loader(x, y, batch=batch, seed=seed)
    crit = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        train_epoch(model, loader, opt, crit, torch.device("cpu"))


def _client_data(X, y, clients):
    """[(x tensor, y array)] for clients: a dict {name: indices}, a list of indices or of (X, y)."""
    from iomtlab.train.data import to_input
    items = clients.values() if isinstance(clients, dict) else clients
    out = []
    for c in items:
        if isinstance(c, tuple):
            out.append((to_input(c[0]), np.asarray(c[1])))
        else:
            out.append((to_input(X[c]), np.asarray(y[c])))
    return out


def run_fedavg(model0, X, y, clients, X_te, y_te, rounds: int = 20, epochs: int = 1,
               batch: int = 16, bits: int = 32, lr: float = 1e-3, seed: int = 0,
               aggregate=None) -> list:
    """FedAvg with all clients at every round; returns the history (round, macro-F1, bytes).

    model0 is the starting model (not changed); X, y the windows (W, T, C) in
    physical units and their labels; clients a dict {name: indices}, a list
    of index arrays, or a list of (X_k, y_k). At every round the server sends
    the global model in float32 to every client, each client trains epochs
    epochs with batches of batch, and sends back its state (bits=32) or the
    int8 delta from the global model (bits=8, fl.quantize_delta); the server
    averages with weights n_k (fl.fedavg). Bytes per round: sum over clients
    of update_bytes down (32) and up (bits). aggregate(states, n_samples) replaces
    fl.fedavg (the Lab 8 «Scrivi tu» exercise); None uses fl.fedavg.
    """
    from iomtlab.fl.aggregate import dequantize_delta, fedavg, quantize_delta
    data = _client_data(X, y, clients)
    glob = copy.deepcopy(model0)
    per_round = len(data) * (update_bytes(model0, 32) + update_bytes(model0, bits))
    hist, sent = [(0, _f1(glob, X_te, y_te), 0)], 0
    for r in range(1, rounds + 1):
        g_state = {k: v.clone() for k, v in glob.state_dict().items()}
        states, ns = [], []
        for k, (xk, yk) in enumerate(data):
            m = copy.deepcopy(glob)                          # giù: il modello globale
            _local_epochs(m, xk, yk, epochs, batch, lr, seed + 1000 * r + k)
            s = m.state_dict()
            if bits == 8:                                    # su: la differenza in int8
                s = dequantize_delta(quantize_delta(s, g_state, 8), g_state)
            states.append(s)
            ns.append(len(yk))
        glob.load_state_dict((aggregate or fedavg)(states, ns))   # la media pesata con n_k
        sent += per_round
        hist.append((r, _f1(glob, X_te, y_te), sent))
    return hist


def run_centralised(model0, X, y, X_te, y_te, epochs: int = 20, raw_bytes: int = 0,
                    seed: int = 0) -> list:
    """The network trained in one place with the course recipe, one row per epoch.

    Batches of 64 in an order fixed by seed, Adam 1e-3 kept across epochs,
    cross-entropy weighted by inverse class frequency (train.RECIPE). Every
    row carries raw_bytes: the windows are moved once, before training.
    """
    from iomtlab.train.compare import BATCH, LR, make_loader
    from iomtlab.train.data import to_input
    from iomtlab.train.loops import inverse_freq_weights, set_seed, train_epoch
    m = copy.deepcopy(model0)
    set_seed(seed)
    loader = make_loader(to_input(X), y, batch=BATCH, seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    crit = torch.nn.CrossEntropyLoss(weight=inverse_freq_weights(np.asarray(y)))
    hist = [(0, _f1(m, X_te, y_te), raw_bytes)]
    for e in range(1, epochs + 1):
        train_epoch(m, loader, opt, crit, torch.device("cpu"))
        hist.append((e, _f1(m, X_te, y_te), raw_bytes))
    return hist


def run_dsgd(model0, X, y, clients, W, X_te, y_te, rounds: int = 20, epochs: int = 1,
             batch: int = 16, lr: float = 1e-3, seed: int = 0) -> list:
    """Decentralised SGD on a graph: each node trains locally, then averages with its neighbours.

    W is the (n, n) mixing matrix (fl.metropolis_weights); clients as in
    run_fedavg, one per node. A round: epochs local epochs on every node, then
    θ_i <- sum_j W_ij θ_j. Each node broadcasts its model once per round in
    float32: n x update_bytes bytes. The macro-F1 of a row is the mean over
    the nodes of the macro-F1 of each node's model.
    """
    data = _client_data(X, y, clients)
    W = np.asarray(W, dtype=np.float64)
    if W.shape != (len(data), len(data)):
        raise ValueError(f"W è {W.shape}, i nodi sono {len(data)}")
    models = [copy.deepcopy(model0) for _ in data]
    per_round = len(data) * update_bytes(model0, 32)

    def score():
        return float(np.mean([_f1(m, X_te, y_te) for m in models]))

    hist, sent = [(0, score(), 0)], 0
    for r in range(1, rounds + 1):
        for k, (xk, yk) in enumerate(data):
            _local_epochs(models[k], xk, yk, epochs, batch, lr, seed + 1000 * r + k)
        states = [{k: v.clone() for k, v in m.state_dict().items()} for m in models]
        for i, m in enumerate(models):                       # la media con i vicini
            new = {}
            for key, ref in states[i].items():
                if not torch.is_floating_point(ref):
                    new[key] = ref
                    continue
                acc = torch.zeros_like(ref, dtype=torch.float64)
                for j in np.nonzero(W[i])[0]:
                    acc += float(W[i, j]) * states[j][key].to(torch.float64)
                new[key] = acc.to(ref.dtype)
            m.load_state_dict(new)
        sent += per_round
        hist.append((r, score(), sent))
    return hist


def bytes_to_reach(history, target: float):
    """Cumulative bytes at the first row of history with macro-F1 >= target; None if never."""
    for r, f, b in history:
        if f >= target:
            return b
    return None
