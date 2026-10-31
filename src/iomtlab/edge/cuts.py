"""The cuts of a network in Lab 6: what crosses each boundary, how long it takes, the real split.

  cut_profile  one row per cut of a model that declares stages(), plus the two
               ends: «finestra» (send the raw window, everything on the server)
               and «classe» (send the class, everything on the node).
  cut_delay    the delay at each cut: head on the node, link, tail on the server.
  run_split    the cut done for real, window by window: head, pack_int8, bytes,
               unpack_int8, tail.
  train_exit   a linear exit on the activation at a cut, head frozen.
  exit_curve   threshold against share decided on the node, energy, accuracy.

The rows of cut_profile go to edge.cut_cost (energy) and then to cut_delay
(time), as in the notebook:

    rows = cut_delay(cut_cost(cut_profile(model, act_bits=8), e_mac, e_bit, rate),
                     40e6, 2.4e9, rate)

Costs are computed from declared constants (config.ENERGY, block 5.1), not
measured on a node.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from iomtlab.config import REPO_ROOT

__all__ = ["CURVE_FILE", "cut_profile", "cut_delay", "run_split", "train_exit", "exit_curve", "code_entropy"]

#: Where Lab 6 writes the exit curve that Lab 7 reads (TUG; a project: active().work()).
CURVE_FILE = REPO_ROOT / "work" / "curve.json"

def cut_profile(model: nn.Module, act_bits: int = 8, window_bits: int | None = None,
                in_channels: int | None = None, seq_len: int | None = None,
                n_classes: int | None = None, raw: str = "window") -> list[dict]:
    """One row per cut: name, values, bits, mac_head, mac_tail; first the raw end, last «classe».

    The internal cuts are the boundaries after each stage of model.stages()
    except the last (cutting after the output is sending the class). Values
    are those of the activation for one window, bits = values * act_bits.
    The window is in_channels x seq_len samples at window_bits each; the class
    takes ceil(log2(n_classes)) bits, 3 for five classes. mac_head counts the
    MACs of the stages before the cut (models.cost.module_macs, as layer_profile),
    mac_tail the rest: their sum is the same at every cut. The defaults
    (16 bits, 6 x 50, 5 classes) come from the active dataset (spec.active()).

    raw says what the raw end sends per decision. "window" (default, the
    theory): the whole window, row «finestra». "stream": only the samples
    that are new since the last decision, step x channels x window_bits, row
    «flusso»; with windows that overlap the server keeps the rest. The other
    rows stay per window: a head that runs window by window sends its whole
    activation at every decision (a convolutional head computed incrementally
    could send only the new columns; that is not modelled).
    """
    from iomtlab.spec import active
    S = active()
    if raw not in ("window", "stream"):
        raise ValueError("raw deve essere 'window' o 'stream'")
    window_bits = window_bits or S.sample_bits
    in_channels, seq_len = in_channels or S.n_channels, seq_len or S.window
    n_classes = n_classes or S.n_classes
    if not hasattr(model, "stages"):
        raise NotImplementedError(f"{type(model).__name__} non ha il metodo stages()")
    from iomtlab.models.cost import counted_modules, module_macs
    count = {"macs": 0}

    def hook(mod, inp, out):
        count["macs"] += module_macs(mod, inp, out)[0]   # sommo i MAC di ogni strato che gira

    handles = [m.register_forward_hook(hook) for _, m in counted_modules(model)]
    stages = list(model.stages())
    inner = []
    model.eval()
    try:
        with torch.no_grad():
            x = torch.zeros(1, in_channels, seq_len)
            for name, f in stages:
                x = f(x)                            # una fase alla volta, sulla finestra di zeri
                inner.append((name, int(x[0].numel()), count["macs"]))
    finally:
        for h in handles:
            h.remove()
    total = count["macs"]
    w_vals = in_channels * seq_len
    if raw == "stream":                             # solo i campioni nuovi dall'ultima decisione
        w_vals = in_channels * min(S.step, seq_len)
    rows = [{"name": "finestra" if raw == "window" else "flusso", "values": w_vals,
             "bits": w_vals * window_bits, "mac_head": 0, "mac_tail": total}]
    for name, vals, cum in inner[:-1]:              # l'ultima fase è l'uscita: lì si manda la classe
        rows.append({"name": name, "values": vals, "bits": vals * act_bits,
                     "mac_head": cum, "mac_tail": total - cum})
    rows.append({"name": "classe", "values": 1, "bits": int(math.ceil(math.log2(n_classes))),
                 "mac_head": total, "mac_tail": 0})
    return rows


def cut_delay(rows, node_macs_s: float, server_macs_s: float, rate_bps: float,
              hops: int = 1) -> list[dict]:
    """The rows with the delay of each cut: t_node_s, t_link_s, t_server_s, t_tot_s.

    t = mac_head / node_macs_s + hops * bits / rate_bps + mac_tail / server_macs_s:
    the head on the node, the activation on the link (h hops one after the
    other), the tail on the server. Connection, acknowledgements,
    retransmissions and queues are left out, as in the theory. hops must be
    the same as in edge.cut_cost.
    """
    out = []
    for r in rows:
        t_node = r["mac_head"] / node_macs_s
        t_link = hops * r["bits"] / rate_bps
        t_srv = r["mac_tail"] / server_macs_s
        out.append({**r, "t_node_s": t_node, "t_link_s": t_link, "t_server_s": t_srv,
                    "t_tot_s": t_node + t_link + t_srv})
    return out


@torch.no_grad()
def run_split(model: nn.Module, at: str, X) -> dict:
    """The cut at stage at done for real on windows X, one at a time.

    For each window: the head on the node, pack_int8 of its output (the bytes
    that go on the link), unpack_int8 on the server, the tail. X are windows
    (W, T, C) in physical units or a tensor (W, C, T). Returns a dict with
    packets (one bytes object per window), logits_split (W, K) after the
    link, logits_full (W, K) of the whole network, values (per window) and
    header_B (bytes of the packet that are not values).
    """
    from iomtlab.edge.split import split_model
    from iomtlab.edge.wire import pack_int8, unpack_int8
    from iomtlab.train.data import to_input
    x = X if isinstance(X, torch.Tensor) else to_input(X)
    head, tail = split_model(model, at)
    model.eval()
    packets, z = [], []
    for i in range(len(x)):
        a = head(x[i:i + 1])                        # sul nodo
        b = pack_int8(a)                            # i byte sul collegamento
        packets.append(b)
        z.append(tail(torch.from_numpy(unpack_int8(b))))   # sul server
    full = torch.cat([model(x[i:i + 512]) for i in range(0, len(x), 512)])
    values = int(a[0].numel()) if len(x) else 0
    return {"packets": packets, "logits_split": torch.cat(z).numpy(),
            "logits_full": full.numpy(), "values": values,
            "header_B": (len(packets[0]) - values) if packets else 0}


def train_exit(model: nn.Module, at: str, Xtr, ytr, epochs: int | None = None,
               seed: int | None = None, n_classes: int | None = None) -> nn.Module:
    """A linear exit after the stage at, trained with the head frozen; returned in eval mode.

    The exit is Flatten + Linear(values at the cut, n_classes): after pool2
    of the example network 768 x 5 = 3 840 MACs. The head is not changed:
    its outputs on Xtr are computed once and the exit is trained on them with
    the course recipe (train.RECIPE: seed, epochs, batches of 64, Adam 1e-3,
    inverse-frequency weights). Apply it to the head output: exit(head(x)).
    """
    from iomtlab.edge.split import split_model
    from iomtlab.train.compare import RECIPE, make_loader
    from iomtlab.train.data import to_input
    from iomtlab.train.loops import fit, inverse_freq_weights, set_seed
    from iomtlab.spec import active
    n_classes = n_classes or active().n_classes
    seed = RECIPE["seed"] if seed is None else seed
    epochs = RECIPE["epochs"] if epochs is None else epochs
    head, _ = split_model(model, at)
    model.eval()
    x = to_input(Xtr)
    with torch.no_grad():
        h = torch.cat([head(x[i:i + 512]) for i in range(0, len(x), 512)])
    set_seed(seed)                                  # prima di costruire l'uscita: pesi iniziali fissi
    ex = nn.Sequential(nn.Flatten(), nn.Linear(int(h[0].numel()), n_classes))
    fit(ex, make_loader(h, ytr, seed=seed), epochs=epochs,
        weights=inverse_freq_weights(np.asarray(ytr), n_classes), lr=RECIPE["lr"], seed=seed)
    return ex.eval()


def exit_curve(z_exit, z_tail, y, thresholds, e_head_J: float, e_link_J: float) -> list[dict]:
    """For each threshold θ: share q decided on the node, energy, accuracy, macro-F1.

    A window is decided on the node when the highest softmax probability of
    the exit logits z_exit is at least θ; otherwise the activation is sent and
    the class is that of z_tail. Energy per decision e_head_J + (1 - q) e_link_J:
    e_head_J (head and exit) is paid by every window, e_link_J only by those
    sent. Each row has theta, q, energy_J, accuracy, macro_f1, acc_local (the
    accuracy of the windows decided on the node, None if there are none).
    """
    from iomtlab.evaluation import macro_f1
    y = np.asarray(y)
    z = np.asarray(z_exit, dtype=np.float64)
    p = np.exp(z - z.max(1, keepdims=True))         # softmax, stabile: tolgo il massimo
    p /= p.sum(1, keepdims=True)
    conf, local_pred = p.max(1), p.argmax(1)
    tail_pred = np.asarray(z_tail).argmax(1)
    rows = []
    for th in thresholds:
        loc = conf >= th                            # qui decide il nodo
        pred = np.where(loc, local_pred, tail_pred)
        q = float(loc.mean())
        rows.append({"theta": float(th), "q": q, "energy_J": e_head_J + (1 - q) * e_link_J,
                     "accuracy": float(np.mean(pred == y)), "macro_f1": macro_f1(y, pred),
                     "acc_local": float(np.mean(local_pred[loc] == y[loc])) if loc.any() else None})
    return rows


def code_entropy(model: nn.Module, at: str, X, bits: int = 8) -> dict:
    """How many bits the code at the cut would need with an entropy coder.

    The head runs on X, its output is quantised to 2**bits levels (one scale
    per tensor, as on the link) and the zero-order entropy of the levels is
    measured: values taken one by one, no context, so it is an estimate of
    what an entropy coder alone would save, not a bound for a coder trained
    with a rate term (block 4.1, «A bottleneck that compresses»).
    Returns values per window, fixed bits per window, entropy per value and
    the estimated bits per window.
    """
    from iomtlab.edge.split import split_model
    from iomtlab.train.data import to_input
    head, _ = split_model(model.eval(), at)
    x = X if torch.is_tensor(X) else to_input(X)
    with torch.no_grad():
        z = head(x).detach().cpu().numpy().astype(np.float64)
    lo, hi = z.min(), z.max()
    levels = 2 ** bits - 1
    q = np.zeros(z.shape, dtype=np.int64) if hi == lo else np.round((z - lo) / (hi - lo) * levels).astype(np.int64)
    _, counts = np.unique(q, return_counts=True)
    p = counts / counts.sum()
    h = float(-(p * np.log2(p)).sum())
    per_window = int(np.prod(z.shape[1:]))
    return {"values": per_window, "bits_fixed": per_window * bits, "entropy_per_value": h,
            "bits_entropy": per_window * h}
