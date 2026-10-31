"""Routing on conformal sets (Lab 7): the node decides on singletons, forwards the rest.

The rule is a construction of the course, the simplest that uses the set
(deck «Affidabilità», «The rule»): the node runs the network, builds the
conformal set with q̂; with exactly one class it decides and sends the
decision (3 bits), otherwise (two or more classes, or an empty set) it
forwards the raw window and the set as a mask of K bits. What is guaranteed is
the coverage of the sets, not the accuracy of the node's decisions.

  nonconformity        s = 1 - p(true class)
  coverage_by_subject  coverage of the sets per subject
  routing_table        for each alpha: q̂, coverage, set size, share decided on
                       the node, accuracy of node, server and system
  energy_constants     the energies of the two configurations, from the
                       curve of Lab 6 if there is one
  energy_per_decision  e_local + (1 - q) e_forward
  link_load            bits per second per node and nodes per channel
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from iomtlab.config import ENERGY, N_CHANNELS, N_CLASSES, WINDOW_SIZE

__all__ = ["nonconformity", "coverage_by_subject", "routing_table", "energy_constants",
           "energy_per_decision", "link_load"]


def nonconformity(p, y) -> np.ndarray:
    """Non-conformity scores 1 - p[i, y_i] of probabilities p (n, K) and labels y (n,)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    return 1.0 - p[np.arange(len(y)), y]


def coverage_by_subject(sets, y_true, subjects) -> pd.Series:
    """Coverage of the sets for each subject, as a Series indexed by subject, sorted by name."""
    sets = np.asarray(sets, dtype=bool)
    y_true = np.asarray(y_true, dtype=np.int64)
    hit = sets[np.arange(len(y_true)), y_true]     # vero dove l'insieme contiene la classe vera
    s = pd.Series(hit.astype(float), index=np.asarray(subjects, dtype=object).astype(str))
    return s.groupby(level=0).mean().sort_index()


def routing_table(p_cal, y_cal, p_test, y_test, alphas, server_pred=None) -> pd.DataFrame:
    """For each alpha, the sets on p_test and the routing on singletons: one row per alpha.

    Columns: alpha, qhat, coverage, mean_size, q (share of singletons, decided
    on the node), empty (share of empty sets, forwarded), acc_local
    (accuracy of the node on its singletons), and, with server_pred (the
    class the server gives to every test window), acc_server (accuracy of the
    server on the forwarded windows) and acc_system (node on singletons,
    server on the rest). Accuracies on an empty group are NaN.
    """
    from iomtlab.uncertainty.conformal import conformal_quantile
    p_test = np.asarray(p_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.int64)
    s_cal = nonconformity(p_cal, y_cal)
    rows = []
    for a in alphas:
        qh = conformal_quantile(s_cal, a)
        S = (1.0 - p_test) <= qh                    # l'insieme: le classi con 1 - p <= q̂
        size = S.sum(1)
        single = size == 1                          # qui decide il nodo
        local = S.argmax(1)                         # la classe del singoletto
        hit = S[np.arange(len(y_test)), y_test]
        r = {"alpha": float(a), "qhat": float(qh), "coverage": float(hit.mean()),
             "mean_size": float(size.mean()), "q": float(single.mean()),
             "empty": float((size == 0).mean()),
             "acc_local": float(np.mean(local[single] == y_test[single])) if single.any()
             else float("nan")}
        if server_pred is not None:
            sp = np.asarray(server_pred)
            fwd = ~single
            r["acc_server"] = float(np.mean(sp[fwd] == y_test[fwd])) if fwd.any() else float("nan")
            r["acc_system"] = float(np.mean(np.where(single, local, sp) == y_test))
        rows.append(r)
    return pd.DataFrame(rows)


def energy_constants(curve=None, model=None, cut: str = "pool2") -> dict:
    """The energies per decision of the two configurations of Lab 7, in J.

    A: the whole network on the node, e_full_J = MACs * e_mac; forwarding
       sends the raw window, e_raw_J = 6 x 50 x 16 bits * e_bit.
    B: the head up to cut on the node, e_head_J; forwarding sends the int8
       activation at the cut, e_link_J.
    curve is the dict of io.load_curve (work/curve.json, written by Lab 6):
    if given, e_head_J and e_link_J are read from it (head plus exit, as in
    Lab 6); otherwise they are computed for model (default ExampleCNN) with
    edge.cuts.cut_profile, the head without exit. The source is in "source".
    Constants of config.ENERGY on IEEE 802.15.4; the raw window is that of the
    active dataset (window x channels x sample_bits).
    """
    from iomtlab.edge.cuts import cut_profile
    from iomtlab.models import ExampleCNN
    e_mac, e_bit = ENERGY["e_mac_J"], ENERGY["e_bit_lowpower_J"]
    rows = {r["name"]: r for r in cut_profile(model or ExampleCNN())}
    from iomtlab.spec import active
    E = {"e_full_J": rows["classe"]["mac_head"] * e_mac,
         "e_raw_J": active().window_bits * e_bit}
    if curve is not None:
        E.update(e_head_J=float(curve["e_head_J"]), e_link_J=float(curve["e_link_J"]),
                 source="work/curve.json (Lab 6)")
    else:
        E.update(e_head_J=rows[cut]["mac_head"] * e_mac, e_link_J=rows[cut]["bits"] * e_bit,
                 source=f"cut_profile, taglio dopo {cut}, senza uscita")
    return E


def energy_per_decision(q, e_local_J: float, e_forward_J: float):
    """e_local_J + (1 - q) e_forward_J: the local part is paid by every window, the rest only
    by the forwarded ones. q can be a number or an array (a float or an array comes back)."""
    q = np.asarray(q, dtype=np.float64)
    e = e_local_J + (1.0 - q) * e_forward_J
    return float(e) if e.ndim == 0 else e


def link_load(q, decisions_per_s: float = 2.0, class_bits: int | None = None,
              forward_bits: int | None = None,
              rate_bps: float = ENERGY["rate_lowpower_bps"],
              usable_fraction: float = ENERGY["usable_fraction"]) -> dict:
    """Bits per second on the air for one node, and nodes per channel, for a share q decided locally.

    A decision is one 802.15.4 frame with class_bits; a forwarded window is
    the raw window plus the set as a mask of K bits (forward_bits), in frames
    of iomtlab.radio (116 useful bytes, 17 of headers). nodes is how many
    nodes fit in usable_fraction of rate_bps, rounded down as in the fourth
    test of Lab 5 (520 nodes at q = 1, 13 at q = 0). Returns {"bit_s",
    "nodes"}, floats or arrays like q.
    """
    from iomtlab import radio
    from iomtlab.spec import active
    S = active()
    class_bits = class_bits or S.class_bits
    forward_bits = forward_bits or S.window_bits + S.n_classes   # la finestra e la maschera dell'insieme
    q = np.asarray(q, dtype=np.float64)
    a_dec = 8 * radio.air_bytes(int(math.ceil(class_bits / 8)))
    a_fwd = 8 * radio.air_bytes(int(math.ceil(forward_bits / 8)))
    bit_s = decisions_per_s * (q * a_dec + (1.0 - q) * a_fwd)   # media per nodo
    nodes = np.floor(usable_fraction * rate_bps / bit_s)
    if q.ndim == 0:
        return {"bit_s": float(bit_s), "nodes": float(nodes)}
    return {"bit_s": bit_s, "nodes": nodes}
