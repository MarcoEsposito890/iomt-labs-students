"""Deciding with more than one node: fuse the scores, or average them with the neighbours.

Lab 8. Two nodes that see the same event (the phone and the watch of TUG)
can each decide alone, exchange only their class probabilities, or send the
raw samples to one of them. Many nodes without a sink average their scores
with the neighbours they hear (fl.consensus): every iteration costs one
transmission per node.

  fuse_mean             average of the probability vectors of the nodes
  fuse_confidence       average weighted by the confidence of each node
                        (its largest probability)
  fuse                  apply a fusion rule to a list of (W, K) arrays; the
                        rule is a function, so a project can plug its own
  softmax               logits -> probabilities, row by row
  align_by_group        rows of several nodes that belong to the same event
  score_bits            bits of K probabilities at bits_per_value each
  consensus_cost        radio energy and air time of k iterations of
                        averaging on a graph: one broadcast per node each
  gateway_cost          the same values sent once to a sink, hops away

For consensus, reception is counted at the cost of a transmission (rx=True), as in
block 6.2; set rx=False to compare with edge.split.cut_cost, which counts only what is
sent. The radio is a Link of edge.regimes (frames, LoRa airtime).
"""
from __future__ import annotations

import numpy as np

__all__ = ["softmax", "fuse_mean", "fuse_confidence", "fuse", "align_by_group", "score_bits",
           "consensus_cost", "gateway_cost"]


def softmax(z) -> np.ndarray:
    """Probabilities from logits (W, K), row by row; stable (the maximum is subtracted)."""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fuse_mean(probs) -> np.ndarray:
    """The average of the probability vectors of the nodes: probs is a list of (W, K) arrays."""
    return np.mean(np.stack([np.asarray(p, dtype=np.float64) for p in probs]), axis=0)


def fuse_confidence(probs) -> np.ndarray:
    """Average weighted, window by window, by the largest probability of each node.

    A node sure of its class counts more than a hesitant one; the weights sum
    to one in every window.
    """
    P = np.stack([np.asarray(p, dtype=np.float64) for p in probs])      # (N, W, K)
    w = P.max(axis=2, keepdims=True)                                   # la confidenza di ogni nodo
    return (w * P).sum(0) / w.sum(0)


def fuse(probs, rule=fuse_mean) -> np.ndarray:
    """rule(probs) on a list of (W, K) probability arrays, checked: rows still sum to one."""
    out = np.asarray(rule(list(probs)), dtype=np.float64)
    if out.shape != np.asarray(probs[0]).shape:
        raise ValueError(f"la regola restituisce {out.shape}, attesi {np.asarray(probs[0]).shape}")
    if not np.allclose(out.sum(1), 1.0, atol=1e-6):
        raise ValueError("le righe della fusione non sommano a uno: non sono probabilità")
    return out


def align_by_group(arrays: dict, groups: dict):
    """Line up the rows of several nodes by event: arrays and groups map a node to (W_node, ...) and (W_node,).

    Keeps the groups that every node has (one row each; a node with two rows
    for the same group keeps the first) and returns (list of arrays in the
    order of the keys of arrays, common groups as an array).
    """
    nodes = list(arrays)
    first = {}
    for n in nodes:
        g = np.asarray(groups[n])
        _, idx = np.unique(g, return_index=True)
        first[n] = dict(zip(g[idx].tolist(), idx.tolist()))
    common = sorted(set.intersection(*(set(first[n]) for n in nodes)))
    out = [np.asarray(arrays[n])[[first[n][c] for c in common]] for n in nodes]
    return out, np.array(common, dtype=object)


def score_bits(n_classes: int, bits_per_value: int = 8) -> int:
    """Bits of one vector of n_classes probabilities, bits_per_value each (40 for 5 classes in int8)."""
    return int(n_classes * bits_per_value)


def consensus_cost(A, iterations: int, value_bits: int, link, rx: bool = True) -> dict:
    """Radio cost of iterations of neighbour averaging on the graph A (adjacency, 0/1).

    Every iteration each node broadcasts its value once (value_bits) and hears
    the broadcast of each neighbour. With rx (default, as in block 6.2,
    «Scalar average on 802.15.4») a received frame costs what a sent one
    costs; the node with the mean degree is reported. Returns energy_J_node
    (None if the link has no power), energy_J_total, t_air_s_node and
    frames_node (sent and, with rx, received) over all the iterations.
    """
    from iomtlab.edge.regimes import link_cost
    A = np.asarray(A)
    n = int(A.shape[0])
    d = float((A > 0).sum(1).mean()) if n else 0.0         # vicini medi
    c = link_cost(link, value_bits, 1)
    per_it = 1.0 + (d if rx else 0.0)                       # una trasmissione, d ricezioni
    e = None if c["energy_J"] is None else iterations * per_it * c["energy_J"]
    return {"nodes": n, "iterations": int(iterations), "mean_degree": d,
            "energy_J_node": e, "energy_J_total": None if e is None else n * e,
            "t_air_s_node": iterations * c["t_air_s"], "frames_node": iterations * per_it * c["frames"]}


def gateway_cost(n_nodes: int, value_bits: int, link, hops: int = 1, rx: bool = True) -> dict:
    """Radio cost of sending each node's value once to a sink hops away: the alternative to consensus.

    hops transmissions per value (the node and the relays) and, with rx, as
    many receptions at the same cost. The decision is not sent back here:
    add it if the nodes need it.
    """
    from iomtlab.edge.regimes import link_cost
    c = link_cost(link, value_bits, 1)
    per = hops * (2 if rx else 1)
    e = None if c["energy_J"] is None else per * c["energy_J"]
    return {"nodes": int(n_nodes), "hops": int(hops), "energy_J_node": e,
            "energy_J_total": None if e is None else n_nodes * e,
            "t_air_s_node": hops * c["t_air_s"], "frames_node": per * c["frames"]}
