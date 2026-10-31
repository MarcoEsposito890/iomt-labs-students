"""Averaging without a server (Lab 8): graphs, mixing weights, the rate of consensus.

Each node holds a value and repeats x(k+1) = W x(k), averaging with the
neighbours it hears. With W symmetric and rows summing to one (so columns
too) on a connected graph, every node converges to the true mean, at a rate
set by the second largest eigenvalue modulus ρ of W (Xiao and Boyd, Systems &
Control Letters 53, 2004, Theorem 1).

  ring_graph, grid_graph     adjacency matrices, 0/1, no self-loops
  metropolis_weights         W_ij = 1 / (1 + max(d_i, d_j)) on the edges,
                             W_ii = 1 - sum of the row (Xiao, Boyd and Lall,
                             IPSN 2005, Eq. 9): symmetric, doubly stochastic
  uniform_weights            W_ij = 1 / (1 + d_i) for j = i and the
                             neighbours: rows sum to one, columns not
  second_eigenvalue          ρ, the second largest eigenvalue modulus
  run_consensus              the trajectory x(0), ..., x(k)
  disagreement               ||x(k) - mean(x(0))|| / ||x(0) - mean(x(0))||
  iterations_to              first k with disagreement <= tol
"""
from __future__ import annotations

import numpy as np

__all__ = ["ring_graph", "grid_graph", "metropolis_weights", "uniform_weights",
           "second_eigenvalue", "run_consensus", "disagreement", "iterations_to"]


def ring_graph(n: int) -> np.ndarray:
    """Adjacency matrix of a ring of n nodes (n >= 3): each node hears the two next to it."""
    if n < 3:
        raise ValueError("un anello ha almeno 3 nodi")
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    return A


def grid_graph(rows: int, cols: int) -> np.ndarray:
    """Adjacency matrix of a rows x cols grid, not wrapped: corners have 2 neighbours, inner nodes 4."""
    n = rows * cols
    A = np.zeros((n, n), dtype=int)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                A[i, i + 1] = A[i + 1, i] = 1          # il vicino a destra
            if r + 1 < rows:
                A[i, i + cols] = A[i + cols, i] = 1    # il vicino sotto
    return A


def metropolis_weights(A) -> np.ndarray:
    """Metropolis-Hastings weights of the graph A: symmetric, rows and columns summing to one."""
    A = np.asarray(A) > 0
    d = A.sum(1)
    n = len(A)
    W = np.zeros((n, n))
    for i in range(n):
        for j in np.nonzero(A[i])[0]:
            W[i, j] = 1.0 / (1 + max(d[i], d[j]))
        W[i, i] = 1.0 - W[i].sum()                 # il resto al nodo stesso
    return W


def uniform_weights(A) -> np.ndarray:
    """Equal weights 1 / (1 + d_i) on the node and its neighbours: rows sum to one, columns not."""
    A = np.asarray(A) > 0
    W = (A | np.eye(len(A), dtype=bool)).astype(float)
    return W / W.sum(1, keepdims=True)


def second_eigenvalue(W) -> float:
    """ρ: the second largest modulus among the eigenvalues of W (the largest is 1)."""
    lam = np.sort(np.abs(np.linalg.eigvals(np.asarray(W, dtype=np.float64))))[::-1]
    return float(lam[1])


def run_consensus(W, x0, iterations: int) -> np.ndarray:
    """The trajectory of x(k+1) = W x(k): an (iterations + 1, n) array, row 0 is x0."""
    W = np.asarray(W, dtype=np.float64)
    x = np.asarray(x0, dtype=np.float64).copy()
    out = [x.copy()]
    for _ in range(iterations):
        x = W @ x                                  # ogni nodo fa la media pesata con i vicini
        out.append(x.copy())
    return np.array(out)


def disagreement(traj) -> np.ndarray:
    """Distance of every row from the true mean of row 0, relative to row 0's distance."""
    traj = np.asarray(traj, dtype=np.float64)
    mu = traj[0].mean()                            # la media vera, quella delle letture iniziali
    d0 = np.linalg.norm(traj[0] - mu)
    return np.linalg.norm(traj - mu, axis=1) / (d0 if d0 > 0 else 1.0)


def iterations_to(err, tol: float = 0.01):
    """The first k with err[k] <= tol, as an int; None if err never gets there."""
    hit = np.nonzero(np.asarray(err) <= tol)[0]
    return int(hit[0]) if len(hit) else None
