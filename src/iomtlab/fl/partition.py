"""How the data are split among clients.

subject_clients is the natural split: the data are already separated by
person, and a person is a client. dirichlet_partition is artificial; it
controls how skewed the class mix of each client is, to see where FedAvg
degrades.
"""
from __future__ import annotations

import numpy as np


__all__ = ["subject_clients", "dirichlet_partition"]


def subject_clients(subject) -> dict:
    """One client per subject: {subject_id: array of the indices of its rows}."""
    subject = np.asarray(subject, dtype=object).astype(str)
    return {s: np.where(subject == s)[0] for s in np.unique(subject)}


def dirichlet_partition(y, n_clients: int, beta: float = 0.5,
                        seed: int = 0, min_per_client: int = 1):
    """Non-IID split of the rows by label, drawn from a Dirichlet(beta).

    For each class the share of each client is drawn from Dirichlet(beta):
    large beta gives similar clients, small beta clients with nearly one
    class. With small beta a client can end up with fewer than
    min_per_client rows; the missing rows are moved from the largest client
    and counted in info["moved"].

    Returns (parts, info): parts is a list of index arrays, one per client;
    info has beta, seed, moved and sizes.
    """
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = [[] for _ in range(n_clients)]
    for k in np.unique(y):
        pos = np.where(y == k)[0]            # le righe della classe k
        rng.shuffle(pos)                     # le mescolo, così ogni client ne riceve un sottoinsieme a caso
        w = rng.dirichlet([beta] * n_clients)   # quota della classe k per ogni client
        cuts = (np.cumsum(w)[:-1] * len(pos)).astype(int)   # trasformo le quote in punti di taglio
        for i, part in enumerate(np.split(pos, cuts)):
            idx[i].extend(part.tolist())
    parts = [np.array(sorted(i), dtype=np.int64) for i in idx]

    # se un client è rimasto sotto il minimo gli sposto righe dal client più grande
    moved = 0
    pool = int(np.argmax([len(p) for p in parts]))
    for i, p in enumerate(parts):
        while len(parts[i]) < min_per_client and len(parts[pool]) > min_per_client:
            parts[i] = np.append(parts[i], parts[pool][-1])
            parts[pool] = parts[pool][:-1]
            moved += 1

    # controllo che ogni riga sia finita in esattamente un client
    flat = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    if len(flat) != len(y) or len(np.unique(flat)) != len(y):
        raise AssertionError("una riga è finita in due client, o in nessuno")
    return parts, {"beta": beta, "seed": seed, "moved": moved,
                   "sizes": [len(p) for p in parts]}
