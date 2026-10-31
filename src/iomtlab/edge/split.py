"""Cutting a network in two, and the cost of each boundary.

A cut is a boundary between layers: there the intermediate result is a
complete tensor that can be sent. Inside a layer there is nothing to send.
"""
from __future__ import annotations

import torch.nn as nn

__all__ = ["split_model", "cut_points", "cut_cost"]


class _Chain(nn.Module):
    """A run of stages declared by the model; shares the parameters of the original."""

    def __init__(self, owner: nn.Module, stages, label: str):
        super().__init__()
        self.owner = owner
        self._stages = list(stages)
        self.label = label

    def forward(self, x):
        for _, f in self._stages:
            x = f(x)                         # applico le fasi una dopo l'altra
        return x

    def __repr__(self):
        return f"_Chain({self.label}: {[n for n, _ in self._stages]})"


def split_model(model: nn.Module, at: str) -> tuple[nn.Module, nn.Module]:
    """Return (head, tail), cutting after the stage named at.

    The cut is allowed only where the model declares a boundary through
    stages(). nn.Sequential(*children) would not do: the forward of these
    networks has a flatten(1) and a permute that are not modules, and the
    two pieces would compute something else. Raises NotImplementedError for
    a model without stages(), such as DeepConvLSTM, and ValueError if at is
    not one of the stage names.
    """
    if not hasattr(model, "stages"):         # DeepConvLSTM, per esempio, non dichiara confini
        raise NotImplementedError(
            f"{type(model).__name__} non ha il metodo stages(), che dichiara dove "
            f"si può tagliare")
    stages = list(model.stages())            # le fasi che il modello dichiara, in ordine
    names = [n for n, _ in stages]
    if at not in names:
        raise ValueError(f"'{at}' non è un confine; i confini sono {names}")
    i = names.index(at)
    # la testa arriva fino alla fase `at` compresa, la coda è tutto il resto
    return (_Chain(model, stages[:i + 1], "head"),
            _Chain(model, stages[i + 1:], "tail"))


def cut_points(profile, bits: int = 32) -> list[dict]:
    """For each layer boundary: output values, bits, MACs of head and tail.

    profile is a ModelCost (models.layer_profile); bits is the size of one
    value. Returns one dict per layer with name, values, bits, mac_head and
    mac_tail. mac_head + mac_tail is the total at every cut.
    """
    rows, cum = [], 0
    total = int(profile.macs)                # i MAC totali della rete
    for L in profile.layers:
        cum += int(L.macs)                   # MAC accumulati fino a questo strato: sono quelli della testa
        vals = int(L.out_elements)           # i valori in uscita, cioè quelli da mandare sul collegamento
        rows.append({"name": L.name, "values": vals, "bits": vals * bits,
                     "mac_head": cum, "mac_tail": total - cum})
    return rows


def cut_cost(rows, e_mac_J: float, e_bit_J: float, rate_bps: float,
             hops: int = 1) -> list[dict]:
    """Time and energy at each cut point: the rows of cut_points with more keys.

    e_mac_J is the energy per MAC on the node [J], e_bit_J the energy per
    transmitted bit [J], rate_bps the link rate [bit/s]; the course values
    are in config.ENERGY (block 5.1). Adds e_node_J, e_link_J, e_tot_J (J),
    t_link_s (s) and hops.

    hops is the number of radio hops to the receiver (default 1, the theory).
    With h hops every bit is sent h times, once by the node and once by each
    of the h - 1 relays: e_link_J is the radio energy of all of them, h Z
    e_bit, and t_link_s the time the channel is busy, h Z / r (store and
    forward, one transmission at a time). e_link_node_J is the share of the
    node alone, Z e_bit. Reception is not counted, as in the theory.
    """
    if hops < 1:
        raise ValueError("hops deve essere almeno 1")
    out = []
    for r in rows:
        e_node = r["mac_head"] * e_mac_J     # energia di calcolo della testa sul nodo
        e_tx = r["bits"] * e_bit_J           # una trasmissione dell'attivazione al taglio
        e_link = hops * e_tx                 # il nodo e chi inoltra: h trasmissioni
        out.append({**r,
                    "e_node_J": e_node,
                    "e_link_J": e_link,
                    "e_link_node_J": e_tx,
                    "e_tot_J": e_node + e_link,
                    "t_link_s": hops * r["bits"] / rate_bps,   # canale occupato per h trasmissioni
                    "hops": int(hops)})
    return out
