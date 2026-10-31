"""Working memory of one inference: the largest input plus output of a layer.

During inference a layer that has finished frees its input, so the RAM
needed is the largest sum of the input and the output of one layer, not the
sum of all activations. Element-wise layers (ReLU, dropout) work in place and
are not counted. Training is different: every activation is kept for the
backward pass.
"""
from __future__ import annotations

import torch.nn as nn

from iomtlab.config import N_CHANNELS, WINDOW_SIZE
from iomtlab.models.cost import layer_profile

__all__ = ["peak_activation_values", "peak_activation_bytes"]

_IN_PLACE = ("relu", "dropout", "identity")


def peak_activation_values(model: nn.Module, in_channels: int | None = None,
                           seq_len: int | None = None) -> int:
    """Largest number of values held at once by one layer: its input plus its output."""
    from iomtlab.spec import active
    in_channels = in_channels or active().n_channels
    seq_len = seq_len or active().window
    prev = in_channels * seq_len                 # l'ingresso della prima fase è la finestra
    peak = 0
    for l in layer_profile(model, in_channels, seq_len).layers:
        if l.kind in _IN_PLACE:
            continue                             # lavorano sul posto: niente memoria in più
        peak = max(peak, prev + l.out_elements)
        prev = l.out_elements
    return peak


def peak_activation_bytes(model: nn.Module, dtype_bytes: int = 4, **kw) -> int:
    """peak_activation_values in bytes, at dtype_bytes per value (4 float32, 1 int8)."""
    return peak_activation_values(model, **kw) * dtype_bytes
