"""GapCNN: the reference solution of the model exercises of Labs 4 and 6.

Two conv + ReLU + max-pool blocks and a global average pooling over time, then
the output layer: no dense layer on the flattened activation, so the number of
parameters does not grow with the window. Input (B, C, T), logits (B, K).

GapCNN.stages() is the Lab 4 solution (four stages: blocco1, blocco2, media,
out); GapCNNFine.stages() the Lab 6 one, with the second block split before
its pooling (conv2, pool2), so that one more cut is possible.
"""
from __future__ import annotations

import torch.nn as nn

__all__ = ["GapCNN", "GapCNNFine"]


class GapCNN(nn.Module):
    """conv(C->16, 5) + ReLU + pool, conv(16->32, 5) + ReLU + pool, mean over time, Linear(32->K)."""

    def __init__(self, in_channels: int | None = None, num_classes: int | None = None,
                 width=(16, 32)):
        super().__init__()
        from iomtlab.spec import active
        S = active()
        C, K = in_channels or S.n_channels, num_classes or S.n_classes
        self.conv1 = nn.Conv1d(C, width[0], 5, padding=2)
        self.conv2 = nn.Conv1d(width[0], width[1], 5, padding=2)
        self.relu, self.pool = nn.ReLU(), nn.MaxPool1d(2)
        self.out = nn.Linear(width[1], K)

    def stages(self):
        r, p = self.relu, self.pool
        return [("blocco1", lambda x: p(r(self.conv1(x)))),
                ("blocco2", lambda x: p(r(self.conv2(x)))),
                ("media", lambda x: x.mean(dim=2)),          # media sul tempo: (B, 32, T/4) -> (B, 32)
                ("out", self.out)]

    def forward(self, x):
        for _, f in self.stages():
            x = f(x)
        return x


class GapCNNFine(GapCNN):
    """GapCNN with the second block split before its pooling: one more possible cut."""

    def stages(self):
        r, p = self.relu, self.pool
        return [("blocco1", lambda x: p(r(self.conv1(x)))),
                ("conv2", lambda x: r(self.conv2(x))),        # confine prima del pooling
                ("pool2", p),
                ("media", lambda x: x.mean(dim=2)),
                ("out", self.out)]
