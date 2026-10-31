"""The example network of the theory, and the same network with a bottleneck.

ExampleCNN (block 2.1, «The example network, block by block»):

    conv1  Conv1d(6, 32, 5, padding="same") + ReLU     32 x 50
    pool1  MaxPool1d(2)                                32 x 25
    conv2  Conv1d(32, 64, 5, padding="same") + ReLU    64 x 25
    pool2  MaxPool1d(2)                                64 x 12
    conv3  Conv1d(64, 64, 5, padding="same") + ReLU    64 x 12
    pool3  MaxPool1d(2)                                64 x 6, flattened to 384
    dense  Linear(384, 64) + ReLU
    out    Linear(64, 5)                               the logits

56 805 parameters, 574 656 MACs per window, receptive field 36 samples.

BottleneckCNN adds two 1 x 1 convolutions after pool2, 64 -> 4 -> 64 channels
(«A bottleneck at the cut»): 57 385 parameters and 580 800 MACs, 1 % more,
and 48 values cross the boundary after the bottleneck instead of 768 after
pool2. It is built to be cut there (Lab 6). With at="pool1" the bottleneck
goes after pool1 instead, 32 -> c -> 32 channels on 25 samples: the head on
the node is then conv1 and the 1 x 1 (49 600 MACs for c = 2, 400 bits in
int8), far less than after pool2 (block 4.1, «Where and how wide»). code="linear"
leaves the code without ReLU: with one or two channels a ReLU code can die (every value
zero, a constant network).

Channels, classes and window length default to the active dataset
(iomtlab.spec.active()): the TUG numbers above unless a project changes it.
"""
from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["ExampleCNN", "BottleneckCNN"]


class ExampleCNN(nn.Module):
    """Three conv + ReLU + max-pool blocks, a dense layer, the output. Input (B, C, T)."""

    def __init__(self, in_channels: int | None = None, num_classes: int | None = None,
                 width=(32, 64, 64), kernel: int = 5, hidden: int = 64,
                 seq_len: int | None = None):
        super().__init__()
        from iomtlab.spec import active      # canali, classi e campioni del dataset attivo
        S = active()
        in_channels = in_channels or S.n_channels
        num_classes = num_classes or S.n_classes
        seq_len = seq_len or S.window
        c1, c2, c3 = width
        self.width, self.kernel, self.hidden = tuple(width), kernel, hidden
        self.conv1 = nn.Conv1d(in_channels, c1, kernel, padding="same")
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(c1, c2, kernel, padding="same")
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(c2, c3, kernel, padding="same")
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)
        t = seq_len // 2 // 2 // 2               # 50 -> 25 -> 12 -> 6: la divisione tronca
        self.dense = nn.Linear(c3 * t, hidden)
        self.relu4 = nn.ReLU()
        self.out = nn.Linear(hidden, num_classes)

    def config(self) -> dict:
        """The constructor arguments that change the architecture."""
        return {"width": list(self.width), "kernel": self.kernel, "hidden": self.hidden}

    def _after_pool2(self, x):
        """What follows pool2; BottleneckCNN inserts its two 1 x 1 convolutions here."""
        return x

    def _after_pool1(self, x):
        """What follows pool1; BottleneckCNN(at="pool1") inserts its convolutions here."""
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self._after_pool1(x)
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self._after_pool2(x)
        x = self.pool3(self.relu3(self.conv3(x))).flatten(1)
        return self.out(self.relu4(self.dense(x)))

    def stages(self):
        """The forward as (name, callable) pairs, in order; composed they give forward.

        edge.split_model cuts only at these boundaries.
        """
        return [
            ("conv1", lambda x: self.relu1(self.conv1(x))),
            ("pool1", self.pool1),
            *self._extra_stages("pool1"),
            ("conv2", lambda x: self.relu2(self.conv2(x))),
            ("pool2", self.pool2),
            *self._extra_stages("pool2"),
            ("conv3", lambda x: self.relu3(self.conv3(x))),
            ("pool3", lambda x: self.pool3(x).flatten(1)),
            ("dense", lambda x: self.relu4(self.dense(x))),
            ("out", self.out),
        ]

    def _extra_stages(self, where: str = "pool2"):
        return []


class BottleneckCNN(ExampleCNN):
    """ExampleCNN with a 1 x 1 bottleneck after pool2 (default) or pool1: C -> bottleneck -> C channels."""

    def __init__(self, in_channels: int | None = None, num_classes: int | None = None,
                 width=(32, 64, 64), kernel: int = 5, hidden: int = 64,
                 bottleneck: int = 4, seq_len: int | None = None, at: str = "pool2",
                 code: str = "relu"):
        super().__init__(in_channels, num_classes, width, kernel, hidden, seq_len)
        if at not in ("pool1", "pool2"):
            raise ValueError("at deve essere 'pool1' o 'pool2'")
        if code not in ("relu", "linear"):
            raise ValueError("code deve essere 'relu' o 'linear'")
        self.at, self.code = at, code
        c = width[0] if at == "pool1" else width[1]          # i canali nel punto del collo
        self.bottleneck = nn.Conv1d(c, bottleneck, 1)       # il collo: pochi canali sul confine
        # il codice che attraversa il taglio: con ReLU un collo di 1-2 canali può spegnersi del
        # tutto (tutti i valori a zero, la rete diventa costante); "linear" lo lascia lineare
        self.relu_b = nn.ReLU() if code == "relu" else nn.Identity()
        self.expand = nn.Conv1d(bottleneck, c, 1)           # e di nuovo i canali di prima
        self.relu_e = nn.ReLU()

    def config(self) -> dict:
        return {**super().config(), "bottleneck": self.bottleneck.out_channels, "at": self.at,
                "code": self.code}

    def _neck(self, x):
        return self.relu_e(self.expand(self.relu_b(self.bottleneck(x))))

    def _after_pool1(self, x):
        return self._neck(x) if self.at == "pool1" else x

    def _after_pool2(self, x):
        return self._neck(x) if self.at == "pool2" else x

    def _extra_stages(self, where: str = "pool2"):
        if where != self.at:
            return []
        return [("bottleneck", lambda x: self.relu_b(self.bottleneck(x))),
                ("expand", lambda x: self.relu_e(self.expand(x)))]
