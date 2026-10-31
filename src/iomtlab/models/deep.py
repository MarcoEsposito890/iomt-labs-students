"""The larger models of the compression path.

We start from the model that works best and shrink it, measuring what is
lost at each step:

    DeepConvLSTM   4 convolutional blocks + 2 LSTM layers
    DeepCNN        the same 4 convolutional blocks, no recurrence
    SimpleCNN      one convolutional block (iomtlab.models.simple_cnn)

Measured on three subject folds, subjects s01-s12, smartphone, 25 epochs
(the reference values of the labs):

    model          params       MACs   macro-F1        inference b=1
    SimpleCNN      21 829      210 k   0.723 +- 0.029        0.06 ms
    DeepCNN        63 941     3.17 M   0.736 +- 0.033        0.21 ms
    DeepConvLSTM  295 685     14.6 M   0.784 +- 0.046        1.53 ms

Convolutional depth adds about one point of macro-F1 for 15 times the
operations; the recurrence adds six.

All models take input (B, C, T) and return logits (B, num_classes).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from iomtlab.config import N_CHANNELS, N_CLASSES


def _conv_stack(in_ch: int, n_filters: int, kernel: int, n_blocks: int) -> nn.Sequential:
    """n_blocks of Conv1d + ReLU with "same" padding, so T is unchanged."""
    layers: list[nn.Module] = []
    for i in range(n_blocks):
        # il primo blocco riceve i canali del sensore, gli altri le uscite dei filtri
        layers += [nn.Conv1d(in_ch if i == 0 else n_filters, n_filters, kernel,
                             padding="same"), nn.ReLU()]
    return nn.Sequential(*layers)


class DeepCNN(nn.Module):
    """Four convolutional blocks, global average pooling, a linear classifier.

    DeepConvLSTM without the recurrence: it separates the effect of depth
    from that of temporal memory.
    """

    def __init__(self, in_channels: int | None = None, num_classes: int | None = None,
                 n_filters: int = 64, kernel_size: int = 5, n_blocks: int = 4):
        super().__init__()
        from iomtlab.spec import active      # canali e classi del dataset attivo
        in_channels = in_channels or active().n_channels
        num_classes = num_classes or active().n_classes
        self.blocks = _conv_stack(in_channels, n_filters, kernel_size, n_blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # blocchi convolutivi, media su tutta la finestra, classificatore lineare
        return self.fc(self.pool(self.blocks(x)).flatten(1))

    def stages(self):
        """The forward as (name, callable) pairs, in order. See SimpleCNN.stages."""
        # ogni coppia Conv1d + ReLU diventa un confine: block1, block2, ...
        out = [(f"block{i}", b) for i, b in enumerate(self.blocks, 1)]
        out.append(("pool", lambda x: self.pool(x).flatten(1)))
        out.append(("fc", lambda x: self.fc(x)))
        return out


class DeepConvLSTM(nn.Module):
    """Four convolutional blocks followed by two LSTM layers.

    The reference architecture for wearable HAR (Ordonez and Roggen, 2016):
    the convolutions extract local features, the LSTM models how they evolve
    along the window. We classify from the output at the last time step.
    """

    def __init__(self, in_channels: int | None = None, num_classes: int | None = None,
                 n_filters: int = 64, kernel_size: int = 5,
                 hidden: int = 128, n_lstm: int = 2, n_blocks: int = 4):
        super().__init__()
        from iomtlab.spec import active      # canali e classi del dataset attivo
        in_channels = in_channels or active().n_channels
        num_classes = num_classes or active().n_classes
        self.blocks = _conv_stack(in_channels, n_filters, kernel_size, n_blocks)
        self.lstm = nn.LSTM(n_filters, hidden, num_layers=n_lstm, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # porto il tempo in seconda posizione: (B, T, F), come vuole l'LSTM
        z = self.blocks(x).permute(0, 2, 1)
        out, _ = self.lstm(z)
        return self.fc(out[:, -1])                   # classifico con l'uscita dell'ultimo passo


class EarlyExitCNN(nn.Module):
    """SimpleCNN with an early-exit head after the global pooling.

    The exit head is one linear layer, 325 parameters. On this model it
    reaches nearly the same macro-F1 as the full model, and without
    calibration a confidence threshold of 0.9 lets no sample exit (numbers in
    reference values of the labs). The early exit saves little here because the cost is
    not spread along the depth.
    """

    def __init__(self, in_channels: int | None = None, num_classes: int | None = None,
                 n_filters: int = 64, kernel_size: int = 10, hidden: int = 256):
        super().__init__()
        from iomtlab.spec import active      # canali e classi del dataset attivo
        in_channels = in_channels or active().n_channels
        num_classes = num_classes or active().n_classes
        self.conv = nn.Conv1d(in_channels, n_filters, kernel_size, padding="same")
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.exit_head = nn.Linear(n_filters, num_classes)
        self.fc1 = nn.Linear(n_filters, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Output of the global pooling, (B, n_filters): the input of both heads."""
        return self.pool(self.relu(self.conv(x))).flatten(1)

    def forward(self, x: torch.Tensor, return_exit: bool = False):
        """Logits of the full model; (exit_logits, logits) if return_exit."""
        z = self.features(x)                         # parte comune alle due teste
        logits = self.fc2(self.relu(self.fc1(z)))    # testa completa, due strati lineari
        if return_exit:
            # tutte e due le uscite dallo stesso passaggio: la testa anticipata non ricalcola z
            return self.exit_head(z), logits
        return logits

    def stages(self):
        """The forward as (name, callable) pairs.

        The "pool" boundary is also where the early exit sits, so a cut there
        and the exit coincide.
        """
        return [
            ("conv", lambda x: self.relu(self.conv(x))),
            ("pool", lambda x: self.pool(x).flatten(1)),
            ("fc1", lambda x: self.relu(self.fc1(x))),
            ("fc2", lambda x: self.fc2(x)),
        ]
