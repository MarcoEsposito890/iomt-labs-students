"""The reference classifier, read as a filter bank.

    Conv1d(6, 64, kernel_size=10)
        a bank of 64 FIR filters of order 9, each over the 6 input channels.
        The coefficients are learned instead of designed.

    ReLU
        half-wave rectification of each filter output.

    AdaptiveAvgPool1d(1)
        mean of the rectified output over the window: the DC component, the
        energy picked up by that filter.

    Linear(64 -> 256) -> ReLU -> Linear(256 -> 5)
        classifier on the 64 energies.

21 829 parameters, 87 316 bytes in float32. frequency_response returns the
magnitude response of the learned filters.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from iomtlab.config import N_CHANNELS, N_CLASSES, TARGET_HZ


class SimpleCNN(nn.Module):
    """Learned filter bank and classifier. Input (B, C, T), output logits (B, K)."""

    def __init__(self, in_channels: int | None = None, num_classes: int | None = None,
                 n_filters: int = 64, kernel_size: int = 10,
                 hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        from iomtlab.spec import active      # canali, classi e campioni del dataset attivo
        S = active()
        in_channels = in_channels or S.n_channels
        num_classes = num_classes or S.n_classes
        self.conv = nn.Conv1d(in_channels, n_filters, kernel_size, padding="same")
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc1 = nn.Linear(n_filters, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))          # banco di filtri e rettificazione: (B, 64, T)
        x = self.pool(x).flatten(1)          # media su tutta la finestra: un'energia per filtro, (B, 64)
        x = self.drop(self.relu(self.fc1(x)))  # il classificatore sulle 64 energie
        return self.fc2(x)                   # un logit per classe

    @torch.no_grad()
    def frequency_response(self, n_fft: int = 256, fs: float = TARGET_HZ
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Magnitude response of the learned filters, as (freqs, H).

        freqs is (n_fft//2+1,) in Hz, H is (n_filters, n_channels, n_fft//2+1).
        After training the filters concentrate on the bands where the gait
        energy is.
        """
        w = self.conv.weight.detach().cpu().numpy()  # i coefficienti dei filtri: (F, C, K)
        # la risposta in ampiezza è il modulo della DFT dei coefficienti
        H = np.abs(np.fft.rfft(w, n=n_fft, axis=-1))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)   # asse delle frequenze in Hz
        return freqs, H

    def stages(self):
        """The forward as (name, callable) pairs, in order; composed they give forward.

        edge.split_model cuts only at these boundaries. forward contains a
        flatten(1) that is not a module, so nn.Sequential(*children) would
        give wrong pieces.
        """
        return [
            ("conv", lambda x: self.relu(self.conv(x))),
            ("pool", lambda x: self.pool(x).flatten(1)),
            ("fc1", lambda x: self.drop(self.relu(self.fc1(x)))),
            ("fc2", lambda x: self.fc2(x)),
        ]
