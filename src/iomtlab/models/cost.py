"""Cost of a model, layer by layer.

MACs (multiply-accumulate operations) per window, as in the offloading
literature:

    linear layer          o(l) = D * G
    convolutional layer   o(l) = C_in * C_out * K * T
    LSTM layer            o(l) = 4 * (C_in * H + H * H) * T   (four gates per step)
    GRU layer             o(l) = 3 * (C_in * H + H * H) * T   (three gates per step)

T is the length of the output activation; a bidirectional recurrent layer
counts twice. Bias, normalisations and activations are not counted. A layer
of another kind (attention, a custom block) is counted only if it declares
its MACs with a method macs_per_window(inp, out) -> int; uncounted_modules
lists the layers with parameters that are neither counted nor normalisations,
so that a model cannot look cheaper than it is. Three different quantities come out:

    parameters     size of the model at rest: memory, and the payload of a
                   federated update
    MACs           cost of one inference: time and energy
    activation     values that cross a cut point
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from iomtlab.config import N_CHANNELS, WINDOW_SIZE


#: Layers with parameters and no MACs by convention (block 3.1): normalisations.
NORMS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d)


def module_macs(mod: nn.Module, inp, out):
    """(MACs, kind) of one call of mod with input inp and output out; None if mod is not counted.

    Conv1d, Linear, LSTM, GRU, and any module with a macs_per_window(inp, out)
    method. Shared by layer_profile and edge.cuts.cut_profile, so the two
    always agree.
    """
    if hasattr(mod, "macs_per_window"):
        return int(mod.macs_per_window(inp, out)), "custom"
    if isinstance(mod, nn.Conv1d):
        t = out.shape[-1]                    # T è la lunghezza dell'uscita
        # C_in * C_out * K * T, con C_in diviso per i gruppi della convoluzione
        return (mod.in_channels // mod.groups) * mod.out_channels * mod.kernel_size[0] * t, "conv1d"
    if isinstance(mod, nn.Linear):
        o = out[0] if isinstance(out, tuple) else out
        rows = int(o[0].numel() // mod.out_features)   # un Linear applicato a ogni passo conta T volte
        return mod.in_features * mod.out_features * max(rows, 1), "linear"
    if isinstance(mod, (nn.LSTM, nn.GRU)):
        gates = 4 if isinstance(mod, nn.LSTM) else 3
        seq = out[0]
        t = seq.shape[1] if mod.batch_first else seq.shape[0]   # passi temporali
        d = 2 if mod.bidirectional else 1
        macs = 0
        for lay in range(mod.num_layers):
            # il primo strato riceve l'ingresso, i successivi l'uscita del precedente
            i = mod.input_size if lay == 0 else mod.hidden_size * d
            macs += d * gates * (i * mod.hidden_size + mod.hidden_size ** 2) * t
        return macs, "lstm" if gates == 4 else "gru"
    return None


def counted_modules(model: nn.Module):
    """(name, module) of the modules whose MACs are counted, in definition order.

    The children of a module with macs_per_window are skipped: its method
    counts them already.
    """
    inside = []                              # prefissi dei blocchi che dichiarano i loro MAC
    for name, mod in model.named_modules():  # i genitori vengono prima dei figli
        if any(name.startswith(p) for p in inside):
            continue
        if hasattr(mod, "macs_per_window"):
            inside.append(name + "." if name else "")
            yield name, mod
        elif isinstance(mod, (nn.Conv1d, nn.Linear, nn.LSTM, nn.GRU)):
            yield name, mod


def uncounted_modules(model: nn.Module) -> list[str]:
    """Names of the modules with their own parameters whose MACs nobody counts.

    Normalisations (NORMS) are excluded by convention. A non-empty list means
    that layer_profile underestimates the model: the layer needs a
    macs_per_window method.
    """
    counted = dict(counted_modules(model))
    inside = [n + "." if n else "" for n, m in counted.items() if hasattr(m, "macs_per_window")]
    out = []
    for name, mod in model.named_modules():
        if name in counted or isinstance(mod, NORMS) or any(name.startswith(p) for p in inside):
            continue
        if sum(q.numel() for q in mod.parameters(recurse=False)):
            out.append(name or "root")
    return out


@dataclass
class LayerCost:
    """Cost of one layer: parameters, MACs and output values for one window."""
    name: str
    kind: str
    parameters: int
    macs: int
    out_elements: int          # valori dell'attivazione in uscita, per una finestra

    def out_bytes(self, dtype_bytes: int = 4) -> int:
        """Bytes of the output activation with dtype_bytes per value."""
        return self.out_elements * dtype_bytes


@dataclass
class ModelCost:
    """Cost profile of a model: the list of its profiled layers, in forward order."""
    layers: list[LayerCost] = field(default_factory=list)

    @property
    def parameters(self) -> int:
        """Parameters of the profiled layers only.

        Normalisation layers have parameters but no MACs and are not
        profiled, so this can be less than count_parameters(model).
        """
        return sum(l.parameters for l in self.layers)

    @property
    def macs(self) -> int:
        """Total MACs for one window."""
        return sum(l.macs for l in self.layers)

    def table(self) -> str:
        """The profile as a text table, one row per layer plus the total."""
        w = max(len(l.name) for l in self.layers) + 2
        out = [f"{'strato':{w}s}{'param':>10s}{'MAC':>12s}{'uscita':>10s}{'byte fp32':>11s}"]
        for l in self.layers:
            out.append(f"{l.name:{w}s}{l.parameters:10,d}{l.macs:12,d}"
                       f"{l.out_elements:10,d}{l.out_bytes():11,d}")
        out.append(f"{'TOTALE':{w}s}{self.parameters:10,d}{self.macs:12,d}")
        return "\n".join(out)

    def to_dict(self) -> dict:
        """The profile as a JSON-serialisable dict."""
        import dataclasses
        return {"layers": [dataclasses.asdict(l) for l in self.layers],
                "total_macs": int(self.macs),
                "total_params": int(self.parameters)}

    def to_json(self, path):
        """Write the profile to path (.json) and return the path."""
        import json
        from pathlib import Path
        p = Path(path).with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1))
        return p


def layer_profile(model: nn.Module, in_channels: int | None = None,
                  seq_len: int | None = None) -> ModelCost:
    """Cost profile (ModelCost) from one instrumented forward pass on a zero window.

    PyTorch forward hooks record every Conv1d, Linear, LSTM, ReLU, pooling and
    dropout layer as it runs, so the activation sizes are the real ones even
    if the architecture changes. The input is (1, in_channels, seq_len).
    """
    from iomtlab.spec import active          # canali e campioni del dataset attivo
    in_channels = in_channels or active().n_channels
    seq_len = seq_len or active().window
    cost = ModelCost()
    handles = []

    def hook(name):
        def fn(mod, inp, out):
            # parametri propri del modulo, senza quelli dei sottomoduli
            p = sum(q.numel() for q in mod.parameters(recurse=False))
            r = module_macs(mod, inp, out) if mod in counted_set else None
            macs, kind = r if r is not None else (0, mod.__class__.__name__.lower())
            o = out[0] if isinstance(out, tuple) else out        # LSTM e GRU restituiscono una tupla
            cost.layers.append(LayerCost(name, kind, p, macs, int(o[0].numel())))
        return fn

    # aggancio un hook a ogni modulo che voglio nel profilo: quelli contati, più ReLU,
    # pooling e dropout per le dimensioni delle attivazioni
    counted = dict(counted_modules(model))
    counted_set = set(counted.values())
    for name, mod in model.named_modules():
        if name in counted or isinstance(mod, (nn.ReLU, nn.AdaptiveAvgPool1d, nn.MaxPool1d, nn.Dropout)):
            handles.append(mod.register_forward_hook(hook(name or "root")))

    model.eval()
    with torch.no_grad():
        # per le dimensioni basta un passaggio su una finestra di zeri
        model(torch.zeros(1, in_channels, seq_len))
    for h in handles:
        h.remove()                           # tolgo gli hook: il modello torna com'era
    return cost


def count_parameters(model: nn.Module) -> int:
    """Number of parameters of model, normalisation layers included."""
    return sum(p.numel() for p in model.parameters())


def model_size_bytes(model: nn.Module, dtype_bytes: int = 4) -> int:
    """Bytes of the parameters at dtype_bytes per value: 4 for float32, 1 to estimate int8.

    An estimate: the file on disk also holds a header (edge.size_on_disk).
    """
    return count_parameters(model) * dtype_bytes


def federated_round_bytes(model: nn.Module, n_clients: int,
                          dtype_bytes: int = 4) -> int:
    """Bytes of one federated round: the model down and the update up, for every client."""
    # ogni client riceve il modello e rimanda l'aggiornamento: due volte i pesi
    return 2 * n_clients * count_parameters(model) * dtype_bytes
