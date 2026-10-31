"""The contract of a model designed by a project, and its check.

A model goes into the chain of Labs 4-7 if:

1. forward takes (B, C, T) and returns logits (B, K), with C, T, K of the
   active dataset (spec.active());
2. it declares its stages with stages(): an ordered list of (name, function),
   each taking the output of the previous one, whose composition is the
   whole network. The boundaries between stages are the possible cuts of
   Lab 6 (edge.split_model, edge.cuts.cut_profile);
3. every layer with parameters is counted by models.cost: Conv1d, Linear,
   LSTM, GRU, or a module with a method macs_per_window(inp, out) -> int.
   Normalisations are not counted, by convention. Otherwise layer_profile,
   the four tests and the cuts underestimate the model;
4. it is registered with train.register_model(name, factory), so that
   fit_reference(name) trains it with the course recipe.

What works on any model that respects the contract: quantize_state and
int8_bytes (on Conv1d and Linear; the weights of LSTM and GRU stay in float32
and int8_bytes counts them at 4 bytes), prune_magnitude (Conv1d and Linear), train_student (distillation), four_tests,
split_model, cut_profile, run_split, train_exit, the calibration and the
sets of Lab 7. What does not: edge.compress.shrink_example_cnn, the
structured pruning of Lab 5, is written for ExampleCNN.

check_model(model) runs the checks and returns one row per check.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["check_model"]


def check_model(model: nn.Module, atol: float = 1e-5) -> list[tuple[str, str, str]]:
    """Check the contract on model; rows (status, what, detail), status PASS, AVVISO or FALLITO."""
    from iomtlab.edge.compress import dequantize_state, int8_bytes, quantize_state
    from iomtlab.edge.cuts import cut_profile
    from iomtlab.edge.split import split_model
    from iomtlab.models.cost import count_parameters, layer_profile, uncounted_modules
    from iomtlab.models.memory import peak_activation_values
    from iomtlab.spec import active
    S = active()
    rows = []
    model.eval()
    x = torch.randn(4, S.n_channels, S.window, generator=torch.Generator().manual_seed(0))
    try:
        with torch.no_grad():
            z = model(x)
    except Exception as e:                                   # noqa: BLE001
        return [("FALLITO", "forward su (B, C, T)", f"{type(e).__name__}: {e}")]
    ok = tuple(z.shape) == (4, S.n_classes)
    rows.append(("PASS" if ok else "FALLITO", "uscita (B, K)",
                 f"{tuple(z.shape)}, attesa (4, {S.n_classes})"))
    if not ok:
        return rows

    miss = uncounted_modules(model)
    prof = layer_profile(model)
    rows.append(("FALLITO" if miss else "PASS", "ogni strato con parametri è contato",
                 f"non contati: {', '.join(miss)} (serve macs_per_window)" if miss
                 else f"{count_parameters(model)} parametri, {prof.macs} MAC per finestra"))

    if not hasattr(model, "stages"):
        rows.append(("FALLITO", "stages() dichiarate", "senza fasi non c'è taglio (Lab 6)"))
    else:
        names = [n for n, _ in model.stages()]
        bad = []
        with torch.no_grad():
            for n in names[:-1]:
                h, t = split_model(model, n)
                if not torch.allclose(t(h(x)), z, atol=atol):
                    bad.append(n)
        rows.append(("FALLITO" if bad else "PASS", "testa + coda = rete a ogni confine",
                     f"diversa a: {', '.join(bad)}" if bad else f"{len(names) - 1} confini: {', '.join(names[:-1])}"))
        r = cut_profile(model)
        same = r[-1]["mac_head"] == prof.macs
        rows.append(("PASS" if same else "FALLITO", "cut_profile e layer_profile d'accordo",
                     f"{r[-1]['mac_head']} e {prof.macs} MAC"))

    try:
        q = dequantize_state(quantize_state(model, 8), model)
        with torch.no_grad():
            agree = float((q(x).argmax(1) == z.argmax(1)).float().mean())
        ib = int8_bytes(model)
        rows.append(("PASS", "int8 per canale",
                     f"{ib['total_B'] / 1024:.1f} KiB, picco attivazioni "
                     f"{peak_activation_values(model)} valori; stessa classe su {agree:.0%} di 4 finestre a caso"))
        if ib["other_B"]:
            rows.append(("AVVISO", "parametri che restano in float32",
                         f"{ib['other_B']} B su {ib['total_B']} (per esempio LSTM e GRU): quantize_state "
                         "quantizza solo Conv1d e Linear"))
    except Exception as e:                                   # noqa: BLE001
        rows.append(("AVVISO", "int8 per canale", f"{type(e).__name__}: {e}"))
    return rows
