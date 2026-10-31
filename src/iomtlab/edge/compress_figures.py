"""Figures of Lab 5: model bytes, bits against macro-F1, pruning, distillation, four tests.

Every function returns the matplotlib figure, accepts ax (axes for
plot_four_tests) and never calls show(), as in iomtlab.plots.
"""
from __future__ import annotations

import numpy as np

from iomtlab.plots import _axes

__all__ = ["plot_model_bytes", "plot_bits_f1", "plot_pruning", "plot_distillation",
           "plot_four_tests"]

BLU, ARANCIO, GRIGIO, ROSSO = "#4878A8", "#E07B39", "#A0A0A0", "#C00000"


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_model_bytes(sizes: dict, limits: dict | None = None, ax=None):
    """KiB of the model in several formats, as horizontal bars, with flash limits as lines.

    sizes is {label: bytes}; limits is {label: bytes}, for example the flash
    of a class-1 and a class-2 node.
    """
    fig, ax = _axes(ax, (6.0, 3.0))
    labels = list(sizes)
    kib = np.array([sizes[k] for k in labels], dtype=float) / 1024
    ax.barh(range(len(labels)), kib, color=[BLU if "float32" in k else ARANCIO for k in labels])
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    for i, v in enumerate(kib):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize="small")
    for name, b in (limits or {}).items():
        ax.axvline(b / 1024, color=GRIGIO, ls="--", lw=1)
        ax.text(b / 1024, len(labels) - 0.4, f" {name}", fontsize="small", color="#606060")
    ax.set_xlabel("KiB")
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_bits_f1(bits, f1, reference=None, ax=None):
    """Macro-F1 against the bits per weight; reference (float32) as a dashed line."""
    fig, ax = _axes(ax, (4.0, 3.0))
    ax.plot(bits, f1, "o-", color=ARANCIO)
    if reference is not None:
        ax.axhline(reference, color=GRIGIO, ls="--", lw=1, label=f"float32 {reference:.3f}")
        ax.legend(frameon=False, fontsize="small")
    ax.set_xlabel("bit per peso")
    ax.set_ylabel("macro-F1")
    ax.invert_xaxis()                        # da 8 bit verso 2: sempre meno informazione
    ax.set_xticks(sorted(bits))
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_pruning(amounts, curves: dict, reference=None, ax=None):
    """Macro-F1 against the share removed per layer.

    curves is {label: [macro-F1 per amount]}; labels ending in
    «riaddestrato» are drawn solid, the others dashed (no retraining).
    """
    fig, ax = _axes(ax, (6.0, 3.0))
    x = 100 * np.asarray(amounts, dtype=float)
    for label, v in curves.items():
        col = ARANCIO if label.startswith("strutturato") else BLU
        ax.plot(x, v, "o-" if label.endswith("riaddestrato") else "o--", color=col, label=label)
    if reference is not None:
        ax.axhline(reference, color=GRIGIO, ls=":", lw=1, label=f"rete intera {reference:.3f}")
    ax.set_xlabel("quota tolta per strato [%]")
    ax.set_ylabel("macro-F1")
    ax.legend(frameon=False, fontsize="small")
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_distillation(temperatures, mean, std, alone_mean, alone_std, teacher=None, ax=None):
    """Student macro-F1 with the teacher at each temperature, against the student alone (band).

    mean and std are over seeds, one value per temperature; the grey band is
    the student trained on the labels only, mean plus or minus std.
    """
    fig, ax = _axes(ax, (6.0, 3.0))
    T = np.asarray(temperatures, dtype=float)
    ax.axhspan(alone_mean - alone_std, alone_mean + alone_std, color="#E0E0E0",
               label=f"da solo {alone_mean:.3f}")
    ax.errorbar(T, mean, yerr=std, fmt="o-", color=ARANCIO, capsize=3, label="con il maestro")
    if teacher is not None:
        ax.axhline(teacher, color=GRIGIO, ls="--", lw=1, label=f"maestro {teacher:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(T, [f"{t:g}" for t in T])
    ax.set_xlabel("temperatura T")
    ax.set_ylabel("macro-F1 dello studente")
    ax.legend(frameon=False, fontsize="small")
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_four_tests(rows, node: dict, axes=None):
    """The four tests, one panel each: KiB of flash, ms per window, days, nodes per channel.

    rows are dicts with label, model_B and the keys of four_tests. Each bar
    carries its value; a bar is red where the version fails the test. The
    limit of the node is dashed (tests 3 and 4 only when node gives min_days
    or min_nodes).
    """
    import matplotlib.pyplot as plt
    from iomtlab.edge.compress import NODE
    n = {**NODE, **node}
    if axes is None:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    fig = axes[0].figure
    labels = [r["label"] for r in rows]
    tests = [("1. entra", "KiB di flash", [r["model_B"] / 1024 for r in rows],
              [r["flash_ok"] for r in rows], n["flash_B"] / 1024, "{:.1f}"),
             ("2. veloce", "ms per finestra", [1e3 * r["t_s"] for r in rows],
              [r["fast"] for r in rows], 1e3 * n["deadline_s"], "{:.1f}"),
             ("3. batteria", "giorni", [r["days"] for r in rows],
              [r["battery_ok"] for r in rows], n["min_days"], "{:.0f}"),
             ("4. canale", "nodi per canale", [r["nodes"] for r in rows],
              [r["channel_ok"] for r in rows], n["min_nodes"], "{:.0f}")]
    for k, (ax, (title, xlabel, v, ok, lim, f)) in enumerate(zip(axes, tests)):
        ax.barh(range(len(v)), v, color=["#0070C0" if o else ROSSO for o in ok], edgecolor="#202020",
                lw=0.6)
        for i, val in enumerate(v):
            txt = f.format(val)
            ax.text(val, i, "  " + (txt if val >= 10 or f == "{:.0f}" else txt), va="center",
                    fontsize="small")
        ax.set_yticks(range(len(v)), labels if k == 0 else [""] * len(v))
        ax.tick_params(axis="y", length=0 if k else 3)
        ax.invert_yaxis()
        if lim is not None:
            ax.axvline(lim, color="#C05030", ls="--", lw=1.2)
        top = max([lim or 0] + list(v))
        ax.set_xlim(0, 1.45 * top if top else 1)
        ax.set_title(title, fontsize="medium")
        ax.set_xlabel(xlabel)
        _clean(ax)
        if k:
            ax.spines["left"].set_visible(True)
    fig.tight_layout()
    return fig
