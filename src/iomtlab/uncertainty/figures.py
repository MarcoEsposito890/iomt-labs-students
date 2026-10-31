"""Figures of Lab 7: calibration before and after T, coverage of the sets, routing.

Every function returns the matplotlib figure, accepts axes and never calls
show(), as in iomtlab.plots. tab is the DataFrame of uncertainty.routing_table.
"""
from __future__ import annotations

import numpy as np

__all__ = ["plot_calibration_pair", "plot_coverage", "plot_routing"]

BLU, ARANCIO, GRIGIO, ROSSO, VERDE = "#4878A8", "#E07B39", "#A0A0A0", "#C00000", "#5B9B5B"


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _two(axes, figsize=(10.0, 3.6)):
    import matplotlib.pyplot as plt
    if axes is None:
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        return fig, list(axs)
    axs = list(np.atleast_1d(axes).ravel())
    if len(axs) < 2:
        raise ValueError("servono due assi")
    return axs[0].figure, axs[:2]


def plot_calibration_pair(p_before, p_after, y, n_bins: int = 15, axes=None,
                          titles=("senza temperatura", "con la temperatura")):
    """Two reliability diagrams of the same windows: probabilities p_before and p_after (n, K)."""
    from iomtlab.plots import reliability_diagram
    from iomtlab.uncertainty import ece
    fig, (a, b) = _two(axes)
    y = np.asarray(y)
    for ax, p, t in ((a, p_before, titles[0]), (b, p_after, titles[1])):
        p = np.asarray(p)
        conf, ok = p.max(1), (p.argmax(1) == y)
        reliability_diagram(conf, ok.astype(float), n_bins, ax=ax)
        ax.set_title(f"{t}: ECE {ece(conf, ok, n_bins):.3f}", fontsize="medium")
    fig.tight_layout()
    return fig


def plot_coverage(tab, axes=None):
    """Left: measured coverage against the promise 1 - alpha. Right: set size, singletons, empty sets.

    Set size is on the left axis of the right panel (classes per set); the
    shares of singletons and of empty sets on its right axis.
    """
    fig, (a, b) = _two(axes)
    prom = 1 - tab["alpha"].to_numpy()
    a.plot(prom, prom, "--", color=GRIGIO, lw=1, label="promessa 1 − α")
    a.plot(prom, tab["coverage"], "o-", color=BLU, label="copertura misurata")
    a.set_xlabel("1 − α")
    a.set_ylabel("copertura")
    a.legend(frameon=False, fontsize="small")
    al = tab["alpha"].to_numpy()
    b.plot(al, tab["mean_size"], "o-", color=BLU, label="classi per insieme")
    b.set_xlabel("α")
    b.set_ylabel("classi per insieme")
    b2 = b.twinx()
    b2.plot(al, tab["q"], "s-", color=ARANCIO, label="singoletti")
    b2.plot(al, tab["empty"], "^-", color=ROSSO, label="vuoti")
    b2.set_ylim(0, 1)
    b2.set_ylabel("quota delle finestre")
    h1, l1 = b.get_legend_handles_labels()
    h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, frameon=False, fontsize="small", loc="center right")
    for ax in (a, b):
        _clean(ax)
    b2.spines["top"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_routing(tab, energies: dict, axes=None):
    """Left: share decided on the node and accuracies against alpha. Right: energy per decision.

    energies is {label: energy per alpha, in J}; a key starting with «_» is
    a constant drawn as a dashed line (for example "_all_local", everything
    on the node).
    """
    fig, (a, b) = _two(axes)
    al = tab["alpha"].to_numpy()
    a.plot(al, tab["q"], "o-", color=ARANCIO, label="q, decise sul nodo")
    a.plot(al, tab["acc_local"], "s-", color=BLU, label="accuratezza sul nodo")
    if "acc_system" in tab:
        a.plot(al, tab["acc_system"], "^-", color=VERDE, label="accuratezza del sistema")
    a.axhline(0.9, color=GRIGIO, ls=":", lw=1)
    a.set_xlabel("α")
    a.set_ylim(0, 1.02)
    a.legend(frameon=False, fontsize="small")
    names = {"A": "A: rete intera, finestra se inoltra", "B": "B: testa, attivazione se inoltra",
             "_all_local": "tutto sul nodo"}
    cols = iter((ARANCIO, BLU, VERDE, ROSSO))
    for k, e in energies.items():
        lab = names.get(k, k.lstrip("_"))
        if k.startswith("_"):
            b.axhline(1e3 * float(e), color=GRIGIO, ls="--", lw=1, label=f"{lab}, {1e3 * float(e):.3f} mJ")
        else:
            b.plot(al, 1e3 * np.asarray(e, dtype=float), "o-", color=next(cols), label=lab)
    b.set_xlabel("α")
    b.set_ylabel("mJ per decisione")
    b.legend(frameon=False, fontsize="small")
    for ax in (a, b):
        _clean(ax)
    fig.tight_layout()
    return fig
