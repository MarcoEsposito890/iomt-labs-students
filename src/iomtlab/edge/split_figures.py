"""Figures of Lab 6: bits at each cut, energy at each cut, the bottleneck sweep, the exit curve.

Every function returns the matplotlib figure, accepts ax or axes and never
calls show(), as in iomtlab.plots. The rows are those of edge.cuts.cut_profile,
edge.cut_cost and edge.cuts.cut_delay.
"""
from __future__ import annotations

import numpy as np

from iomtlab.plots import _axes

__all__ = ["plot_cut_bits", "plot_cut_energy", "plot_bottleneck_sweep", "plot_exit_curve",
           "plot_mz", "plot_regimes"]

BLU, ARANCIO, GRIGIO, ROSSO = "#4878A8", "#E07B39", "#A0A0A0", "#C00000"


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _axes_list(axes, n, figsize):
    """n axes: the ones given (one axis or an array) or a new figure with n panels."""
    import matplotlib.pyplot as plt
    if axes is None:
        fig, axs = plt.subplots(1, n, figsize=figsize, squeeze=False)
        return fig, list(axs[0])
    axs = list(np.atleast_1d(axes).ravel())
    if len(axs) < n:
        raise ValueError(f"servono {n} assi, ne ho ricevuti {len(axs)}")
    return axs[0].figure, axs[:n]


def plot_cut_bits(rows8, rows32=None, ax=None):
    """Bits per window at each cut, log scale: int8 activations, and float32 if given.

    rows8 and rows32 are cut_profile rows with act_bits 8 and 32; the dashed
    line is the raw window (the first row).
    """
    fig, ax = _axes(ax, (7.0, 3.2))
    names = [r["name"] for r in rows8]
    x = np.arange(len(names))
    if rows32 is not None:
        ax.plot(x, [r["bits"] for r in rows32], "s--", color=GRIGIO, label="attivazioni float32")
    ax.plot(x, [r["bits"] for r in rows8], "o-", color=ARANCIO, label="attivazioni int8")
    ax.axhline(rows8[0]["bits"], color=BLU, ls="--", lw=1,
               label=f"finestra grezza, {rows8[0]['bits']} bit")
    ax.set_yscale("log")
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylabel("bit per finestra")
    ax.legend(frameon=False, fontsize="small")
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_cut_energy(costs: dict, axes=None):
    """Energy per window at each cut, one panel per link: computation (blue) and radio (orange).

    costs is {title: rows of cut_cost}. The bar of the minimum has a red
    edge and its value in mJ above it, in red.
    """
    fig, axs = _axes_list(axes, len(costs), (6.0 * len(costs), 3.2))
    for ax, (title, rows) in zip(axs, costs.items()):
        names = [r["name"] for r in rows]
        x = np.arange(len(rows))
        en = np.array([r["e_node_J"] for r in rows]) * 1e3
        el = np.array([r["e_link_J"] for r in rows]) * 1e3
        ax.bar(x, en, color=BLU, label="calcolo sul nodo")
        ax.bar(x, el, bottom=en, color=ARANCIO, label="radio")
        k = int(np.argmin(en + el))                 # il taglio di energia minima
        ax.bar([k], [en[k] + el[k]], fill=False, edgecolor=ROSSO, lw=2)
        ax.text(k, (en[k] + el[k]) * 1.04, f"{en[k] + el[k]:.3f}", ha="center", va="bottom",
                color=ROSSO, fontsize="small")
        ax.set_xticks(x, names, rotation=30, ha="right")
        ax.set_ylabel("mJ per finestra")
        ax.set_title(title, fontsize="medium")
        ax.legend(frameon=False, fontsize="small")
        _clean(ax)
    fig.tight_layout()
    return fig


def plot_bottleneck_sweep(trials, e_all_node_J=None, axes=None):
    """The bottleneck sweep: energy at the cut and macro-F1 against the channels at the cut.

    trials are dicts with channels, e_tot_J and macro_f1, one per network;
    e_all_node_J (everything on the node) is drawn as a dashed line.
    """
    fig, (a, b) = _axes_list(axes, 2, (10.0, 3.2))
    ch = [t["channels"] for t in trials]
    a.plot(ch, [1e3 * t["e_tot_J"] for t in trials], "o-", color=ARANCIO, label="taglio al collo")
    if e_all_node_J is not None:
        a.axhline(1e3 * e_all_node_J, color=BLU, ls="--", lw=1,
                  label=f"tutto sul nodo, {1e3 * e_all_node_J:.3f} mJ")
    a.set_ylabel("mJ per finestra")
    b.plot(ch, [t["macro_f1"] for t in trials], "o-", color=BLU)
    b.set_ylabel("macro-F1")
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ch, [str(c) for c in ch])
        ax.set_xlabel("canali al collo")
        _clean(ax)
    a.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    return fig


def plot_exit_curve(curve, e_all_node_J=None, e_all_server_J=None, axes=None):
    """The exit curve: q and macro-F1 against θ (left), energy against q (right).

    curve are the rows of exit_curve; the two ends, everything on the node
    and everything on the server, are dashed lines on the right.
    """
    fig, (a, b) = _axes_list(axes, 2, (10.0, 3.2))
    th = [r["theta"] for r in curve]
    q = np.array([r["q"] for r in curve])
    a.plot(th, q, "o-", color=ARANCIO, label="q, decise sul nodo")
    a.plot(th, [r["macro_f1"] for r in curve], "s-", color=BLU, label="macro-F1")
    a.set_xlabel("soglia θ")
    a.set_ylim(0, 1.05)
    a.legend(frameon=False, fontsize="small")
    b.plot(q, [1e3 * r["energy_J"] for r in curve], "o-", color=ARANCIO, label="con l'uscita")
    for e, lab, col in ((e_all_node_J, "tutto sul nodo", BLU),
                        (e_all_server_J, "tutto sul server", GRIGIO)):
        if e is not None:
            b.axhline(1e3 * e, color=col, ls="--", lw=1, label=f"{lab}, {1e3 * e:.3f} mJ")
    b.set_xlabel("q, quota decisa sul nodo")
    b.set_ylabel("mJ per decisione")
    b.legend(frameon=False, fontsize="small")
    for ax in (a, b):
        _clean(ax)
    fig.tight_layout()
    return fig


def plot_mz(rows, ax=None, label_ends: bool = True):
    """Every cut on the plane z (bits over the raw end) by m (share of MACs on the node).

    The diagonal m + z = 1 joins the two ends: only a cut below it can be the
    cheapest for some k = ρh (edge.regimes.rule). z is clipped at 3 for the
    figure; the true value is in the label.
    """
    from iomtlab.edge.regimes import rule
    fig, ax = _axes(ax, (5.0, 4.0))
    ax.fill_between([0, 1], [1, 0], 0, color=ARANCIO, alpha=0.12, lw=0)
    ax.plot([0, 1], [1, 0], color=GRIGIO, lw=1, ls="--")
    ax.text(0.04, 0.06, "m + z < 1: può vincere", color=ARANCIO, fontsize="small")
    for i, x in enumerate(rule(rows)):
        end = i in (0, len(rows) - 1)
        if end and not label_ends:
            continue
        z = min(x["z"], 3.0)
        c = BLU if end else (ARANCIO if x["can_win"] else GRIGIO)
        ax.plot(z, x["m"], "o", color=c)
        ax.annotate(f"{x['name']}" + ("" if x["z"] <= 3 else f" (z={x['z']:.1f})"), (z, x["m"]),
                    textcoords="offset points", xytext=(4, 3), fontsize="small", color=c)
    ax.set_xlim(-0.05, 3.1)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("z = bit al taglio / bit dell'estremo grezzo")
    ax.set_ylabel("m = quota dei MAC sul nodo")
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_regimes(rows, links: dict | None = None, exclusions=((),), k_range=(1.0, 1e4), ax=None):
    """Which cut wins along k = ρh (MAC per bit, log scale), one band per set of exclusions.

    links maps a label to its k (e_bit / e_MAC times hops), drawn as vertical
    lines. exclusions as in edge.regimes.regime_map: () nothing excluded,
    ("classe",) a model that does not fit the node, ...
    """
    from iomtlab.edge.regimes import envelope
    import matplotlib.pyplot as plt
    fig, ax = _axes(ax, (7.0, 0.9 + 0.55 * len(exclusions)))
    names = [r["name"] for r in rows]
    cmap = plt.get_cmap("tab10")
    color = {n: (BLU if n == names[0] else ARANCIO if n == names[-1] else cmap(2 + i % 7))
             for i, n in enumerate(names)}
    lo, hi = k_range
    for j, ex in enumerate(exclusions):
        for s in envelope(rows, exclude=ex, k_max=hi * 10):
            a, b = max(s["k_from"], lo), min(s["k_to"], hi)
            if b <= a:
                continue
            ax.barh(j, b - a, left=a, height=0.6, color=color[s["name"]], alpha=0.8)
            ax.text((a * b) ** 0.5, j, s["name"], ha="center", va="center", fontsize="small")
    for lab, k in (links or {}).items():
        ax.axvline(k, color=ROSSO, lw=1)
        ax.text(k, len(exclusions) - 0.45, lab, rotation=90, va="bottom", ha="right",
                fontsize="x-small", color=ROSSO)
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_yticks(range(len(exclusions)),
                  ["nessuna esclusione" if not ex else "senza " + ", ".join(ex) for ex in exclusions])
    ax.set_xlabel("k = ρh: MAC che costa un bit, per i salti")
    _clean(ax)
    fig.tight_layout()
    return fig
