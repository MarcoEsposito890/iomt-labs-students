"""Figures of Lab 8: class mix of the clients, cost of a round, consensus.

Every function returns the matplotlib figure, accepts ax and never calls
show(), as in iomtlab.plots. The rounds themselves are drawn by
plots.plot_rounds.
"""
from __future__ import annotations

import numpy as np

from iomtlab.config import LABEL_NAMES
from iomtlab.plots import PHASE_SHORT, _axes

__all__ = ["plot_client_mix", "plot_round_cost", "plot_consensus"]

BLU, ARANCIO, GRIGIO = "#4878A8", "#E07B39", "#A0A0A0"
CLASSI = ["#BDBDBD", "#F4B183", "#4878A8", "#9DC3E6", "#C55A11"]   # una per classe, in ordine


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_client_mix(parts, y, ax=None, title=None, names=None, legend: bool = True):
    """Share of each class in each client, one stacked column per client.

    parts is {client: indices} (subject_clients) or a list of index arrays
    (dirichlet_partition); y the labels of the rows the indices point to.
    """
    fig, ax = _axes(ax, (7.0, 3.0))
    y = np.asarray(y)
    items = list(parts.values()) if isinstance(parts, dict) else list(parts)
    from iomtlab.spec import active, short_label
    names = list(names or active().classes)
    K = len(names)
    share = np.array([np.bincount(y[p], minlength=K)[:K] / max(1, len(p)) for p in items])
    bottom = np.zeros(len(items))
    for k in range(K):
        ax.bar(np.arange(len(items)), share[:, k], bottom=bottom, color=CLASSI[k % len(CLASSI)],
               width=0.85, label=short_label(names[k]))
        bottom += share[:, k]
    ax.set_xticks([])
    ax.set_xlabel(f"{len(items)} client")
    ax.set_ylabel("quota delle finestre")
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title, fontsize="medium")
    if legend:
        ax.legend(frameon=False, fontsize="x-small", ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.1))
    _clean(ax)
    return fig


def plot_round_cost(costs: dict, ax=None):
    """Energy of one client in one round: radio (orange) and local training (blue), J.

    costs is {label: dict of fl.round_cost}.
    """
    fig, ax = _axes(ax, (6.0, 2.6))
    labels = list(costs)
    r = np.array([costs[k]["radio_J"] for k in labels])
    t = np.array([costs[k]["train_J"] for k in labels])
    yy = np.arange(len(labels))
    ax.barh(yy, r, color=ARANCIO, label="radio, giù e su")
    ax.barh(yy, t, left=r, color=BLU, label="addestramento locale")
    for i in range(len(labels)):
        ax.text(r[i] + t[i], i, f" {r[i] + t[i]:.2f} J", va="center", fontsize="small")
    ax.set_yticks(yy, labels)
    ax.invert_yaxis()
    ax.set_xlabel("J per client e per round")
    ax.set_xlim(0, 1.3 * float((r + t).max()))          # spazio per i valori a destra delle barre
    ax.legend(frameon=False, fontsize="small", loc="lower right")
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_consensus(errors: dict, rho: dict | None = None, ax=None, tol: float = 0.01):
    """Distance from the true mean against the iterations, log scale; dashed the line ρ^k.

    errors is {label: disagreement array}; rho is {label: ρ} for the labels
    whose ρ^k line is drawn; tol a dotted horizontal line.
    """
    fig, ax = _axes(ax, (6.0, 3.0))
    cols = iter((ARANCIO, BLU, "#5B9B5B", "#C00000"))
    for k, e in errors.items():
        c = next(cols)
        e = np.asarray(e)
        ax.semilogy(np.arange(len(e)), e, color=c, label=k)
        if rho and k in rho:
            ax.semilogy(np.arange(len(e)), rho[k] ** np.arange(len(e)), "--", color=c, lw=1)
    ax.axhline(tol, color=GRIGIO, ls=":", lw=1)
    ax.set_xlabel("iterazione k")
    ax.set_ylabel("distanza dalla media vera")
    ax.set_ylim(1e-4, 2)
    ax.legend(frameon=False, fontsize="small")
    _clean(ax)
    return fig
