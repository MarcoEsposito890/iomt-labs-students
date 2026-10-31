"""Figures of the evaluation: validations, confusion matrices, losses, confidence (Labs 3, 4).

Every function returns the matplotlib figure, accepts ax and never calls
show(), as in iomtlab.plots.
"""
from __future__ import annotations

import numpy as np

from iomtlab.config import LABEL_NAMES
from iomtlab.plots import PHASE_SHORT, _axes

__all__ = ["plot_validation", "plot_confusion", "plot_losses", "plot_confidence"]


def plot_validation(entries, ax=None, ylabel: str = "macro-F1"):
    """One bar per validation scheme: the mean over folds, with each fold as a point.

    entries is a sequence of (label, scores), scores one value per fold.
    The mean is written above each bar as «media 0.xxx».
    """
    fig, ax = _axes(ax, (5.5, 3.4))
    rng = np.random.default_rng(0)            # solo per spostare i punti di lato, sempre uguale
    for i, (lab, sc) in enumerate(entries):
        sc = np.asarray(sc, dtype=float)
        ax.bar(i, sc.mean(), width=0.6, color="#DEEBF7", edgecolor="#1F4E79", zorder=2)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(sc)), sc, s=14, color="#1F4E79", zorder=3)
        ax.text(i, sc.max() + 0.01, f"media {sc.mean():.3f}", ha="center", va="bottom")
    lo = min(np.min(sc) for _, sc in entries)
    ax.set_ylim(max(0.0, lo - 0.08), 1.0)
    ax.set_xticks(range(len(entries)), [e[0] for e in entries])
    ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_confusion(report, ax=None, title=None):
    """The confusion matrix of evaluation.confusion_report, with the value in each cell.

    With normalize "row" the diagonal is the recall, with "col" the
    precision; without normalisation the cells are counts. Rows are the true
    class, columns the predicted one.
    """
    M = np.asarray(report["normalized"], dtype=float)
    norm = report.get("normalize")
    from iomtlab.spec import active, short_label
    names = [short_label(n) for n in report.get("names", active().classes)]
    fig, ax = _axes(ax, (4.6, 4.0))
    vmax = 1.0 if norm else M.max()
    ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}" if norm else f"{int(v)}", ha="center", va="center",
                    color="white" if v > 0.6 * vmax else "black", fontsize="small")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("classe predetta")
    ax.set_ylabel("classe vera")
    ax.set_title(title or {"row": "per riga: richiamo sulla diagonale",
                           "col": "per colonna: precisione sulla diagonale"}.get(norm, "conteggi"))
    fig.tight_layout()
    return fig


def plot_losses(histories, ax=None):
    """Training loss per epoch, one line per network; histories is {name: [loss per epoch]}.

    The dotted line is ln 5: the loss of a network that gives the same
    probability to the five classes, where every training should start.
    """
    fig, ax = _axes(ax, (5.5, 3.4))
    for name, h in histories.items():
        ax.plot(range(1, len(h) + 1), h, marker="o", ms=3, lw=1.2, label=name)
    ax.axhline(np.log(5), color="#A0A0A0", ls=":", lw=1)
    ax.set_xlabel("epoca")
    ax.set_ylabel("loss di addestramento")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_confidence(conf, correct, ax=None, threshold=None, bins: int = 20):
    """Histogram of the confidence (largest softmax probability), right and wrong decisions apart.

    The vertical axis is logarithmic: the wrong decisions are few. threshold,
    if given, is drawn as a vertical line.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    fig, ax = _axes(ax, (5.5, 3.4))
    edges = np.linspace(conf.min() if len(conf) else 0.2, 1.0, bins + 1)
    ax.hist(conf[correct], bins=edges, color="#4878A8", alpha=0.8,
            label=f"giuste ({int(correct.sum())})")
    ax.hist(conf[~correct], bins=edges, color="#C05030", alpha=0.8,
            label=f"sbagliate ({int((~correct).sum())})")
    if threshold is not None:
        ax.axvline(threshold, color="black", ls="--", lw=1, label=f"soglia {threshold:g}")
    ax.set_yscale("log")
    ax.set_xlabel("confidenza (probabilità massima della softmax)")
    ax.set_ylabel("finestre")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
