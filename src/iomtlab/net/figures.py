"""The figures of Lab 1.

Each function takes what a function of iomtlab.net.measure returned, draws
it and returns the figure; none calls show() or writes to disk. Each accepts
ax (or axes) so the lab slides can draw the same figure at their own size.
"""
from __future__ import annotations

import numpy as np

from iomtlab.plots import _axes

__all__ = ["plot_recv", "plot_encodings", "plot_overhead", "plot_paths"]

#: One colour per series, the same in every figure of the lab.
COLOURS = {"tcp": "#7F7F7F", "qos0": "#0070C0", "qos1": "#ED7D31", "qos2": "#2E8B57",
           "nodo": "#0070C0", "sottoscrittori": "#ED7D31"}


def plot_recv(sent: list[int], received: list[int], ax=None):
    """Two rows of blocks on a byte axis: what the sender wrote, what recv() returned."""
    fig, ax = _axes(ax, (8, 2.2))
    for row, sizes, colour in ((1, sent, "#DEEBF7"), (0, received, "#FBE4D7")):
        x = 0
        for b in sizes:
            # un blocco per chiamata, largo quanti byte ha passato
            ax.barh(row, b, left=x, height=0.6, color=colour, edgecolor="#101010", lw=0.8)
            x += b
    ax.set_yticks([1, 0], [f"sendall: {len(sent)}", f"recv: {len(received)}"])
    ax.set_xlabel("byte nel flusso TCP")
    ax.set_xlim(0, max(sum(sent), sum(received)))
    ax.set_ylim(-0.5, 1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_encodings(sizes: dict[str, int], windows_per_s: float = 2.0, ax=None):
    """Bytes per window in each encoding (encoding_sizes), horizontal bars on a log axis.

    Each bar is labelled with its bytes and the bit rate at windows_per_s.
    """
    fig, ax = _axes(ax, (8, 3))
    names = list(sizes)[::-1]                          # la prima codifica in alto
    vals = np.array([sizes[k] for k in names], dtype=float)
    colours = ["#ED7D31" if k == "int16" else "#DEEBF7" for k in names]
    ax.barh(names, vals, color=colours, edgecolor="#101010", lw=0.8)
    for y, v in enumerate(vals):
        kbit = v * 8 * windows_per_s / 1000            # byte per finestra -> kbit/s
        ax.text(v * 1.08, y, f"{v:.0f} B · {kbit:.1f} kbit/s", va="center")
    ax.set_xscale("log")
    ax.set_xlim(vals.min() / 2, vals.max() * 8)
    ax.set_xlabel(f"byte per finestra (kbit/s a {windows_per_s:g} finestre/s)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_overhead(table, ax=None):
    """Share of the bytes on the node's link that is not payload, against payload size.

    table is what overhead_table returned. The legend gives, for each
    series, the bytes per message that are not payload.
    """
    fig, ax = _axes(ax, (8, 3.5))
    x = table["payload_B"].to_numpy(dtype=float)
    labels = {"tcp": "TCP con framing", "qos0": "MQTT QoS 0", "qos1": "MQTT QoS 1",
              "qos2": "MQTT QoS 2"}
    for col in [c for c in labels if c in table]:
        over = table[col].to_numpy(dtype=float) - x    # byte per messaggio che non sono payload
        share = 100 * over / table[col].to_numpy(dtype=float)
        lo, hi = over.min(), over.max()
        extra = f"{lo:.0f} B" if lo == hi else f"{lo:.0f}–{hi:.0f} B"
        ax.plot(x, share, "o-", color=COLOURS[col], label=f"{labels[col]}  (+{extra})")
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("payload [byte]")
    ax.set_ylabel("non payload [%]")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_paths(path, fanout, axes=None):
    """Two panels: bytes per message on the two hops by QoS, and fan-out.

    path is what path_table returned, fanout what fanout_table returned.
    Blue is the node's link, orange the broker's side towards the
    subscribers; one legend above both panels.
    """
    import matplotlib.pyplot as plt
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    else:
        fig = axes[0].figure
    a, b = axes
    q = [f"QoS {k}" for k in path.index]
    a.bar(q, path["nodo"], color=COLOURS["nodo"], label="collegamento del nodo")
    a.bar(q, path["sottoscrittori"], bottom=path["nodo"], color=COLOURS["sottoscrittori"],
          label="broker → sottoscrittori")
    for k, tot in enumerate(path["totale"]):
        a.text(k, tot * 1.02, f"{tot:.0f} B", ha="center", va="bottom")
    a.set_ylim(0, path["totale"].max() * 1.2)
    a.set_xlabel("un sottoscrittore, due salti")
    a.set_ylabel("byte per messaggio")
    n = np.arange(len(fanout))
    w = 0.38
    b.bar(n - w / 2, fanout["nodo"], w, color=COLOURS["nodo"])
    b.bar(n + w / 2, fanout["sottoscrittori"], w, color=COLOURS["sottoscrittori"])
    for k, v in zip(n, fanout["sottoscrittori"]):
        # scrivo il valore solo sulle barre del broker: quelle del nodo sono tutte uguali
        b.text(k + w / 2, v * 1.02, f"{v:.0f}", ha="center", va="bottom")
    b.set_xticks(n, [str(k) for k in fanout.index])
    b.set_xlabel("sottoscrittori")
    b.set_ylim(0, fanout["sottoscrittori"].max() * 1.2)
    for ax_ in axes:
        for s in ("top", "right"):
            ax_.spines[s].set_visible(False)
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))            # lascio spazio in alto per la legenda
    return fig
