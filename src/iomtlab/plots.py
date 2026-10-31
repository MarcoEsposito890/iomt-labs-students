"""The recurring figures of the labs (the Lab 1 figures are in iomtlab.net.figures).

Every function returns the matplotlib figure; none calls show() or writes to
disk. All but plot_rounds accept ax, to compose several panels.
"""
from __future__ import annotations

import numpy as np

__all__ = ["plot_trial", "plot_phases", "plot_intervals", "plot_agreement", "plot_window_purity",
           "plot_folds", "per_subject_bars", "plot_cost_accuracy", "reliability_diagram",
           "plot_rounds"]

#: Background colour of each TUG phase, the same in every lab.
PHASE_COLOURS = {"SEATED": "#EEEEEE", "STANDING_UP": "#FBE4D7", "WALKING": "#DEEBF7",
                 "TURNING": "#F8CBAD", "SITTING_DOWN": "#E2D5F0"}
#: Short Italian name of each phase, written on the plots.
PHASE_SHORT = {"SEATED": "seduto", "STANDING_UP": "si alza", "WALKING": "cammina",
               "TURNING": "gira", "SITTING_DOWN": "si siede"}


def _axes(ax, figsize):
    """(figure, ax) of the given ax, or of a new figure of size figsize (inches)."""
    import matplotlib.pyplot as plt
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


def plot_trial(df, sensor: str = "acc", ax=None):
    """One recording: the three axes of a sensor and their norm, with the phases shaded.

    df is what signal.load_execution returns; sensor is "acc" (m/s^2) or
    "gyro" (rad/s). Time is in seconds from the first sample. Each TUG phase
    gets its background colour (PHASE_COLOURS) and its name (PHASE_SHORT)
    above the axes.
    """
    from matplotlib.transforms import blended_transform_factory
    from iomtlab.tug import phase_table
    fig, ax = _axes(ax, (10, 3.2))
    t0 = df["timestamp"].iloc[0]
    t = (df["timestamp"] - t0).to_numpy() / 1000.0   # i timestamp sono in ms: passo ai secondi dal primo campione
    top = blended_transform_factory(ax.transData, ax.transAxes)   # x in secondi, y in frazione dell'altezza
    alto = False
    for _, r in phase_table(df).iterrows():
        a, b = (r.t0 - t0) / 1000, (r.t1 - t0) / 1000
        # una fascia colorata per fase, dal primo all'ultimo campione della fase
        ax.axvspan(a, b, color=PHASE_COLOURS.get(r.label, "white"), lw=0)
        # il nome di una fase breve lo alterno su due righe, così i nomi vicini non si sovrappongono
        alto = (not alto) and (b - a) < 0.1 * (t[-1] - t[0])
        ax.text((a + b) / 2, 1.13 if alto else 1.02, PHASE_SHORT.get(r.label, r.label),
                transform=top, ha="center", va="bottom", fontsize="small")
    cols = [f"{a}_{sensor}" for a in "xyz"]
    for c, colour in zip(cols, ("#C00000", "#2E8B57", "#0070C0")):
        ax.plot(t, df[c], lw=0.9, color=colour, label=c)
    ax.plot(t, np.linalg.norm(df[cols].to_numpy(), axis=1), lw=1.2, color="#101010",
            label="norma")                             # la norma dei tre assi
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("tempo [s]")
    ax.set_ylabel("m/s²" if sensor == "acc" else "rad/s")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig



def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_phases(phases_sp, phases_sw, ax=None, labels=("telefono", "orologio")):
    """The phases of the two devices on the same time axis, one row each (Lab 2).

    phases_sp and phases_sw are tug.phase_table of the phone and of the
    watch; time is in seconds from the earliest first sample of the two.
    """
    fig, ax = _axes(ax, (10, 1.4))
    t0 = min(phases_sp["t0"].iloc[0], phases_sw["t0"].iloc[0])
    for row, ph in enumerate((phases_sp, phases_sw)):
        for _, r in ph.iterrows():
            a, b = (r.t0 - t0) / 1000, (r.t1 - t0) / 1000
            ax.barh(row, b - a, left=a, height=0.7, color=PHASE_COLOURS.get(r.label, "white"),
                    edgecolor="#606060", lw=0.5)
            if row == 0:
                ax.text((a + b) / 2, -0.55, PHASE_SHORT.get(r.label, r.label), ha="center",
                        va="bottom", fontsize="x-small")
    ax.set_yticks([0, 1], list(labels))
    ax.set_ylim(1.5, -0.8)
    ax.set_xlabel("tempo [s]")
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_intervals(df_sp, df_sw, axes=None, max_ms: float = 60.0):
    """Histograms of the intervals between consecutive timestamps of phone and watch (Lab 2).

    One panel per device, counts on a log scale, bins of 1 ms up to max_ms;
    longer intervals are counted in the last bin. The column at 0 ms is the
    duplicate timestamps.
    """
    import matplotlib.pyplot as plt
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.0))
    axes = list(np.atleast_1d(axes).ravel())
    fig = axes[0].figure
    from iomtlab.signal.timing import timing_report
    for ax, df, name, c in ((axes[0], df_sp, "telefono", "#4878A8"),
                            (axes[1], df_sw, "orologio", "#E07B39")):
        dt = np.diff(df["timestamp"].to_numpy(dtype=np.float64))
        ax.hist(np.clip(dt, 0, max_ms), bins=np.arange(0, max_ms + 2) - 0.5, color=c)
        ax.set_yscale("log")
        r = timing_report(df)
        ax.set_title(f"{name}: σ {r.std_dt_ms:.1f} ms, duplicati {100 * r.duplicate_fraction:.1f} %",
                     fontsize="medium")
        ax.set_xlabel("intervallo fra timestamp [ms]")
        ax.set_ylabel("conteggio")
        _clean(ax)
    fig.tight_layout()
    return fig


def plot_agreement(errors_ms, agreement, ax=None, mark=None, level: float = 0.99):
    """Label agreement of the two devices against the error added to the watch clock (Lab 2).

    agreement is a share (0-1) for each error in ms; with mark (ms), a
    vertical line where the agreement crosses level, drawn as a dotted line.
    """
    fig, ax = _axes(ax, (6, 3.0))
    ax.plot(errors_ms, 100 * np.asarray(agreement, dtype=float), "o-", color="#4878A8")
    if mark is not None:
        ax.axhline(100 * level, color="#A0A0A0", ls=":", lw=1)
        ax.axvline(mark, color="#C00000", lw=1)
        ax.text(mark, ax.get_ylim()[0], f" {mark:.1f} ms", color="#C00000", va="bottom",
                fontsize="small")
    ax.set_xlabel("errore aggiunto all'orologio [ms]")
    ax.set_ylabel("stessa etichetta [%]")
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_window_purity(y, purity, names=None, threshold: float = 0.8, ax=None):
    """Windows per class, pure (purity >= threshold, blue) and ambiguous (orange) (Lab 2)."""
    from iomtlab.spec import active, short_label
    names = list(names or active().classes)
    fig, ax = _axes(ax, (6, 3.0))
    y, purity = np.asarray(y), np.asarray(purity)
    x = np.arange(len(names))
    pure = np.array([np.sum((y == k) & (purity >= threshold)) for k in x])
    amb = np.array([np.sum((y == k) & (purity < threshold)) for k in x])
    ax.bar(x, pure, color="#4878A8", label=f"pure (≥ {threshold:g})")
    ax.bar(x, amb, bottom=pure, color="#E07B39", label="ambigue")
    for k in x:
        tot = pure[k] + amb[k]
        if tot:
            ax.text(k, tot, f"{100 * amb[k] / tot:.0f} %", ha="center", va="bottom",
                    fontsize="small")
    ax.set_xticks(x, [short_label(n) for n in names])
    ax.set_ylabel("finestre")
    ax.legend(frameon=False, fontsize="small")
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_folds(groups, folds, ax=None):
    """Which subject is in which fold: one row per fold, one column per subject.

    Each cell is the fraction of the subject's rows in test. With loso_folds
    the figure is a diagonal, with subject_folds it is made of blocks; with a
    random split over rows every cell is partly lit. The title states how
    many (fold, subject) cells have the subject split between train and test.

    groups has one subject id per data row; folds is an iterable of boolean
    (train_mask, test_mask) over groups. Raises ValueError if a mask has the
    wrong length, as index arrays do.
    """
    groups = np.asarray(groups, dtype=object).astype(str)
    subs = list(np.unique(groups))
    rows = []
    for tr, te in folds:
        te = np.asarray(te, dtype=bool)
        if len(te) != len(groups):
            raise ValueError(
                f"maschera lunga {len(te)}, attesa {len(groups)}: "
                f"sembrano indici invece di maschere booleane")
        # per ogni soggetto calcolo la frazione delle sue righe che va in prova:
        # un valore strettamente fra 0 e 1 vuol dire che il soggetto sta da tutte e due le parti
        rows.append([float(te[groups == s].mean()) for s in subs])
    M = np.array(rows)                       # una riga per piega, una colonna per soggetto

    fig, ax = _axes(ax, (max(6.0, 0.35 * len(subs)), max(2.0, 0.28 * len(M))))
    ax.imshow(M, aspect="auto", vmin=0, vmax=1, cmap="RdYlBu_r",
              interpolation="nearest")
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels(subs, rotation=90, fontsize=7)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f"piega {i + 1}" for i in range(len(M))], fontsize=8)
    ax.set_xlabel("soggetto")
    split = M[(M > 0) & (M < 1)].size        # conto le celle con un soggetto spezzato
    ax.set_title(f"rosso = in prova, blu = in addestramento; soggetti spezzati: {split}",
                 fontsize=9)
    return fig


def per_subject_bars(df, ax=None, reference=None):
    """Per-subject scores as sorted bars, with the mean as a horizontal line.

    df is the table of evaluation.per_subject_scores; the legend gives mean
    and standard deviation across subjects. reference, if given, is drawn
    as a dashed line.
    """
    sc = np.asarray(df["score"], dtype=float)
    subs = list(df["subject"])
    order = np.argsort(sc)                   # ordino i soggetti dal punteggio più basso al più alto
    fig, ax = _axes(ax, (max(6.0, 0.32 * len(subs)), 3.2))
    ax.bar(range(len(sc)), sc[order], color="#4878A8")
    ax.set_xticks(range(len(sc)))
    ax.set_xticklabels([subs[i] for i in order], rotation=90, fontsize=7)
    m = float(sc.mean())
    # la linea della media, con la deviazione standard fra soggetti in legenda
    ax.axhline(m, color="#C05030", lw=1.2,
               label=f"media {m:.3f}  (sigma {sc.std():.3f})")
    if reference is not None:
        ax.axhline(float(reference), color="#808080", lw=1.0, ls="--",
                   label=f"riferimento {float(reference):.3f}")
    ax.set_ylabel("macro-F1")
    ax.set_xlabel("soggetto")
    ax.legend(fontsize=8)
    return fig


def plot_cost_accuracy(points, ax=None, xlabel="MAC per finestra",
                       ylabel="macro-F1", logx=True):
    """Cost against score, with the Pareto front.

    points is a sequence of (name, cost, score) or (name, cost, score,
    spread); the spread becomes an error bar. The cost axis is logarithmic
    unless logx is False.
    """
    rows = [tuple(p) for p in points]
    name = [r[0] for r in rows]
    x = np.array([float(r[1]) for r in rows])     # costo
    y = np.array([float(r[2]) for r in rows])     # prestazione
    err = np.array([float(r[3]) if len(r) > 3 else 0.0 for r in rows])   # dispersione, zero se manca

    # un punto sta sulla frontiera se nessun altro punto costa al massimo quanto lui,
    # rende almeno quanto lui ed è strettamente migliore in una delle due grandezze
    front = [i for i in range(len(x))
             if not any((x[j] <= x[i]) and (y[j] >= y[i]) and (j != i)
                        and (x[j] < x[i] or y[j] > y[i])
                        for j in range(len(x)))]

    fig, ax = _axes(ax, (6.4, 4.0))
    ax.errorbar(x, y, yerr=err if err.any() else None, fmt="o",
                color="#4878A8", ecolor="#A0A0A0", capsize=3, ms=7, zorder=3)
    if len(front) > 1:
        o = np.argsort(x[front])             # unisco i punti della frontiera in ordine di costo
        ax.plot(x[np.array(front)][o], y[np.array(front)][o], "-",
                color="#C05030", lw=1.2, zorder=2, label="frontiera di Pareto")
        ax.legend(fontsize=8)
    for i, n in enumerate(name):
        ax.annotate(n, (x[i], y[i]), textcoords="offset points",
                    xytext=(7, 4), fontsize=8)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return fig


def reliability_diagram(conf, correct, n_bins: int = 15, ax=None):
    """Reliability diagram: stated confidence against observed accuracy.

    Bars are equal-width confidence bins, the dashed diagonal is perfect
    calibration. The count of each bin is written above its bar: ECE weighs
    bins by count, so a nearly empty bin far from the diagonal counts
    little. The title gives uncertainty.ece on the same bins. Raises
    ValueError if conf and correct differ in shape.
    """
    from iomtlab.uncertainty import ece
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if conf.shape != correct.shape:
        raise ValueError(f"conf {conf.shape} e correct {correct.shape} "
                         f"devono avere la stessa forma")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, edges[1:-1])     # il bin di ogni campione, come in uncertainty.ece
    centres = 0.5 * (edges[:-1] + edges[1:])
    cnt = np.array([(idx == b).sum() for b in range(n_bins)])   # quanti campioni cadono in ogni bin
    ok = cnt > 0                             # disegno solo i bin non vuoti
    # l'altezza della barra è la frazione di risposte giuste nel bin
    acc = np.array([correct[idx == b].mean() if n else np.nan for b, n in enumerate(cnt)])

    fig, ax = _axes(ax, (5.0, 4.4))
    w = 1.0 / n_bins
    ax.bar(centres[ok], acc[ok], width=w * 0.92, color="#4878A8",
           edgecolor="white", zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="#808080", lw=1.0, zorder=4)
    for c, a, n in zip(centres[ok], acc[ok], cnt[ok]):
        ax.annotate(str(n), (c, a), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=6, color="#505050")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidenza dichiarata")
    ax.set_ylabel("frequenza osservata")
    ax.set_title(f"ECE = {ece(conf, correct, n_bins):.4f}  ({n_bins} bin, {cnt.sum()} campioni)",
                 fontsize=10)
    return fig


def plot_rounds(history):
    """Federated training in two panels: macro-F1 against rounds and against bytes.

    history is {name: rows} with rows (round, metric, cumulative_bytes), or
    a single sequence of rows. The two panels always come together: against
    rounds a quantised update looks worse, against the bytes sent it may
    not. Returns a new figure; raises ValueError if a row does not have
    three values.
    """
    import matplotlib.pyplot as plt
    if not isinstance(history, dict):
        history = {"": history}              # un solo andamento: gli do un nome vuoto
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))

    for name, rows in history.items():
        a = np.asarray([[float(v) for v in r] for r in rows], dtype=float)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(f"'{name}': attese righe (round, metrica, byte), "
                             f"ottenuta forma {a.shape}")
        axes[0].plot(a[:, 0], a[:, 1], "-o", ms=3, label=name or None)         # a sinistra contro i round
        axes[1].plot(a[:, 2] / 1e6, a[:, 1], "-o", ms=3, label=name or None)   # a destra contro i byte, in MB

    axes[0].set_xlabel("round")
    axes[1].set_xlabel("byte trasmessi, cumulativi (MB)")
    for a in axes:
        a.set_ylabel("macro-F1")
        a.grid(alpha=0.3)
        if len(history) > 1:
            a.legend(fontsize=8)
    axes[0].set_title("a parità di round", fontsize=10)
    axes[1].set_title("a parità di byte", fontsize=10)
    fig.tight_layout()
    return fig
