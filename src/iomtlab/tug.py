"""The Timed Up and Go test as a measuring instrument.

The classifier is read as a virtual sensor: its output, the duration of a
phase, is a physical quantity with an uncertainty (block 6.1). The module
has:

  1. phase_table: the contiguous phases of a recording (Lab 0, Lab 2);
  2. phase_boundaries, phase_durations: boundaries and durations from one
     label per window;
  3. readers for the results of the deployed system and of the manual video
     reference, in the format of the Matey-Sanz thesis repository;
  4. quantisation_budget: the uncertainty due to the window step, following
     the GUM (JCGM 100:2008).

File formats and sub-phase definitions follow
libs/chapter4/analysis/tug_results_processing.py of github.com/matey97/thesis,
rewritten here with the same semantics.

In the deployed-system results each phase has start and end as window
indices and a duration in ms. Over 3 328 phases, duration / number of
windows is 240.0 ms with a standard deviation of 2.2 ms: boundaries fall on
window steps, which is the quantisation quantisation_budget accounts for.
"""
from __future__ import annotations


import json
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "PHASES", "BATTERY_MAH",
    "phase_table", "phase_boundaries", "phase_durations",
    "load_system_results", "load_reference_results", "quantisation_budget",
]

#: The six sub-phases of the test, in time order. The two walks and the two
#: turns share an activity label: only their order tells them apart.
PHASES = ["standing_up", "first_walk", "first_turn", "second_walk",
          "second_turn", "sitting_down"]

#: Declared battery capacity of the devices of the study, in mAh (slides,
#: block 5.1). Source: Matey-Sanz et al., IEEE JBHI 2024, section II-D,
#: measured with BatteryHistorian.
BATTERY_MAH = {"sp": 5160, "sw": 580}


# ══════════════════════════════════ 1-2. from labels to phases and boundaries
def phase_table(df: pd.DataFrame) -> pd.DataFrame:
    """The contiguous phases of one recording, from per-sample labels.

    A new segment starts where the label differs from the previous sample;
    grouping by label alone would merge the two walks and the two turns.
    Returns a DataFrame with one row per phase, in order: label, t0 and t1
    (timestamps of the first and last sample, ms), dur_s and n (samples).
    dur_s runs from the first to the last sample, so it is one sampling
    interval short of the phase.
    """
    # ne(shift) è vero dove l'etichetta cambia: la somma cumulata diventa un contatore di segmenti
    seg = df["label"].ne(df["label"].shift()).cumsum()
    # raggruppo per segmento, non per etichetta; di ogni segmento tengo etichetta, primo e ultimo istante, numero di campioni
    out = (df.groupby(seg, sort=True)
             .agg(label=("label", "first"), t0=("timestamp", "first"),
                  t1=("timestamp", "last"), n=("label", "size"))
             .reset_index(drop=True))
    out["dur_s"] = (out["t1"] - out["t0"]) / 1000.0   # divido per 1000 perché i timestamp sono in ms
    return out[["label", "t0", "t1", "dur_s", "n"]]


def phase_boundaries(labels, step_s: float, min_run: int = 1) -> list[dict]:
    """Boundaries between consecutive phases, from one label per window.

    labels are in time order; step_s is the step between windows in seconds,
    the resolution of a boundary. A run shorter than min_run windows is
    merged into the previous one (with 1, a single differing window opens a
    phase). Returns a list of dicts with label, start_idx, end_idx
    (exclusive), start_s, end_s, duration_s, n_windows.

    start_s is the window index times the step: the boundary time as the
    system can know it. Its distance from the true time is what
    quantisation_budget accounts for.
    """
    labels = list(labels)
    if not labels:
        return []
    runs = []
    start = 0
    # scorro le etichette e chiudo un tratto ogni volta che cambiano, e alla fine
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((labels[start], start, i))   # etichetta, inizio, fine esclusa
            start = i
    # i tratti più corti di min_run li assorbo in quello precedente
    if min_run > 1:
        merged = []
        for lab, a, b in runs:
            if merged and (b - a) < min_run:
                merged[-1] = (merged[-1][0], merged[-1][1], b)   # allungo il tratto precedente fino a b
            else:
                merged.append((lab, a, b))
        runs = merged
    out = []
    for lab, a, b in runs:
        out.append({
            "label": lab, "start_idx": int(a), "end_idx": int(b),
            "start_s": a * step_s, "end_s": b * step_s,   # da indice di finestra a secondi
            "duration_s": (b - a) * step_s, "n_windows": int(b - a),
        })
    return out


def phase_durations(labels, step_s: float, min_run: int = 1) -> dict:
    """Total duration and duration per activity, in seconds, from one label per window.

    Runs with the same label are summed, so the two walks and the two turns
    are not separated (the deployed system separates them by their order,
    with a state machine). Returns a dict with total_s, by_activity_s (dict)
    and n_runs.
    """
    runs = phase_boundaries(labels, step_s, min_run)
    acc: dict = {}
    for r in runs:
        # sommo le durate dei tratti con la stessa etichetta
        acc[r["label"]] = acc.get(r["label"], 0.0) + r["duration_s"]
    total = sum(r["duration_s"] for r in runs)
    return {"total_s": total, "by_activity_s": acc, "n_runs": len(runs)}


# ══════════════════════════════════ 3. results of the study
def load_system_results(path) -> pd.DataFrame:
    """Results of the deployed system (file sNN_sw.json or sNN_sp.json) as a DataFrame.

    One row per execution with the total duration and the six sub-phases in
    seconds. Failed executions have duration -1 in the file and NaN here.
    window_step_s is the median over the phases of duration / number of
    windows.
    """
    path = Path(path)
    raw = json.loads(path.read_text())
    rows = []
    for i, r in enumerate(raw):
        res = r.get("activitiesResults") or {}
        # una durata -1 vuol dire esecuzione fallita e diventa NaN; le altre le porto da ms a s
        row = {"execution": i + 1,
               "duration": np.nan if r.get("duration", -1) in (-1, None)
               else r["duration"] / 1000.0,
               "successful": bool(r.get("successful", False))}
        steps = []
        for ph in PHASES:
            v = res.get(ph.upper())          # nel file le fasi sono in maiuscolo
            if isinstance(v, dict) and v.get("duration") is not None:
                row[ph] = v["duration"] / 1000.0
                n = (v.get("end") or 0) - (v.get("start") or 0)   # start ed end sono indici di finestra
                # durata divisa per il numero di finestre: è il passo fra finestre
                if n > 0:
                    steps.append(v["duration"] / n / 1000.0)
            else:
                row[ph] = np.nan
        row["window_step_s"] = float(np.median(steps)) if steps else np.nan   # la mediana sulle fasi
        rows.append(row)
    return pd.DataFrame(rows)


def load_reference_results(path) -> pd.DataFrame:
    """Manual reference from the video analysis (file sNN_results.csv).

    The file has one instant per boundary, as year/month/day/hour/minute/
    second/ms. Durations are differences, with the definition of the study:

        duration      = sit_end   - stand_start
        standing_up   = stand_end - stand_start
        first_walk    = turn1_start - stand_end
        first_turn    = turn1_end - turn1_start
        second_walk   = turn2_start - turn1_end
        second_turn   = turn2_end - turn2_start
        sitting_down  = sit_end   - turn2_end

    The annotation is manual and has its own uncertainty, which is not zero
    and not known. Missing boundaries give NaN. Returns a DataFrame with one
    row per execution, in seconds.
    """
    df = pd.read_csv(path, converters={"phase": lambda x: x.strip()})   # tolgo gli spazi dai nomi dei confini
    # ricompongo l'istante di ogni confine dai campi separati; i ms diventano microsecondi
    df["t"] = pd.to_datetime(dict(
        year=df.year, month=df.month, day=df.day,
        hour=df.hour, minute=df.minute, second=df.second,
        microsecond=df.ms * 1000))
    df = df.set_index(["execution", "phase"])["t"]   # indicizzo per esecuzione e nome del confine
    rows = []
    for ex, g in df.groupby(level=0):
        # at restituisce l'istante di un confine, NaT se manca
        def at(name):
            try:
                return g.loc[(ex, name)]
            except KeyError:
                return pd.NaT

        # dt è la distanza in secondi fra due confini, NaN se uno dei due manca
        def dt(a, b):
            x, y = at(a), at(b)
            return np.nan if pd.isna(x) or pd.isna(y) else (y - x).total_seconds()

        rows.append({
            "execution": ex,
            "duration": dt("stand_start", "sit_end"),
            "standing_up": dt("stand_start", "stand_end"),
            "first_walk": dt("stand_end", "turn1_start"),
            "first_turn": dt("turn1_start", "turn1_end"),
            "second_walk": dt("turn1_end", "turn2_start"),
            "second_turn": dt("turn2_start", "turn2_end"),
            "sitting_down": dt("turn2_end", "sit_end"),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════ 4. uncertainty budget
def quantisation_budget(step_s: float, k: float = 2.0) -> dict:
    """Uncertainty of a duration due to the quantisation of its boundaries only.

    A boundary is known to within one window step. The GUM (JCGM 100:2008,
    F.2.2.1) treats this as a uniform distribution of width step_s, standard
    uncertainty step_s / sqrt(12). A duration is the difference of two
    independent boundaries, so the variances add:

        u(t_b) = step / sqrt(12)
        u(T)   = sqrt(2) * u(t_b)
        U(T)   = k * u(T)

    The term does not depend on the model: a perfect classifier still places
    boundaries on the window grid, and classification errors add to it.
    Returns a dict with step_s, u_boundary_s, u_duration_s, k, U_duration_s
    (seconds).
    """
    # incertezza tipo di un confine: uniforme di ampiezza pari al passo
    u_b = step_s / np.sqrt(12)
    # la durata è la differenza di due confini indipendenti: le varianze si sommano
    u_T = u_b * np.sqrt(2)
    return {"step_s": float(step_s),
            "u_boundary_s": float(u_b),
            "u_duration_s": float(u_T),
            "k": float(k),
            "U_duration_s": float(k * u_T)}   # incertezza estesa con fattore di copertura k
