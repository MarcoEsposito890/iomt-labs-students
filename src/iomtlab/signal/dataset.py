"""All the windows of one device, from the CSV files to (X, y, purity, subject).

Lab 2 builds them step by step; from Lab 3 on the labs call build_windows,
which repeats the same steps with the same conventions.
"""
from __future__ import annotations

import numpy as np

from iomtlab.config import ALL_SUBJECTS, TARGET_HZ, WINDOW_SHIFT, WINDOW_SIZE

__all__ = ["build_windows", "build_paired_windows"]


def build_windows(device: str = "sp", size: int = WINDOW_SIZE, shift: int = WINDOW_SHIFT,
                  subjects=None) -> dict:
    """Windows of every recording of device, resampled to 50 Hz.

    For each subject and each recording: load_execution, resample_to_hz at
    TARGET_HZ, encode_labels, make_windows(size, shift). The signal stays in
    physical units (m/s^2, rad/s): the features of Lab 2 are computed on it.
    subjects defaults to config.ALL_SUBJECTS; a subject without recordings
    is skipped.

    Returns a dict with X (W, size, 6) float32, y (W,) int64, purity (W,)
    float32, subject (W,) and execution (W,) as object arrays of strings.
    Raises ValueError if no window is produced.
    """
    from iomtlab.signal.loading import list_executions, load_execution
    from iomtlab.signal.resample import resample_to_hz
    from iomtlab.signal.windowing import encode_labels, make_windows

    Xs, ys, ps, subs, exs = [], [], [], [], []
    for s in (subjects if subjects is not None else ALL_SUBJECTS):
        for e in list_executions(s, device):
            _, X, lab = resample_to_hz(load_execution(s, e, device), TARGET_HZ)
            Xw, yw, pw = make_windows(X, encode_labels(lab), size, shift)
            Xs.append(Xw); ys.append(yw); ps.append(pw)
            subs += [s] * len(yw); exs += [str(e)] * len(yw)   # soggetto e prova di ogni finestra
    if not ys or sum(len(v) for v in ys) == 0:
        raise ValueError(f"nessuna finestra per il dispositivo {device!r}: dataset assente?")
    return {"X": np.concatenate(Xs), "y": np.concatenate(ys), "purity": np.concatenate(ps),
            "subject": np.array(subs, dtype=object), "execution": np.array(exs, dtype=object)}


def build_paired_windows(size: int = WINDOW_SIZE, shift: int = WINDOW_SHIFT, subjects=None) -> dict:
    """Windows of phone and watch over the same instants, for the recordings both devices have.

    For each subject and each recording present on both devices:
    load_execution of "sp" and "sw", align_to_common_grid at TARGET_HZ (the
    intersection of the two time ranges, one uniform grid), then
    make_windows on each device with the labels of the phone. Window i of
    X_sp and window i of X_sw cover the same 1 s. The clocks are taken as
    they are written (both synchronised with NTP, Lab 2); no offset is
    corrected.

    Returns a dict with X_sp and X_sw (W, size, 6) float32 in physical
    units, y (W,) int64, purity (W,) float32, subject (W,) and execution (W,)
    as object arrays of strings. Raises ValueError if no window is produced.
    """
    from iomtlab.config import SENSOR_CHANNELS
    from iomtlab.signal.loading import list_executions, load_execution
    from iomtlab.signal.timing import align_to_common_grid
    from iomtlab.signal.windowing import encode_labels, make_windows

    A, B, ys, ps, subs, exs = [], [], [], [], [], []
    for s in (subjects if subjects is not None else ALL_SUBJECTS):
        both = sorted(set(list_executions(s, "sp")) & set(list_executions(s, "sw")))
        for e in both:
            try:
                a, b = align_to_common_grid(load_execution(s, e, "sp"), load_execution(s, e, "sw"), TARGET_HZ)
            except ValueError:
                continue                                   # le due registrazioni non si sovrappongono
            lab = encode_labels(a["label"].to_numpy())     # le etichette del telefono valgono per entrambi
            Xa, yw, pw = make_windows(a[SENSOR_CHANNELS].to_numpy(), lab, size, shift)
            Xb, _, _ = make_windows(b[SENSOR_CHANNELS].to_numpy(), lab, size, shift)
            A.append(Xa); B.append(Xb); ys.append(yw); ps.append(pw)
            subs += [s] * len(yw); exs += [str(e)] * len(yw)
    if not ys or sum(len(v) for v in ys) == 0:
        raise ValueError("nessuna finestra in comune fra telefono e orologio: dataset assente?")
    return {"X_sp": np.concatenate(A), "X_sw": np.concatenate(B), "y": np.concatenate(ys),
            "purity": np.concatenate(ps), "subject": np.array(subs, dtype=object),
            "execution": np.array(exs, dtype=object)}
