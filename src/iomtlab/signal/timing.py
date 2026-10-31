"""Timing of the acquisition chain (Lab 2).

A phone does not deliver its inertial samples at regular intervals: the OS
scheduler adds jitter, and phone and watch behave differently. These
functions measure by how much, and put the two devices on one time axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TimingReport:
    """Timing statistics of one recording. Intervals dt are in ms."""

    n_samples: int
    duration_s: float
    mean_hz: float
    median_dt_ms: float
    std_dt_ms: float
    p99_dt_ms: float
    max_dt_ms: float
    n_gaps: int          # intervalli più lunghi di 3 volte la mediana
    n_duplicates: int    # campioni con lo stesso millisecondo del precedente

    @property
    def duplicate_fraction(self) -> float:
        """Share of intervals equal to 0, between 0 and 1."""
        return self.n_duplicates / max(self.n_samples - 1, 1)

    def __str__(self) -> str:
        return (
            f"{self.n_samples} campioni in {self.duration_s:.3f} s "
            f"({self.mean_hz:.1f} Hz)\n"
            f"  dt: mediana {self.median_dt_ms:.2f} ms, "
            f"sigma {self.std_dt_ms:.2f} ms, "
            f"p99 {self.p99_dt_ms:.2f} ms, max {self.max_dt_ms:.2f} ms\n"
            f"  buchi (dt > 3x mediana): {self.n_gaps}\n"
            f"  duplicati (dt = 0): {self.n_duplicates} "
            f"({100 * self.duplicate_fraction:.1f} %)"
        )


def timing_report(df: pd.DataFrame) -> TimingReport:
    """Timing statistics of one recording, as a TimingReport.

    df["timestamp"] is in ms. Zero intervals (duplicate timestamps) are valid
    and stay in the statistics. A gap is an interval longer than three times
    the median. std_dt_ms is the population standard deviation (ddof=0).
    """
    dt = np.diff(df["timestamp"].to_numpy(dtype=np.float64))   # intervalli fra campioni consecutivi, in ms
    duration_s = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 1000.0   # dal primo all'ultimo, in s
    median = float(np.median(dt))
    return TimingReport(
        n_samples=len(df),
        duration_s=duration_s,
        mean_hz=len(dt) / duration_s,                      # la frequenza media: intervalli al secondo
        median_dt_ms=median,
        std_dt_ms=float(np.std(dt)),                       # la dispersione degli intervalli, con ddof=0
        p99_dt_ms=float(np.percentile(dt, 99)),
        max_dt_ms=float(np.max(dt)),
        n_gaps=int(np.sum(dt > 3 * median)),               # conto come buco un intervallo oltre 3 volte la mediana
        n_duplicates=int(np.sum(dt == 0)),                 # e come duplicato un intervallo nullo
    )


def align_to_common_grid(df_sp: pd.DataFrame, df_sw: pd.DataFrame,
                         hz: float = 50.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample phone and watch onto the same uniform time grid at hz.

    The grid covers the intersection of the two time ranges: outside it one
    of the devices has no data. Returns two DataFrames of equal length with
    the same timestamp column (ms). Raises ValueError if the recordings do
    not overlap.
    """
    from iomtlab.signal.resample import on_grid
    t0 = max(df_sp["timestamp"].iloc[0], df_sw["timestamp"].iloc[0])    # prelevo il tempo t0 come massimo tra istanti iniziali di sp e sw
    t1 = min(df_sp["timestamp"].iloc[-1], df_sw["timestamp"].iloc[-1])  # e t1 come minimo tra gli istanti finali
    if t1 <= t0:
        raise ValueError("i due flussi non si sovrappongono")
    grid = np.arange(t0, t1, 1000.0 / hz)          # griglia uniforme fra t0 e t1, passo 1000/hz ms
    # interpolo entrambi i dispositivi sulla stessa griglia
    return on_grid(df_sp, grid), on_grid(df_sw, grid)


def delivery_regularity(df_sp: pd.DataFrame, df_sw: pd.DataFrame) -> dict:
    """Compare how regularly phone and watch deliver their samples.

    Both devices declare the same nominal period and have about the same
    mean rate. What differs is the spread of the intervals and the share of
    duplicate timestamps: the OS delivers several samples at once when it
    schedules the process again, so the delivery time is not the sampling
    time.

    Returns a dict with sp and sw (TimingReport), median_dt_diff_ms,
    std_ratio and duplicate_ratio (phone over watch; inf when the watch
    value is 0). std_ratio > 1 means the phone is the less regular.
    """
    a = timing_report(df_sp)                       # statistiche del telefono
    b = timing_report(df_sw)                       # e dell'orologio
    return {
        "sp": a,
        "sw": b,
        "median_dt_diff_ms": a.median_dt_ms - b.median_dt_ms,
        # rapporto fra le deviazioni standard degli intervalli: telefono su orologio
        "std_ratio": a.std_dt_ms / b.std_dt_ms if b.std_dt_ms else float("inf"),
        # stesso rapporto sulla frazione di timestamp duplicati
        "duplicate_ratio": (a.duplicate_fraction / b.duplicate_fraction
                            if b.duplicate_fraction else float("inf")),
    }


def device_offset(df_sp: pd.DataFrame, df_sw: pd.DataFrame) -> dict:
    """Time offsets between phone and watch in the same recording, as a dict.

    Both devices write UNIX timestamps and were synchronised with NTP before
    each recording (Matey-Sanz et al., IEEE JBHI 2024, section II-C2). The
    residual offset has three possible causes, not separated here: residual
    synchronisation error, different effective sampling rates, samples lost
    by one device; comparing duration and sample count exposes the third.
    Offsets are watch minus phone in ms, durations in s.
    """
    return {
        # scarto all'inizio e alla fine: orologio meno telefono, in ms
        "start_offset_ms": float(df_sw["timestamp"].iloc[0] - df_sp["timestamp"].iloc[0]),
        "end_offset_ms": float(df_sw["timestamp"].iloc[-1] - df_sp["timestamp"].iloc[-1]),
        # durate in secondi dei due flussi
        "duration_sp_s": float(
            (df_sp["timestamp"].iloc[-1] - df_sp["timestamp"].iloc[0]) / 1000.0),
        "duration_sw_s": float(
            (df_sw["timestamp"].iloc[-1] - df_sw["timestamp"].iloc[0]) / 1000.0),
        "n_samples_sp": len(df_sp),
        "n_samples_sw": len(df_sw),
    }


def shift_clock(df: pd.DataFrame, offset_ms: float = 0.0, skew_ppm: float = 0.0) -> pd.DataFrame:
    """A copy of df with a known clock error added to its timestamps.

    The clock model of the theory («Clocks: offset, skew and drift»):
    t' = t0 + (t - t0)(1 + ρ) + C, with t0 the first timestamp, ρ = skew_ppm
    x 1e-6 the frequency error of the quartz and C = offset_ms the offset.
    """
    out = df.copy()
    t = out["timestamp"].to_numpy(dtype=np.float64)
    t0 = t[0]
    out["timestamp"] = t0 + (t - t0) * (1.0 + skew_ppm * 1e-6) + offset_ms
    return out


def label_agreement(df_sp: pd.DataFrame, df_sw: pd.DataFrame, offset_ms: float = 0.0,
                    skew_ppm: float = 0.0, hz: float = 50.0) -> float:
    """Share of the instants of the common grid where phone and watch have the same label.

    The watch clock is first shifted with shift_clock(offset_ms, skew_ppm);
    then both recordings go onto the common grid at hz
    (align_to_common_grid), where the label of each instant is the one of the
    last sample not after it. The labels of both devices come from the same
    boundaries read on the video, so without added error the agreement is
    close to 1: the residual is the boundaries falling between two samples.
    """
    a, b = align_to_common_grid(df_sp, shift_clock(df_sw, offset_ms, skew_ppm), hz)
    return float(np.mean(a["label"].to_numpy() == b["label"].to_numpy()))


def offset_after(skew_ppm: float, seconds: float) -> float:
    """Offset in ms that a clock with skew_ppm accumulates in seconds: ρ x seconds x 1000.

    40 ppm give 2.4 ms per minute, 144 ms per hour, 3.46 s per day.
    """
    return skew_ppm * 1e-6 * seconds * 1000.0
