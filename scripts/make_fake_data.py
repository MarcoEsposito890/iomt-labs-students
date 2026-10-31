#!/usr/bin/env python3
"""Generate a synthetic dataset with the layout of the course dataset.

Used to run notebooks and tests without the real data, for example in CI.
The numbers it produces have no experimental value.

    python scripts/make_fake_data.py build/fake_data
    export IOMTLAB_DATA_DIR=build/fake_data        # Windows: set IOMTLAB_DATA_DIR=...

Layout written, as in data/DATA:

    sNN/sNN_EE_sp.csv, sNN/sNN_EE_sw.csv
    columns: timestamp (ms, int), x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro, label
    executions_info.csv, subjects_info.csv, FAKE_DATA.txt

The timing imitates the real devices only in order of magnitude: the phone
has jitter and duplicate timestamps, the watch less.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PHASES = [("SEATED", 1.5), ("STANDING_UP", 1.2), ("WALKING", 3.0), ("TURNING", 1.2),
          ("WALKING", 3.0), ("TURNING", 1.2), ("SITTING_DOWN", 1.4), ("SEATED", 1.5)]

DEVICE = {  # nominal step [ms], jitter std [ms], duplicate share, gyro z bias [rad/s]
    "sp": (10.0, 8.0, 0.045, 0.026),
    "sw": (10.0, 2.0, 0.005, 0.010),
}


def _timestamps(rng, t0_ms, duration_s, device):
    """Timestamps in ms with the jitter, duplicates and gaps of device."""
    step, jitter, dup, _ = DEVICE[device]
    n = int(duration_s * 1000 / step)
    dt = np.clip(rng.normal(step, jitter, n - 1), 1.0, None)
    dt[rng.uniform(size=n - 1) < dup] = 0.0                 # qualche timestamp duplicato
    gaps = rng.uniform(size=n - 1) < 0.004
    dt[gaps] += rng.uniform(40, 120, gaps.sum())              # e qualche buco
    return np.round(t0_ms + np.concatenate([[0.0], np.cumsum(dt)])).astype(np.int64)


def _signals(rng, t_s, labels, device):
    """Six channels with gravity, a step component while moving and noise."""
    n = len(t_s)
    out = np.zeros((n, 6))
    moving = np.isin(labels, ["WALKING", "TURNING"])
    trans = np.isin(labels, ["STANDING_UP", "SITTING_DOWN"])
    step_hz = rng.uniform(1.4, 1.9)
    out[:, 2] = 9.81
    out[:, 0] = np.where(labels == "SEATED", 9.81 * 0.3, 0.0)
    out[:, 1] += moving * 1.5 * np.sin(2 * np.pi * step_hz * t_s)
    out[:, 2] += moving * 1.2 * np.sin(2 * np.pi * 2 * step_hz * t_s)
    out[:, 0] += trans * 2.0
    out[:, 5] = np.where(labels == "TURNING", 1.8, 0.0) + DEVICE[device][3]
    out[:, 3] += moving * 0.6 * np.sin(2 * np.pi * step_hz * t_s + 0.5)
    out[:, :3] += rng.normal(0, 0.05, (n, 3))
    out[:, 3:] += rng.normal(0, 0.01, (n, 3))
    return out


def make(dest: Path, subjects: int, executions: int, seed: int) -> None:
    """Write the synthetic dataset to dest."""
    rng = np.random.default_rng(seed)
    dest.mkdir(parents=True, exist_ok=True)
    rows, subj_rows = [], []
    names = [f"s{i:02d}" for i in range(1, 24)]
    # s16 c'è sempre: test e notebook caricano la prova 1 di s16
    chosen = sorted(set(names[: subjects - 1]) | {"s16"})
    for s in chosen:
        (dest / s).mkdir(exist_ok=True)
        # stesse colonne del subjects_info.csv vero
        subj_rows.append({"subject_id": s, "age": int(rng.integers(20, 60)),
                          "gender": str(rng.choice(["F", "M"])),
                          "height": int(rng.integers(155, 195)),
                          "weight": int(rng.integers(50, 95)),
                          "dominant_hand": str(rng.choice(["L", "R"])),
                          "executions": executions})
        for e in range(1, executions + 1):
            durations = [d * rng.uniform(0.8, 1.2) for _, d in PHASES]
            bounds = np.cumsum([0.0] + durations)
            t0 = 1_690_000_000_000 + int(rng.integers(0, 10**9))
            for dev, offset_ms, skew in (("sp", 0, 0.0), ("sw", rng.integers(-5, 6),
                                                          rng.uniform(-3e-4, 3e-4))):
                ts = _timestamps(rng, t0 + offset_ms, bounds[-1], dev)
                t_s = (ts - ts[0]) / 1000.0
                ts = (ts + skew * (ts - ts[0])).astype(np.int64)
                idx = np.clip(np.searchsorted(bounds, t_s, side="right") - 1, 0,
                              len(PHASES) - 1)
                labels = np.array([PHASES[i][0] for i in idx])
                X = _signals(rng, t_s, labels, dev)
                df = pd.DataFrame(X, columns=["x_acc", "y_acc", "z_acc",
                                              "x_gyro", "y_gyro", "z_gyro"])
                df.insert(0, "timestamp", ts)
                df["label"] = labels
                df.to_csv(dest / s / f"{s}_{e:02d}_{dev}.csv", index=False)
            # stesse colonne dell'executions_info.csv vero
            rows.append({"execution_id": f"{s}_{e:02d}",
                         "orientation": str(rng.choice(["front", "front_inv", "back", "back_inv"])),
                         "first_turn": str(rng.choice(["l", "r"])),
                         "second_turn": str(rng.choice(["l", "r"]))})
    pd.DataFrame(rows).to_csv(dest / "executions_info.csv", index=False)
    pd.DataFrame(subj_rows).to_csv(dest / "subjects_info.csv", index=False)
    (dest / "FAKE_DATA.txt").write_text(
        "Synthetic data generated by scripts/make_fake_data.py. Not the course "
        "dataset: do not report any number computed from it.\n")
    print(f"{len(chosen)} soggetti, {executions} prove ciascuno, in {dest}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", type=Path)
    ap.add_argument("--subjects", type=int, default=4)
    ap.add_argument("--executions", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make(a.dest, a.subjects, a.executions, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
