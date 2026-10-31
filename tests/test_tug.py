"""The TUG as a measuring instrument.

The quantisation tests need the results of the deployed system, which are
not in the Zenodo record:

    python scripts/get_data.py --from /path/to/thesis-main
"""
from __future__ import annotations

import glob

import numpy as np
import pytest

from iomtlab import tug
from iomtlab.config import TUG_RESULTS_DIR

has_results = TUG_RESULTS_DIR.is_dir() and any(TUG_RESULTS_DIR.glob("s*/*.json"))
needs_results = pytest.mark.skipif(
    not has_results,
    reason="risultati TUG assenti: scripts/get_data.py --from <thesis>")


# ══════════════════════════════════ confini e durate, senza dati
def test_phase_boundaries_on_a_known_sequence():
    labels = ["SEATED"] * 3 + ["STANDING_UP"] * 2 + ["WALKING"] * 5
    runs = tug.phase_boundaries(labels, step_s=0.25)
    assert [r["label"] for r in runs] == ["SEATED", "STANDING_UP", "WALKING"]
    assert [r["n_windows"] for r in runs] == [3, 2, 5]
    assert runs[1]["start_s"] == pytest.approx(0.75)
    assert runs[1]["duration_s"] == pytest.approx(0.50)


def test_durations_are_multiples_of_the_step():
    """A boundary cannot fall between two windows."""
    rng = np.random.default_rng(0)
    labels = list(rng.integers(0, 3, 200))
    step = 0.24
    for r in tug.phase_boundaries(labels, step):
        assert abs(r["duration_s"] / step - round(r["duration_s"] / step)) < 1e-9


def test_min_run_absorbs_isolated_windows():
    labels = ["WALKING"] * 5 + ["TURNING"] + ["WALKING"] * 5
    assert len(tug.phase_boundaries(labels, 0.25, min_run=1)) == 3
    assert len(tug.phase_boundaries(labels, 0.25, min_run=2)) == 2


def test_phase_durations_sums_by_activity():
    labels = ["WALKING"] * 4 + ["TURNING"] * 2 + ["WALKING"] * 6
    d = tug.phase_durations(labels, 0.25)
    assert d["by_activity_s"]["WALKING"] == pytest.approx(2.5)
    assert d["by_activity_s"]["TURNING"] == pytest.approx(0.5)
    assert d["total_s"] == pytest.approx(3.0)
    assert d["n_runs"] == 3


def test_empty_sequence():
    assert tug.phase_boundaries([], 0.25) == []


# ══════════════════════════════════ GUM budget
def test_quantisation_budget_formula():
    b = tug.quantisation_budget(0.24)
    assert b["u_boundary_s"] == pytest.approx(0.24 / np.sqrt(12))
    assert b["u_duration_s"] == pytest.approx(b["u_boundary_s"] * np.sqrt(2))
    assert b["U_duration_s"] == pytest.approx(2 * b["u_duration_s"])
    # with the step of the published system U is about 0.20 s
    assert 0.18 < b["U_duration_s"] < 0.22


def test_budget_scales_with_the_step():
    a = tug.quantisation_budget(0.25)
    b = tug.quantisation_budget(0.50)
    assert b["U_duration_s"] == pytest.approx(2 * a["U_duration_s"])


def test_budget_does_not_depend_on_the_model():
    """The quantisation term depends only on the step, not on the classifier."""
    import inspect
    params = set(inspect.signature(tug.quantisation_budget).parameters)
    assert params == {"step_s", "k"}


# ══════════════════════════════════ i dati dello studio
@needs_results
def test_system_results_are_quantised():
    """Duration / number of windows is constant over all executions."""
    steps = []
    for f in sorted(glob.glob(str(TUG_RESULTS_DIR / "s*" / "s*_s[wp].json"))):
        df = tug.load_system_results(f)
        steps.extend(df.window_step_s.dropna().tolist())
    s = np.array(steps)
    assert len(s) > 300, f"attese molte esecuzioni, trovate {len(s)}"
    assert 0.23 < np.median(s) < 0.26, \
        f"passo di finestra mediano {np.median(s):.3f} s, atteso ~0,24"
    assert s.std() < 0.01, \
        "se il passo non fosse costante i confini non sarebbero quantizzati"


@needs_results
def test_reference_results_are_not_quantised():
    """The manual reference has no step."""
    vals = []
    for f in sorted(glob.glob(str(TUG_RESULTS_DIR / "s*" / "s*_results.csv"))):
        df = tug.load_reference_results(f)
        vals.extend(df.standing_up.dropna().tolist())
    v = np.array(vals)
    assert len(v) > 100
    r = v / 0.24
    off = np.abs(r - np.round(r))
    assert off.mean() > 0.1, \
        "le durate del riferimento manuale non devono essere multipli del passo"


@needs_results
def test_system_and_reference_are_comparable():
    """Same columns, same order of magnitude."""
    sub = sorted(TUG_RESULTS_DIR.glob("s*/s*_sw.json"))[0].parent
    sysr = tug.load_system_results(sub / f"{sub.name}_sw.json")
    ref = tug.load_reference_results(sub / f"{sub.name}_results.csv")
    for c in tug.PHASES:
        assert c in sysr.columns and c in ref.columns
    a = sysr.duration.dropna().median()
    b = ref.duration.dropna().median()
    assert 0.5 < a / b < 2.0, \
        f"durate incompatibili: sistema {a:.1f} s, riferimento {b:.1f} s"


@needs_results
def test_quantisation_is_a_floor_not_the_whole_uncertainty():
    """The quantisation floor is below the observed spread.

    The observed spread also includes model errors and the uncertainty of the
    manual annotation.
    """
    diffs = []
    for f in sorted(glob.glob(str(TUG_RESULTS_DIR / "s*" / "s*_sw.json"))):
        sub = f.rsplit("/", 2)[-2]
        ref_file = TUG_RESULTS_DIR / sub / f"{sub}_results.csv"
        if not ref_file.exists():
            continue
        sysr = tug.load_system_results(f)
        ref = tug.load_reference_results(ref_file)
        n = min(len(sysr), len(ref))
        d = sysr.duration.values[:n] - ref.duration.values[:n]
        diffs.extend([x for x in d if np.isfinite(x)])
    d = np.array(diffs)
    assert len(d) > 100, f"pochi confronti disponibili: {len(d)}"
    floor = tug.quantisation_budget(0.24)["U_duration_s"]
    observed = 1.96 * d.std()
    assert floor < observed, (
        f"il pavimento di quantizzazione ({floor:.2f} s) non puo' superare "
        f"la dispersione osservata ({observed:.2f} s)")
    assert floor / observed > 0.15, (
        f"il pavimento e' solo il {100*floor/observed:.0f} % della "
        f"dispersione: troppo poco perche' il calcolo dica qualcosa")


def test_battery_capacities_declared():
    assert tug.BATTERY_MAH["sp"] > tug.BATTERY_MAH["sw"]
    assert tug.BATTERY_MAH["sw"] == 580
