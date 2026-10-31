"""The lab figures.

We don't compare pixels. We check that each function runs on the inputs it
gets, and that what it is meant to show is in the data it draws.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from iomtlab.evaluation import (kfold_shuffled, loso_folds, per_subject_scores,
                                subject_folds)
from iomtlab.plots import (per_subject_bars, plot_cost_accuracy, plot_folds,
                           plot_rounds, reliability_diagram)

GROUPS = np.repeat([f"s{i:02d}" for i in range(1, 13)], 7)


def test_plot_folds_on_the_three_schemes():
    for folds in (loso_folds(GROUPS), subject_folds(GROUPS, 4, 0)):
        fig = plot_folds(GROUPS, folds)
        assert fig is not None
        matplotlib.pyplot.close(fig)


def test_plot_folds_shows_split_subjects():
    """The title states how many subjects are split between train and test.

    With a random split over rows nearly every subject ends up in both.
    """
    def declared(fig):
        return int(fig.axes[0].get_title().rsplit("spezzati:", 1)[1])

    fig = plot_folds(GROUPS, kfold_shuffled(len(GROUPS), 4, 0))
    assert declared(fig) >= 10, "a random split over rows splits nearly every subject"
    matplotlib.pyplot.close(fig)

    fig = plot_folds(GROUPS, loso_folds(GROUPS))
    assert declared(fig) == 0
    matplotlib.pyplot.close(fig)


def test_plot_folds_rejects_indices():
    def as_indices(groups):
        groups = np.asarray(groups).astype(str)
        for s in np.unique(groups):
            yield np.flatnonzero(groups != s), np.flatnonzero(groups == s)
    with pytest.raises(ValueError):
        plot_folds(GROUPS, as_indices(GROUPS))


def test_per_subject_bars():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, len(GROUPS))
    p = rng.integers(0, 5, len(GROUPS))
    fig = per_subject_bars(per_subject_scores(GROUPS, y, p), reference=0.3)
    assert len(fig.axes[0].patches) == 12
    matplotlib.pyplot.close(fig)


def test_pareto_front_is_correct():
    """A dominated point is not on the Pareto front."""
    pts = [("a", 1e5, 0.70), ("b", 2e5, 0.80), ("c", 3e5, 0.75),
           ("d", 4e5, 0.90)]
    fig = plot_cost_accuracy(pts)
    lines = [ln for ln in fig.axes[0].lines if ln.get_label() != "_nolegend_"]
    front = [ln for ln in lines if "Pareto" in str(ln.get_label())]
    assert front, "la frontiera deve essere disegnata"
    xs = list(front[0].get_xdata())
    # 'c' costs more than 'b' and scores less: it is dominated
    assert 3e5 not in xs, "un punto dominato e' finito sulla frontiera"
    assert {1e5, 2e5, 4e5} <= set(xs)
    matplotlib.pyplot.close(fig)


def test_cost_accuracy_accepts_dispersion():
    fig = plot_cost_accuracy([("a", 1e5, 0.70, 0.05), ("b", 2e5, 0.72, 0.06)])
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_reliability_diagram_reports_the_same_ece():
    """The ECE in the title equals uncertainty.ece."""
    from iomtlab.uncertainty import ece as ece_fn
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 2000)
    correct = (rng.uniform(size=2000) < conf * 0.8).astype(float)
    fig = reliability_diagram(conf, correct, n_bins=15)
    shown = float(fig.axes[0].get_title().split("=")[1].split()[0])
    # the title shows four decimals, so we compare at that precision
    assert shown == pytest.approx(ece_fn(conf, correct, n_bins=15), abs=5e-5)
    matplotlib.pyplot.close(fig)


def test_reliability_bins_match_the_ece_definition():
    """The diagram and uncertainty.ece use the same bins and weights.

    They are computed by two pieces of code; a different edge convention
    would make them disagree.
    """
    from iomtlab.uncertainty import ece as ece_fn
    rng = np.random.default_rng(1)
    for n_bins in (5, 10, 15, 20):
        conf = rng.uniform(0.0, 1.0, 5000)
        correct = (rng.uniform(size=5000) < conf).astype(float)
        fig = reliability_diagram(conf, correct, n_bins=n_bins)
        shown = float(fig.axes[0].get_title().split("=")[1].split()[0])
        assert shown == pytest.approx(ece_fn(conf, correct, n_bins=n_bins),
                                      abs=5e-5), f"n_bins={n_bins}"
        matplotlib.pyplot.close(fig)


def test_reliability_diagram_on_a_perfect_model():
    conf = np.full(500, 0.9)
    correct = np.zeros(500)
    correct[:450] = 1.0           # 90 % right at confidence 0.9
    fig = reliability_diagram(conf, correct, n_bins=10)
    assert float(fig.axes[0].get_title().split("=")[1].split()[0]) < 1e-9
    matplotlib.pyplot.close(fig)


def test_reliability_diagram_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        reliability_diagram(np.zeros(10), np.zeros(9))


def test_plot_rounds_draws_both_panels():
    """Two panels: against rounds and against bytes."""
    h = {"float32": [(r, 0.5 + 0.03 * r, 1e6 * r) for r in range(1, 11)],
         "int8": [(r, 0.48 + 0.03 * r, 2.5e5 * r) for r in range(1, 11)]}
    fig = plot_rounds(h)
    assert len(fig.axes) == 2
    assert "round" in fig.axes[0].get_xlabel()
    assert "byte" in fig.axes[1].get_xlabel()
    matplotlib.pyplot.close(fig)


def test_plot_rounds_rejects_wrong_row_shape():
    with pytest.raises(ValueError):
        plot_rounds({"a": [(1, 0.5), (2, 0.6)]})


def test_plot_trial_shades_every_phase():
    """One background span per phase of the recording, plus the four lines."""
    import pandas as pd
    lab = ["SEATED"] * 50 + ["WALKING"] * 100 + ["SEATED"] * 50
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 1, (200, 6)),
                      columns=[f"{a}_{s}" for s in ("acc", "gyro") for a in "xyz"])
    df["timestamp"] = np.arange(200) * 10
    df["label"] = lab
    from iomtlab.plots import plot_trial
    fig = plot_trial(df)
    ax = fig.axes[0]
    assert len(ax.lines) == 4
    assert len(ax.patches) == 3                   # seduto, cammina, seduto
    matplotlib.pyplot.close(fig)
