"""The library against the real data.

These run only when the dataset is present. They check properties the labs
take for granted; if one changed, results would be wrong without an error.

Get the dataset from Zenodo (scripts/get_data.py) or from a local copy such
as the Matey-Sanz thesis repository:

    python scripts/get_data.py --from /path/to/thesis-main
"""
from __future__ import annotations

import numpy as np
import pytest

from iomtlab.config import (ALL_SUBJECTS, DATA_DIR, LABEL_NAMES, N_CHANNELS,
                            SENSOR_CHANNELS)
from iomtlab.evaluation import loso_folds, per_subject_scores
from iomtlab.signal.loading import (list_executions, load_execution,
                                    load_executions_info, load_subjects_info)
from iomtlab.signal.timing import device_offset, timing_report

#: These tests assert properties of the published dataset (23 subjects, 223
#: recordings, phone less regular than watch). On the synthetic data of
#: make_fake_data.py they mean nothing, so we skip them when the marker file
#: is present.
_synthetic = (DATA_DIR / "FAKE_DATA.txt").is_file()
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "s01").is_dir() or _synthetic,
    reason=("dati sintetici in " + str(DATA_DIR)) if _synthetic else
           f"dataset assente in {DATA_DIR}: scripts/get_data.py --from <thesis>")


def _all_executions(device="sp", limit=None):
    out = []
    for s in ALL_SUBJECTS:
        for e in list_executions(s, device):
            out.append((s, e))
            if limit and len(out) >= limit:
                return out
    return out


# ══════════════════════════════════ forma e contenuto
def test_dataset_size():
    execs = _all_executions("sp")
    assert len(execs) == 223, f"attese 223 acquisizioni, trovate {len(execs)}"
    assert len(ALL_SUBJECTS) == 23


def test_every_sp_has_its_sw():
    for s in ALL_SUBJECTS:
        a = set(list_executions(s, "sp"))
        b = set(list_executions(s, "sw"))
        assert a == b, f"{s}: acquisizioni spaiate fra dispositivi"


def test_columns_and_labels():
    df = load_execution("s01", 1, "sp")
    for c in SENSOR_CHANNELS + ["timestamp", "label"]:
        assert c in df.columns, f"colonna mancante: {c}"
    assert set(df.label.unique()) <= set(LABEL_NAMES)


def test_all_five_classes_present_in_dataset():
    seen = set()
    for s, e in _all_executions("sp", limit=40):
        seen |= set(load_execution(s, e, "sp").label.unique())
    assert seen == set(LABEL_NAMES), f"classi mancanti: {set(LABEL_NAMES) - seen}"


def test_subjects_have_different_numbers_of_executions():
    """Subjects have different numbers of recordings, so folds can't be balanced."""
    n = [len(list_executions(s, "sp")) for s in ALL_SUBJECTS]
    assert min(n) < max(n), "se fossero tutti uguali il punto non si porrebbe"
    assert min(n) >= 5


# ══════════════════════════════════ temporizzazione
def test_timestamps_are_monotonic():
    for s, e in _all_executions("sp", limit=30):
        t = load_execution(s, e, "sp").timestamp.values
        assert np.all(np.diff(t) >= 0), f"{s}_{e}: timestamp non monotoni"


def test_phone_is_less_regular_than_watch():
    """Same nominal period, very different spread: the phone delivers in bursts."""
    d = {}
    for dev in ("sp", "sw"):
        acc = [np.diff(load_execution(s, e, dev).timestamp.values)
               for s, e in _all_executions(dev, limit=60)]
        d[dev] = np.concatenate(acc).astype(float)
    assert abs(np.median(d["sp"]) - np.median(d["sw"])) <= 1.0, \
        "il periodo nominale dei due dispositivi deve coincidere entro 1 ms"
    assert d["sp"].std() > 2 * d["sw"].std(), \
        "il telefono deve risultare sensibilmente piu' irregolare"
    assert (d["sp"] == 0).mean() > 0.02, \
        "il telefono deve presentare timestamp duplicati"


def test_devices_start_together():
    """The NTP synchronisation stated in the study is visible in the data."""
    offs = []
    for s, e in _all_executions("sp", limit=60):
        o = device_offset(load_execution(s, e, "sp"),
                          load_execution(s, e, "sw"))
        offs.append(abs(o["start_offset_ms"]))
    assert np.median(offs) < 50, \
        "con NTP lo scarto iniziale mediano deve essere di pochi millisecondi"


def test_timing_report_on_real_data():
    r = timing_report(load_execution("s01", 1, "sp"))
    assert r.n_samples > 100
    assert 5 < r.median_dt_ms < 20, "periodo atteso intorno ai 10 ms"
    assert r.duration_s > 3


# ══════════════════════════════════ validazione
def test_loso_covers_every_subject():
    groups = np.concatenate([[s] * len(list_executions(s, "sp"))
                             for s in ALL_SUBJECTS])
    folds = list(loso_folds(groups))
    assert len(folds) == len(ALL_SUBJECTS)
    covered = np.zeros(len(groups), dtype=bool)
    for tr, te in folds:
        assert not (tr & te).any()
        covered |= te
    assert covered.all()


def test_per_subject_scores_shape():
    groups = np.repeat(ALL_SUBJECTS, 4)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, len(groups))
    p = rng.integers(0, 5, len(groups))
    df = per_subject_scores(groups, y, p)
    assert len(df) == len(ALL_SUBJECTS)
    assert set(df.columns) == {"subject", "n", "score"}


# ══════════════════════════════════ metadati
def test_metadata_files():
    ex = load_executions_info()
    su = load_subjects_info()
    assert len(ex) == 223
    assert len(su) == 23
    assert "orientation" in ex.columns
