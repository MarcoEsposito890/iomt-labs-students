"""The dataset description: TUG by default, a project dataset when activated."""
import matplotlib
matplotlib.use("Agg")
import filecmp
from pathlib import Path

import numpy as np
import pytest
import torch

from iomtlab.spec import TUG, DatasetSpec, active, short_label, use, using

ROOT = Path(__file__).resolve().parents[1]
ONE = DatasetSpec(name="uno", channels=("s",), classes=("a", "b", "c"), fs_hz=100.0, window=200,
                  step=100, full_scale=(4.0,), sample_bits=12, class_short=("A", "B", "C"))


def test_tug_is_the_default_and_matches_config():
    from iomtlab.config import FULL_SCALE, N_CHANNELS, N_CLASSES, WINDOW_SIZE
    assert active() is TUG
    assert (TUG.n_channels, TUG.n_classes, TUG.window) == (N_CHANNELS, N_CLASSES, WINDOW_SIZE)
    assert TUG.window_bits == 4800 and TUG.class_bits == 3
    assert TUG.scale() == (FULL_SCALE["sp"]["acc"],) * 3 + (FULL_SCALE["sp"]["gyro"],) * 3
    assert TUG.scale("sw")[3] == FULL_SCALE["sw"]["gyro"]


def test_to_input_is_unchanged_for_tug():
    from iomtlab.signal import normalize_full_scale
    from iomtlab.train import to_input
    X = np.random.default_rng(0).normal(0, 30, (5, 50, 6)).astype(np.float32)
    old = normalize_full_scale(X.reshape(-1, 6), "sp").reshape(X.shape).transpose(0, 2, 1)
    assert torch.equal(to_input(X), torch.from_numpy(np.ascontiguousarray(old)))


def test_use_and_using():
    with using(ONE):
        assert active() is ONE and short_label("b") == "B"
    assert active() is TUG and short_label("WALKING") == "cammina"
    use(ONE)
    try:
        assert active() is ONE
    finally:
        use(None)
    assert active() is TUG


@pytest.mark.parametrize("kw", [dict(full_scale=(4.0, 1.0)), dict(window=4), dict(step=0),
                                dict(classes=("a",)), dict(triads=((0, 1, 2),), triad_names=("x",)),
                                dict(triad_names=("x",))])
def test_check_rejects_inconsistent_specs(kw):
    with pytest.raises(ValueError):
        ONE.with_(**kw).check()


def test_library_follows_the_active_dataset():
    from iomtlab.edge.cuts import cut_profile
    from iomtlab.evaluation import macro_f1
    from iomtlab.features import extract
    from iomtlab.models import BottleneckCNN, ExampleCNN, SimpleCNN, layer_profile
    from iomtlab.train import inverse_freq_weights, to_input
    from iomtlab.uncertainty import energy_constants, link_load
    with using(ONE):
        X = np.full((3, 200, 1), 8.0, dtype=np.float32)
        x = to_input(X)
        assert tuple(x.shape) == (3, 1, 200) and float(x.max()) == 1.0     # 8 / 4, tagliato a 1
        for cls in (ExampleCNN, BottleneckCNN, SimpleCNN):
            assert tuple(cls()(x).shape) == (3, 3)
        r = cut_profile(ExampleCNN())
        assert r[0]["bits"] == 200 * 12 and r[-1]["bits"] == 2
        assert r[-1]["mac_head"] == layer_profile(ExampleCNN()).macs
        F, names = extract(X, blocks=("amplitude", "dispersion", "cross_axis", "norm"))
        assert names == ["mean_s", "min_s", "max_s", "rms_s", "std_s", "iqr_s"] and F.shape == (3, 6)
        assert len(inverse_freq_weights(np.array([0, 1, 1]))) == 3
        assert macro_f1([0, 1, 2], [0, 1, 2]) == 1.0
        assert energy_constants()["e_raw_J"] == pytest.approx(2400 * 240e-9)
        assert link_load(1.0)["bit_s"] == 2 * 8 * 18
    with pytest.raises(ValueError):
        to_input(np.zeros((1, 50, 1)))                                      # TUG di nuovo: 6 canali


def test_tug_features_unchanged():
    from iomtlab.features import extract
    F, names = extract(np.random.default_rng(0).normal(size=(4, 50, 6)),
                       blocks=("amplitude", "dispersion", "dynamics", "norm"))
    assert len(names) == 52 and names[-4:] == ["mean_anorm", "std_anorm", "mean_wnorm", "std_wnorm"]


def test_project_dataset_needs_its_loader(tmp_path):
    from iomtlab.train import load_reference_windows
    with using(ONE.with_(work_dir=tmp_path)):
        with pytest.raises(FileNotFoundError):
            load_reference_windows()


def test_example_project_is_a_copy_of_the_template():
    t, e = ROOT / "projects" / "_template", ROOT / "projects" / "esempio_sintetico"
    if not (t.is_dir() and e.is_dir()):
        pytest.skip("projects/ non ancora pubblicato")
    assert filecmp.cmp(t / "check_port.py", e / "check_port.py", shallow=False)
