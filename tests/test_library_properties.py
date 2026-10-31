"""Properties of the library functions the labs rely on.

The first group runs the property checks in tests/_property_checks.py on the
library. The others test one property each, on synthetic data whose answer is
known.
"""
import numpy as np
import pytest
import torch

import _property_checks as C
from iomtlab import evaluation as E
from iomtlab import net, tug
from iomtlab.config import N_CLASSES
from iomtlab.signal import resample, rotations, spectral, timing, windowing
from iomtlab.train import loops


@pytest.mark.parametrize("check,fn", [
    (C.pack_window, net.pack_window),
    (C.remaining_length, net.remaining_length),
    (C.phase_table, tug.phase_table),
    (C.delivery_regularity, timing.delivery_regularity),
    (C.align_to_common_grid, timing.align_to_common_grid),
    (C.label_window, windowing.label_window),
    (C.timing_report, timing.timing_report),
    (C.device_offset, timing.device_offset),
    (C.resample_to_hz, resample.resample_to_hz),
    (C.lowpass_ma, resample.lowpass_ma),
    (C.window_properties, spectral.window_properties),
    (C.rfft_magnitude, spectral.rfft_magnitude),
    (C.fft_cost, spectral.fft_cost),
    (C.make_windows, windowing.make_windows),
    (C.rotate, rotations.rotate),
    (C.tilt_angles, rotations.tilt_angles),
    (C.complementary_filter, rotations.complementary_filter),
    (C.train_epoch, loops.train_epoch),
])
def test_reference_implementation(check, fn):
    check(fn)


def test_framing():
    C.framing(net.frame, net.read_frame)


def test_class_weights():
    C.class_weights(loops.inverse_freq_weights, loops.effective_num_weights)


# ── rotations
def test_rotation_matrix_is_proper():
    R = rotations.rotation_matrix(roll=0.3, pitch=-0.7, yaw=2.0)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rotate_refuses_reflection():
    with pytest.raises(ValueError):
        rotations.rotate(np.zeros((10, 6)), np.diag([1.0, 1.0, -1.0]))


def test_norm_features_do_not_depend_on_orientation():
    from iomtlab.features import norm
    rng = np.random.default_rng(3)
    t = np.arange(50) / 50.0
    X = np.zeros((4, 50, 6))
    for i in range(4):
        X[i, :, 2] = rotations.GRAVITY
        X[i, :, 0] = 2.0 * np.sin(2 * np.pi * (1.0 + i) * t)
        X[i, :, 3:6] = rng.normal(0, 0.5, size=(50, 3))
    Xr = np.stack([rotations.rotate(x, rotations.rotation_matrix(*rng.uniform(-np.pi, np.pi, 3)))
                   for x in X])
    assert np.allclose(norm(X)[0], norm(Xr)[0], rtol=1e-5, atol=1e-5)
    rest = np.zeros((1, 50, 6))
    rest[0, :, 2] = rotations.GRAVITY
    assert norm(rest)[0][0, 0] == pytest.approx(rotations.GRAVITY)


# ── splits and metrics
def test_loso_folds_are_masks_one_subject_each():
    groups = np.repeat([f"s{i:02d}" for i in range(1, 8)], [4, 6, 3, 5, 4, 7, 2])
    folds = list(E.loso_folds(groups))
    assert len(folds) == 7
    seen = []
    for tr, te in folds:
        assert tr.dtype == bool and len(tr) == len(te) == len(groups)
        assert len(set(groups[te])) == 1
        assert not set(groups[tr]) & set(groups[te])
        seen += list(set(groups[te]))
    assert sorted(seen) == sorted(set(groups))
    again = list(E.loso_folds(groups))
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again))


def test_subject_folds_split_subjects():
    groups = np.repeat([f"s{i:02d}" for i in range(1, 13)], 7)
    folds = list(E.subject_folds(groups, 4, 0))
    assert len(folds) == 4
    seen = []
    for tr, te in folds:
        assert not (tr & te).any()
        assert not set(groups[tr]) & set(groups[te])
        seen += list(set(groups[te]))
    assert sorted(seen) == sorted(set(groups))
    with pytest.raises(Exception):
        list(E.subject_folds(groups, 50, 0))


def test_incremental_loso_holds_the_test_subject_out():
    groups = np.repeat([f"s{i:02d}" for i in range(1, 9)], 5)
    seen_n = []
    for n, i, tr, te in E.incremental_loso_subsets(groups, "s03", sizes=[1, 2, 4, 7],
                                                   repeats=3, seed=0):
        assert set(groups[te]) == {"s03"}
        assert "s03" not in set(groups[tr])
        assert len(set(groups[tr])) == n
        assert not (tr & te).any()
        seen_n.append(n)
    # With all the other subjects there is only one possible draw.
    assert seen_n.count(7) == 1
    assert seen_n.count(4) == 3


def test_incremental_loso_is_reproducible():
    groups = np.repeat([f"s{i:02d}" for i in range(1, 9)], 4)
    a = [tuple(sorted(set(groups[tr])))
         for *_, tr, _ in E.incremental_loso_subsets(groups, "s01", seed=7)]
    b = [tuple(sorted(set(groups[tr])))
         for *_, tr, _ in E.incremental_loso_subsets(groups, "s01", seed=7)]
    assert a == b


def test_macro_f1_matches_sklearn_with_absent_classes():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    cases = [
        (rng.integers(0, N_CLASSES, 300), rng.integers(0, N_CLASSES, 300)),
        (np.zeros(50, dtype=np.int64), np.zeros(50, dtype=np.int64)),
        (rng.integers(0, N_CLASSES, 200), rng.integers(0, N_CLASSES - 1, 200)),
        (rng.integers(0, N_CLASSES - 1, 200), rng.integers(0, N_CLASSES, 200)),
    ]
    for yt, yp in cases:
        want = f1_score(yt, yp, labels=list(range(N_CLASSES)), average="macro",
                        zero_division=0)
        assert E.macro_f1(yt, yp) == pytest.approx(want, abs=1e-9)


# ── uncertainty
def test_ece_known_cases():
    from iomtlab.uncertainty import ece
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 20_000)
    correct = (rng.uniform(size=conf.size) < conf).astype(float)
    assert ece(conf, correct) < 0.02
    conf2 = np.full(10_000, 0.99)
    correct2 = np.r_[np.ones(5000), np.zeros(5000)]
    assert ece(conf2, correct2) == pytest.approx(0.49, abs=0.02)
    # Bins are weighted by their share of samples.
    c3 = np.r_[np.full(9_900, 0.60), np.full(100, 0.99)]
    k3 = np.r_[np.ones(9_900), np.zeros(100)]
    assert ece(c3, k3) < 0.45


def test_conformal_coverage_for_several_alpha():
    from iomtlab.uncertainty import conformal_sets
    rng = np.random.default_rng(0)
    K, n_cal, n_te = N_CLASSES, 500, 4000

    def sample(m):
        y = rng.integers(0, K, m)
        p = rng.dirichlet(np.ones(K) * 0.7, size=m)
        p[np.arange(m), y] += 0.8
        return p / p.sum(1, keepdims=True), y

    p_cal, y_cal = sample(n_cal)
    p_te, y_te = sample(n_te)
    for alpha in (0.05, 0.10, 0.20):
        S = np.asarray(conformal_sets(p_cal, y_cal, p_te, alpha), dtype=bool)
        assert S.shape == (n_te, K)
        cov = S[np.arange(n_te), y_te].mean()
        lo = 1 - alpha - 3 / np.sqrt(n_te)
        hi = 1 - alpha + 1 / (n_cal + 1) + 3 / np.sqrt(n_te)
        assert lo <= cov <= hi


# ── federated learning
def test_fedavg_weights_by_samples():
    from iomtlab.fl import fedavg
    a = {"w": torch.ones(4), "b": torch.zeros(2)}
    b = {"w": torch.ones(4) * 3, "b": torch.ones(2)}
    out = fedavg([a], [7])
    assert all(torch.allclose(out[k], a[k]) for k in a)
    out = fedavg([a, b], [1, 3])
    assert torch.allclose(out["w"], torch.full((4,), 2.5), atol=1e-5)


def test_dirichlet_partition_is_a_partition():
    from iomtlab.fl import dirichlet_partition
    y = np.random.default_rng(0).integers(0, N_CLASSES, 1200)
    for beta in (10.0, 1.0, 0.1):
        parts, info = dirichlet_partition(y, 6, beta, 0)
        flat = np.concatenate(parts)
        assert len(flat) == len(y) == len(np.unique(flat))
        assert info["sizes"] == [len(p) for p in parts]


def test_quantize_delta_error_and_zero_delta():
    from iomtlab.fl import quantize_delta
    g = {"w": torch.zeros(8)}
    loc = {"w": torch.tensor([0.1, -0.2, 0.05, 0.0, 0.3, -0.3, 0.02, -0.01])}
    q, s = quantize_delta(loc, g)["w"]
    assert torch.max(torch.abs(q.to(torch.float32) * s - loc["w"])) <= s
    q0, s0 = quantize_delta(dict(g), g)["w"]
    assert torch.all(torch.isfinite(q0.to(torch.float32) * s0))


# ── edge
def test_int8_roundtrip_keeps_scale():
    from iomtlab.edge import pack_int8, unpack_int8
    rng = np.random.default_rng(0)
    for scale in (1.0, 1e-3, 1e3):
        a = rng.normal(size=(1, 16, 25)).astype(np.float32) * scale
        b = np.asarray(unpack_int8(pack_int8(a)))
        assert b.shape == a.shape
        assert np.abs(a - b).max() / np.abs(a).max() < 0.01
    z = np.zeros((1, 4, 4), dtype=np.float32)
    assert np.allclose(unpack_int8(pack_int8(z)), 0.0)
