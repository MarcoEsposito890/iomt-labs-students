"""Tests of the Lab 4 and Lab 5 additions: the example networks, the data, the compression."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import torch

from iomtlab import radio
from iomtlab.edge.compress import (NODE, dequantize_state, four_tests, int8_bytes, kd_loss,
                                   prune_magnitude, quantize_state, shrink_example_cnn,
                                   update_on_link, weight_error)
from iomtlab.models import BottleneckCNN, ExampleCNN, count_parameters, layer_profile
from iomtlab.models.memory import peak_activation_bytes


def _out(model):
    return {l.name: l.out_elements for l in layer_profile(model).layers}


# ── the networks: the numbers of the theory ────────────────────────────────
def test_example_cnn_matches_the_theory():
    m = ExampleCNN()
    p = layer_profile(m)
    assert count_parameters(m) == p.parameters == 56_805
    assert p.macs == 574_656
    assert _out(m)["conv1"] == 1600 and _out(m)["pool2"] == 768
    assert peak_activation_bytes(m) == 9600            # 9,4 KiB


def test_bottleneck_cnn_matches_the_theory():
    m = BottleneckCNN()
    p = layer_profile(m)
    assert count_parameters(m) == 57_385 and p.macs == 580_800
    assert _out(m)["bottleneck"] == 48


@pytest.mark.parametrize("cls", [ExampleCNN, BottleneckCNN])
def test_stages_compose_to_forward(cls):
    torch.manual_seed(0)
    m = cls().eval()
    x = torch.randn(3, 6, 50)
    y = x
    for _, f in m.stages():
        y = f(y)
    assert torch.allclose(y, m(x))


# ── the data of Labs 4-8 ───────────────────────────────────────────────────
def test_to_input_divides_by_full_scale_and_transposes():
    from iomtlab.config import FULL_SCALE
    from iomtlab.train import to_input
    X = np.zeros((2, 50, 6), np.float32)
    X[:, :, 0] = FULL_SCALE["sp"]["acc"] / 2
    x = to_input(X)
    assert tuple(x.shape) == (2, 6, 50) and torch.allclose(x[:, 0], torch.full((2, 50), 0.5))


def test_train_test_masks_take_fold_0():
    from iomtlab.train import train_test_masks
    split = {"folds": [["s01", "s02"], ["s03"], ["s04"]]}
    g = np.array(["s01", "s03", "s02", "s04", "s01"], dtype=object)
    tr, te = train_test_masks(g, split)
    assert te.tolist() == [True, False, True, False, True] and (tr == ~te).all()


def test_fit_reference_is_reproducible(tmp_path):
    from iomtlab.io import load_model
    from iomtlab.train import build_model, fit_reference, reference_model
    rng = np.random.default_rng(0)
    W = {"X": rng.normal(size=(120, 50, 6)).astype(np.float32), "y": np.arange(120) % 5,
         "subject": np.array([f"s{i % 6:02d}" for i in range(120)], dtype=object)}
    split = {"folds": [["s00", "s01"], ["s02", "s03"], ["s04", "s05"]]}
    a = fit_reference("example_cnn", W, epochs=1, save=False, split=split, width=(4, 8, 8))
    b = fit_reference("example_cnn", W, epochs=1, save=False, split=split, width=(4, 8, 8))
    assert np.allclose(a["logits"], b["logits"]) and a["losses"] == b["losses"]
    # una rete salvata con la sua configurazione si ricostruisce identica
    from iomtlab.io import save_model
    p = save_model(tmp_path / "example_cnn", a["model"], results=a["results"])
    m, res = reference_model("example_cnn", path=p)
    assert m.config()["width"] == [4, 8, 8] and res["config"] == a["results"]["config"]
    x = torch.randn(2, 6, 50)
    assert torch.allclose(m(x), a["model"].eval()(x))
    assert build_model("bottleneck_cnn").bottleneck.out_channels == 4


# ── radio and update ───────────────────────────────────────────────────────
def test_radio_frames_and_air_bytes():
    assert radio.frames(116) == 1 and radio.frames(117) == 2 and radio.frames(0) == 0
    assert radio.air_bytes(600) == 600 + 6 * 17


def test_update_on_link_gives_the_numbers_of_the_deck():
    i8 = update_on_link(int8_bytes(ExampleCNN())["total_B"], header_B=17, nodes=100)
    assert i8["bytes"] == 58_408 and i8["frames"] == 504 and round(i8["air_s"], 2) == 2.14
    f32 = update_on_link(4 * 56_805, header_B=17, nodes=100)
    assert f32["frames"] == 1959 and round(f32["air_s"], 2) == 8.34
    full = update_on_link(4 * 56_805, nodes=100)          # trame piene da 127 B, senza intestazioni
    assert full["frames"] == 1790 and round(full["air_s"], 1) == 7.3


# ── quantisation, pruning, distillation ────────────────────────────────────
def test_quantisation_error_within_half_a_step():
    torch.manual_seed(0)
    m = ExampleCNN()
    for bits in (8, 4):
        for per_channel in (True, False):
            rows = weight_error(m, quantize_state(m, bits, per_channel))
            assert all(r["max_su_mezzo_passo"] <= 1 + 1e-4 for r in rows)
    m8 = dequantize_state(quantize_state(m, 8), m)
    x = torch.randn(4, 6, 50)
    assert (m8(x) - m.eval()(x)).abs().max() < 0.05


def test_unstructured_pruning_zeroes_the_share():
    torch.manual_seed(0)
    m = prune_magnitude(ExampleCNN(), 0.5)
    w = m.conv2.weight
    assert abs(float((w == 0).float().mean()) - 0.5) < 0.01


def test_structured_pruning_shrinks_to_an_equivalent_network():
    torch.manual_seed(0)
    p = prune_magnitude(ExampleCNN(), 0.5, structured=True)
    s = shrink_example_cnn(p)
    assert s.config()["width"] == [16, 32, 32] and s.config()["hidden"] == 32
    x = torch.randn(4, 6, 50)
    assert torch.allclose(p(x), s(x), atol=1e-5)
    assert layer_profile(s).macs < layer_profile(p).macs / 3


def test_kd_loss_is_zero_when_the_student_copies_the_teacher():
    z = torch.randn(8, 5) * 5
    y = torch.randint(0, 5, (8,))
    assert kd_loss(z, z, y, T=4, alpha=1.0).abs() < 1e-6
    assert kd_loss(torch.zeros(8, 5), z, y, T=4, alpha=1.0) > 0


# ── the four tests ─────────────────────────────────────────────────────────
def test_four_tests_give_the_numbers_of_the_theory():
    r = four_tests(4 * 56_805, 574_656, 9600, 3)
    assert r["passes"] and round(1e3 * r["t_s"], 1) == 14.4
    assert round(r["days"]) == 105 and r["nodes"] == 520
    raw = four_tests(0, 0, 0, 4800)
    assert round(raw["days"]) == 45 and raw["nodes"] == 13
    assert four_tests(0, 0, 0, 4800, {"e_bit_J": 60e-9})["days"] > r["days"]
    slow = four_tests(4 * 56_805, 574_656, 9600, 3, {**NODE, "macs_per_s": 4e6})
    assert not slow["fast"] and not slow["passes"]


def test_the_figures_return_a_figure():
    from iomtlab.edge.compress_figures import (plot_bits_f1, plot_distillation, plot_four_tests,
                                               plot_model_bytes, plot_pruning)
    from iomtlab.evaluation_figures import plot_confidence, plot_losses
    assert plot_model_bytes({"a float32": 1000, "b int8": 250}, {"classe 1": 800}) is not None
    assert plot_bits_f1([8, 4, 2], [0.8, 0.7, 0.2], 0.8) is not None
    assert plot_pruning([0.3, 0.5], {"strutturato": [0.7, 0.5], "strutturato, riaddestrato": [0.8, 0.7]}) is not None
    assert plot_distillation([1, 2], [0.7, 0.72], [0.01, 0.02], 0.68, 0.01, 0.8) is not None
    rows = [{"label": "x", "model_B": 1000, **four_tests(1000, 1000, 100, 3)}]
    assert plot_four_tests(rows, NODE) is not None
    assert plot_losses({"a": [1.6, 1.0, 0.5]}) is not None
    assert plot_confidence(np.array([0.5, 0.9, 0.99]), np.array([False, True, True]), threshold=0.9) is not None


def test_listening_shortens_the_battery():
    base = four_tests(4 * 56_805, 574_656, 9600, 3)
    l1 = four_tests(4 * 56_805, 574_656, 9600, 3, {"listen_fraction": 0.01})
    assert round(base["days"]) == 105 and round(l1["days"]) == 70     # 0,6 mW in più su 1,19
    stud0 = four_tests(4264, 44_976, 700, 3)
    stud1 = four_tests(4264, 44_976, 700, 3, {"listen_fraction": 0.01})
    assert stud0["days"] / base["days"] > 2 > stud1["days"] / l1["days"]   # il vantaggio dello studente si riduce
