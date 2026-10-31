"""Tests of the models and of the cost profile."""
import pytest

torch = pytest.importorskip("torch")

from iomtlab.config import N_CHANNELS, N_CLASSES, WINDOW_SIZE
from iomtlab.models import (DeepCNN, DeepConvLSTM, EarlyExitCNN, SimpleCNN,
                            count_parameters, federated_round_bytes,
                            layer_profile, model_size_bytes)

BATCH = 4
#: The three models of Lab 4, largest first.
LAB4_MODELS = [DeepConvLSTM, DeepCNN, SimpleCNN]


def _x():
    return torch.zeros(BATCH, N_CHANNELS, WINDOW_SIZE)


@pytest.mark.parametrize("Model", LAB4_MODELS)
def test_output_shape(Model):
    assert Model()(_x()).shape == (BATCH, N_CLASSES)


def test_cost_decreases_from_largest_to_smallest():
    """Parameters and MACs decrease from DeepConvLSTM to SimpleCNN."""
    pars, macs = [], []
    for M in LAB4_MODELS:
        c = layer_profile(M())
        pars.append(c.parameters)
        macs.append(c.macs)
    assert pars == sorted(pars, reverse=True)
    assert macs == sorted(macs, reverse=True)


def test_reference_counts():
    """Reference values quoted in the labs and in the slides."""
    assert count_parameters(SimpleCNN()) == 21_829
    assert count_parameters(DeepCNN()) == 63_941
    assert count_parameters(DeepConvLSTM()) == 295_685
    assert layer_profile(SimpleCNN()).macs == 209_664
    assert layer_profile(DeepCNN()).macs == 3_168_320
    assert layer_profile(DeepConvLSTM()).macs == 14_637_440


def test_simplecnn_layer_split():
    """The convolution dominates the MACs, the dense layers the parameters."""
    c = layer_profile(SimpleCNN())
    conv = next(l for l in c.layers if l.kind == "conv1d")
    lin = [l for l in c.layers if l.kind == "linear"]
    assert conv.macs / c.macs > 0.9
    assert sum(l.parameters for l in lin) / c.parameters > 0.8


def test_pooling_is_the_narrowest_internal_cut():
    """The internal bottleneck is the global pooling output: 64 values."""
    layers = layer_profile(SimpleCNN()).layers[:-1]      # the last layer is the decision
    cut = min(layers, key=lambda l: l.out_elements)
    assert cut.name == "pool" and cut.out_elements == 64
    assert cut.out_bytes() == 256
    assert cut.out_bytes(1) == 64          # after int8 quantisation


def test_size_and_federated_traffic():
    m = SimpleCNN()
    assert model_size_bytes(m) == 87_316
    assert model_size_bytes(m, 1) == 21_829
    # 100 rounds, 20 clients, float32: about 350 MB
    assert 340e6 < 100 * federated_round_bytes(m, 20) < 360e6


def test_early_exit_head_is_cheap():
    m = EarlyExitCNN()
    e, f = m(_x(), return_exit=True)
    assert e.shape == f.shape == (BATCH, N_CLASSES)
    assert count_parameters(m.exit_head) == 325     # 64*5 + 5
