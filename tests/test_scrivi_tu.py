"""Le soluzioni di riferimento degli esercizi «Scrivi tu» (Lab 1-8): coerenti con la libreria."""
import numpy as np
import pytest


def test_quantize_array_matches_quantize_state():
    torch = pytest.importorskip("torch")
    from iomtlab.edge.compress import quantize_array, quantize_state
    from iomtlab.models import ExampleCNN
    torch.manual_seed(0)
    m = ExampleCNN()
    for pc in (True, False):
        qs = quantize_state(m, 8, per_channel=pc)
        q, s = quantize_array(m.conv2.weight.detach().numpy(), 8, pc)
        assert np.allclose(s, qs["conv2"]["scale"].numpy(), rtol=1e-6)
        assert np.mean(q == qs["conv2"]["q"].numpy()) > 0.999     # float32 contro float64: rari pareggi


def test_gap_models_respect_the_contract():
    pytest.importorskip("torch")
    from iomtlab.models import GapCNN, GapCNNFine
    from iomtlab.models.contract import check_model
    for m in (GapCNN(), GapCNNFine()):
        rows = check_model(m)
        assert not [r for r in rows if r[0] == "FALLITO"], rows
    assert [n for n, _ in GapCNNFine().stages()] == ["blocco1", "conv2", "pool2", "media", "out"]


def test_run_fedavg_accepts_an_aggregator():
    pytest.importorskip("torch")
    from iomtlab.fl.aggregate import fedavg
    from iomtlab.fl.rounds import run_fedavg
    from iomtlab.models import SimpleCNN
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 50, 6)).astype("float32"); y = rng.integers(0, 5, 60)
    calls = []
    def agg(states, n):
        calls.append(len(states)); return fedavg(states, n)
    h = run_fedavg(SimpleCNN(), X, y, [np.arange(0, 30), np.arange(30, 60)], X[:10], y[:10],
                   rounds=1, aggregate=agg)
    assert calls == [2] and len(h) == 2
