"""The contract of a project model: MAC counting, registration, check_model."""
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from iomtlab.models import layer_profile, uncounted_modules
from iomtlab.models.contract import check_model

ROOT = Path(__file__).resolve().parents[1]


class Rec(nn.Module):
    def __init__(self, kind=nn.GRU, bidirectional=False, layers=1):
        super().__init__()
        self.r = kind(6, 32, num_layers=layers, batch_first=True, bidirectional=bidirectional)
        self.f = nn.Linear(32 * (2 if bidirectional else 1), 5)

    def stages(self):
        return [("r", lambda x: self.r(x.transpose(1, 2))[0][:, -1]), ("out", self.f)]

    def forward(self, x):
        for _, g in self.stages():
            x = g(x)
        return x


@pytest.mark.parametrize("kind,gates", [(nn.GRU, 3), (nn.LSTM, 4)])
def test_recurrent_macs(kind, gates):
    assert layer_profile(Rec(kind)).macs == gates * (6 * 32 + 32 * 32) * 50 + 32 * 5
    two = layer_profile(Rec(kind, layers=2)).macs - 32 * 5
    assert two == gates * (6 * 32 + 32 * 32) * 50 + gates * (32 * 32 + 32 * 32) * 50
    bi = layer_profile(Rec(kind, bidirectional=True)).macs - 64 * 5
    assert bi == 2 * gates * (6 * 32 + 32 * 32) * 50


def test_custom_block_counts_itself_and_hides_its_children():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.MultiheadAttention(6, 2, batch_first=True)

        def macs_per_window(self, inp, out):
            return 1234

        def forward(self, x):
            h = x.transpose(1, 2)
            return self.a(h, h, h)[0].mean(1)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.b, self.f = Block(), nn.Linear(6, 5)

        def forward(self, x):
            return self.f(self.b(x))

    assert layer_profile(Net()).macs == 1234 + 30 and uncounted_modules(Net()) == []
    raw = nn.Module()
    raw.a = nn.MultiheadAttention(6, 2)
    assert uncounted_modules(raw) == ["a"]


def test_reference_macs_unchanged():
    from iomtlab.models import DeepConvLSTM, ExampleCNN
    assert layer_profile(ExampleCNN()).macs == 574_656
    assert layer_profile(DeepConvLSTM()).macs == 14_637_440


def test_register_model():
    from iomtlab.train import build_model, register_model
    register_model("rec_test", Rec)
    assert isinstance(build_model("rec_test"), Rec)
    with pytest.raises(ValueError):
        register_model("example_cnn", Rec)


def test_check_model_passes_and_fails():
    rows = check_model(Rec())
    assert [r[0] for r in rows if r[0] == "FALLITO"] == []
    assert any("float32" in r[1] for r in rows)                  # la GRU resta in float32

    class Wrong(Rec):
        def stages(self):                                         # le fasi non ricompongono forward
            return [("r", lambda x: self.r(x.transpose(1, 2))[0][:, 0]), ("out", self.f)]

        def forward(self, x):
            return self.f(self.r(x.transpose(1, 2))[0][:, -1])
    assert any(r[0] == "FALLITO" and "confine" in r[1] for r in check_model(Wrong()))
    assert check_model(nn.Linear(3, 2))[0][0] == "FALLITO"


@pytest.mark.parametrize("project", ["_template", "esempio_sintetico"])
def test_project_models_respect_the_contract(project):
    from iomtlab.spec import using
    d = ROOT / "projects" / project
    if not d.is_dir():
        pytest.skip("projects/ non ancora pubblicato")
    spec_mod = importlib.util.spec_from_file_location(f"spec_{project}", d / "spec.py")
    sm = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(sm)
    with using(sm.SPEC):
        mm_spec = importlib.util.spec_from_file_location(f"model_{project}", d / "model.py")
        mm = importlib.util.module_from_spec(mm_spec)
        mm_spec.loader.exec_module(mm)
        from iomtlab.train import build_model
        rows = check_model(build_model(mm.NAME))
    assert all(r[0] != "FALLITO" for r in rows), rows
