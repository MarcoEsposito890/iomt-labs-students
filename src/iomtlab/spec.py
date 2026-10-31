"""The description of a dataset in one object, and the dataset the library is working on.

Every lab works on the TUG dataset (Matey-Sanz et al., 2023), and that is
the default: nothing changes for the labs. A final project describes its own
dataset with a DatasetSpec and activates it at the start of the notebook:

    from iomtlab.spec import DatasetSpec, use
    MIO = DatasetSpec(name="ecg", channels=("ecg",), classes=("N", "V", "A"),
                      fs_hz=100.0, window=200, step=100, full_scale=(5.0,),
                      units=("mV",), sample_bits=12, work_dir="projects/ecg/work")
    use(MIO)

From then on the functions that need the number of channels, classes or
samples, the full scale, the class names or the files of work/ read them
from active(): the models, train.to_input, the features, the cuts and the
costs of Labs 5-7, the metrics and the figures. What is not covered:
signal.build_windows and the loading functions read the TUG CSV files (a
project writes its own loader), and federated learning (Lab 9, optional) is not part
of the project template.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

__all__ = ["DatasetSpec", "TUG", "use", "active", "using", "short_label"]


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset of windows (W, T, C) with one class per window.

    name        short identifier, used in messages and file names.
    channels    one name per channel, in the order of the last axis of X.
    classes     one name per class; the label of a window is the index.
    fs_hz       sampling rate of the windows, Hz.
    window      samples per window, T.
    step        samples between the starts of two windows.
    full_scale  the declared full scale of each channel, in its unit: the
                network input is X / full_scale, clipped to [-1, 1]. A
                constant of the sensor, not a statistic of the data. A dict
                {device: tuple} if the dataset has more than one device.
    units       one unit per channel, for the axes of the figures.
    sample_bits bits per raw sample on the node: the raw window is
                window x channels x sample_bits bits.
    triads      groups of three channels of the same vector sensor (indices),
                for the orientation-free norm features; () if there are none.
    triad_names one short name per triad, used in the feature names.
    short       short channel names for the feature names (default: channels).
    class_short short class names for the figures (default: classes).
    work_dir    the folder of windows.npz, split.json, models/, curve.json.
    device      the key of full_scale to use when it is a dict.
    source      where the data come from: reference, licence, version.
    """
    name: str
    channels: tuple
    classes: tuple
    fs_hz: float
    window: int
    step: int
    full_scale: object
    units: tuple = ()
    sample_bits: int = 16
    triads: tuple = ()
    triad_names: tuple = ()
    short: tuple = ()
    class_short: tuple = ()
    work_dir: object = None
    device: str = ""
    source: str = ""
    extra: dict = field(default_factory=dict)

    # ── quantità derivate ────────────────────────────────────────────────
    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def window_s(self) -> float:
        """Duration of a window, s."""
        return self.window / self.fs_hz

    @property
    def window_bits(self) -> int:
        """Bits of one raw window on the node."""
        return self.window * self.n_channels * self.sample_bits

    @property
    def class_bits(self) -> int:
        """Bits to send one class: ceil(log2(K)), at least 1."""
        return max(1, int(math.ceil(math.log2(self.n_classes))))

    def scale(self, device: str | None = None) -> tuple:
        """The full scale of each channel, for device if full_scale is a dict."""
        fs = self.full_scale
        if isinstance(fs, dict):
            fs = fs[device or self.device]
        return tuple(float(v) for v in fs)

    def short_names(self) -> list:
        return list(self.short or self.channels)

    def class_names(self) -> list:
        return list(self.class_short or self.classes)

    def work(self) -> Path:
        """The work folder, created if missing; relative paths start from the repository root."""
        from iomtlab.config import REPO_ROOT
        p = Path(self.work_dir) if self.work_dir else REPO_ROOT / "work"
        if not p.is_absolute():
            p = REPO_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def with_(self, **kw) -> "DatasetSpec":
        """A copy with some fields changed, for example spec.with_(window=100, step=50)."""
        return replace(self, **kw)

    def check(self) -> "DatasetSpec":
        """Raise ValueError if the description is inconsistent; return self."""
        C = self.n_channels
        if C < 1 or self.n_classes < 2:
            raise ValueError("servono almeno un canale e due classi")
        if self.window < 8:
            raise ValueError("la rete di esempio ha tre pooling: servono almeno 8 campioni")
        if not 0 < self.step <= self.window:
            raise ValueError("il passo deve essere fra 1 e la finestra")
        fs = self.full_scale.values() if isinstance(self.full_scale, dict) else [self.full_scale]
        for f in fs:
            if len(f) != C or min(f) <= 0:
                raise ValueError(f"full_scale: servono {C} valori positivi, uno per canale")
        for name, t in (("units", self.units), ("short", self.short)):
            if t and len(t) != C:
                raise ValueError(f"{name}: servono {C} valori, uno per canale")
        if self.class_short and len(self.class_short) != self.n_classes:
            raise ValueError("class_short: un nome per classe")
        if len(self.triad_names) != len(self.triads):
            raise ValueError("triad_names: un nome per terna")
        for t in self.triads:
            if len(t) != 3 or not all(0 <= i < C for i in t):
                raise ValueError(f"terna {t}: tre indici di canale fra 0 e {C - 1}")
        return self


def _tug() -> DatasetSpec:
    from iomtlab.config import (FULL_SCALE, LABEL_NAMES, SENSOR_CHANNELS, TARGET_HZ, WINDOW_SHIFT,
                                WINDOW_SIZE)
    by_dev = {d: (v["acc"],) * 3 + (v["gyro"],) * 3 for d, v in FULL_SCALE.items()}
    return DatasetSpec(
        name="tug", channels=tuple(SENSOR_CHANNELS), classes=tuple(LABEL_NAMES),
        fs_hz=TARGET_HZ, window=WINDOW_SIZE, step=WINDOW_SHIFT, full_scale=by_dev, device="sp",
        units=("m/s²",) * 3 + ("rad/s",) * 3, sample_bits=16,
        triads=((0, 1, 2), (3, 4, 5)), triad_names=("a", "w"),
        short=("ax", "ay", "az", "gx", "gy", "gz"),
        class_short=("seduto", "si alza", "cammina", "gira", "si siede"),
        work_dir=None,
        source="Matey-Sanz, Casteleyn e Granell, Data in Brief 51, 2023; Zenodo 8398688")


#: The dataset of the labs: phone and watch, 6 channels, 5 TUG phases, 1 s windows at 50 Hz.
TUG = _tug().check()
_ACTIVE = [TUG]


def use(spec: DatasetSpec | None) -> DatasetSpec:
    """Make spec the dataset of the library (None: back to TUG); returns it after check()."""
    _ACTIVE[0] = (spec or TUG).check()
    return _ACTIVE[0]


def active() -> DatasetSpec:
    """The dataset the library is working on: TUG unless use() changed it."""
    return _ACTIVE[0]


def short_label(name: str) -> str:
    """The short name of class name in the active dataset (class_short), else name itself."""
    S = active()
    m = dict(zip(S.classes, S.class_names()))
    return m.get(str(name), str(name))


@contextlib.contextmanager
def using(spec: DatasetSpec):
    """with using(spec): ... — spec active only inside the block (tests, comparisons)."""
    old = _ACTIVE[0]
    use(spec)
    try:
        yield spec
    finally:
        _ACTIVE[0] = old
