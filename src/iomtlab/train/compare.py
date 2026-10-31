"""The reference split and the four networks of Lab 4, trained with one recipe.

Lab 3 writes the split; Lab 4 trains the four networks and saves them in
work/models; Labs 5 to 8 read them back with reference_model.

The recipe, the same for every network: seed fixed before the network is
built (so the initial weights are fixed too), 15 epochs, batches of 64,
Adam with learning rate 1e-3, cross-entropy weighted by inverse class
frequency; training on folds 1-4, test on fold 0.
"""
from __future__ import annotations

import time

import numpy as np

from iomtlab.config import REPO_ROOT

__all__ = ["REFERENCE_SPLIT", "MODELS_DIR", "MODEL_NAMES", "RECIPE", "save_reference_split",
           "build_model", "register_model", "fit_reference", "reference_model", "make_loader", "train_windows"]

#: Where Lab 3 writes the folds: 5 subject folds, seed 0.
REFERENCE_SPLIT = REPO_ROOT / "work" / "split.json"
#: Where Lab 4 writes the four trained networks.
MODELS_DIR = REPO_ROOT / "work" / "models"
#: The four networks of Lab 4, from the smallest to the largest.
MODEL_NAMES = ["simple_cnn", "example_cnn", "bottleneck_cnn", "deepconvlstm"]

EPOCHS, BATCH, LR, SEED = 15, 64, 1e-3, 0
#: The course recipe in one place. The seed must be set (set_seed) before the
#: network is built: it fixes the initial weights too.
RECIPE = {"seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
          "class_weights": "inverse_freq"}


def save_reference_split(groups, path=None, k: int = 5, seed: int = 0) -> dict:
    """Write the reference split, k subject folds with seed, and return it as load_split does.

    The folds are those of evaluation.subject_folds(groups, k, seed), stored
    as lists of subjects. path defaults to split.json in the work folder of the
    active dataset (REFERENCE_SPLIT, work/split.json, for TUG).
    """
    from iomtlab.evaluation import subject_folds
    from iomtlab.io import load_split, save_split
    groups = np.asarray(groups, dtype=object).astype(str)
    folds = [sorted(np.unique(groups[te]).tolist()) for _, te in subject_folds(groups, k, seed)]
    from iomtlab.spec import active
    p = save_split(path or active().work() / "split.json", folds, np.unique(groups), seed)
    return load_split(p)


#: Networks added by a project with register_model: name -> constructor.
_REGISTERED: dict = {}


def register_model(name: str, factory) -> None:
    """Make factory(**kw) -> nn.Module available as name to build_model, fit_reference and
    reference_model, with the same recipe as the reference networks. The names of
    MODEL_NAMES cannot be replaced."""
    if name in MODEL_NAMES:
        raise ValueError(f"{name} è una rete di riferimento del corso: scegli un altro nome")
    _REGISTERED[name] = factory


def build_model(name: str, **kw):
    """A new network of MODEL_NAMES or registered with register_model; kw go to the constructor."""
    from iomtlab.models import BottleneckCNN, DeepConvLSTM, ExampleCNN, SimpleCNN
    cls = {"simple_cnn": SimpleCNN, "example_cnn": ExampleCNN,
           "bottleneck_cnn": BottleneckCNN, "deepconvlstm": DeepConvLSTM, **_REGISTERED}
    if name not in cls:
        raise ValueError(f"rete sconosciuta: {name}; disponibili {MODEL_NAMES + list(_REGISTERED)}")
    return cls[name](**kw)


def make_loader(x, y, batch: int = BATCH, shuffle: bool = True, seed: int = SEED):
    """DataLoader over tensors x and labels y; with shuffle the order depends only on seed."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(x, torch.as_tensor(y, dtype=torch.long)), batch_size=batch,
                      shuffle=shuffle, generator=g if shuffle else None)


def train_windows(model, X, y, epochs: int = EPOCHS, seed: int = SEED,
                  class_weights: bool = True, device: str | None = None) -> list[float]:
    """Train model on windows X (W, T, C) in physical units with the recipe; return the losses.

    The same steps as fit_reference after the network is built: loader with
    batches of 64 shuffled by seed, cross-entropy weighted by inverse class
    frequency, Adam 1e-3. For the same initial weights as fit_reference, call
    set_seed(seed) before building the network. device chooses the full
    scale of the input ("sw" for the watch of TUG; None: the default).
    """
    from iomtlab.train.data import to_input
    from iomtlab.train.loops import fit, inverse_freq_weights
    loader = make_loader(to_input(X, device), y, seed=seed)
    wts = inverse_freq_weights(np.asarray(y)) if class_weights else None
    return fit(model, loader, epochs=epochs, weights=wts, lr=LR, seed=seed)


def fit_reference(name: str, W=None, epochs: int = EPOCHS, class_weights: bool = True,
                  seed: int = SEED, save: bool = True, split=None, **model_kw) -> dict:
    """Train network name with the recipe on folds 1-4 and test it on fold 0.

    W are the reference windows (loaded if None). model_kw change the
    architecture, for example width=(16, 32, 32). With save the network is
    written to work/models/<name>.pt, with its cost profile and results.

    Returns a dict with model, losses (one per epoch), seconds (training
    time), logits and pred on the test windows, and results (macro_f1,
    accuracy, per_class_f1, config).
    """
    import torch
    from iomtlab.io import save_model
    from iomtlab.models import layer_profile
    from iomtlab.train.data import load_reference_windows, logits_of, train_test_masks
    from iomtlab.train.loops import set_seed
    from iomtlab.evaluation import confusion_report
    W = W if W is not None else load_reference_windows()
    tr, te = train_test_masks(W["subject"], split)
    set_seed(seed)                               # prima di costruire la rete: fissa anche i pesi iniziali
    model = build_model(name, **model_kw)
    t0 = time.perf_counter()
    losses = train_windows(model, W["X"][tr], W["y"][tr], epochs, seed, class_weights)
    seconds = time.perf_counter() - t0
    z = logits_of(model, W["X"][te])
    pred = z.argmax(1)
    rep = confusion_report(W["y"][te], pred)
    cfg = model.config() if hasattr(model, "config") else {}
    results = {"macro_f1": rep["macro_f1"], "accuracy": rep["accuracy"],
               "per_class_f1": rep["f1"].tolist(), "epochs": epochs,
               "class_weights": class_weights, "seed": seed, "name": name, "config": cfg}
    if save:
        from iomtlab.spec import active
        save_model(active().work() / "models" / f"{name}.pt", model, layer_profile(model), results,
                   note=f"{name}, ricetta di riferimento, piega 0 in prova")
    return {"model": model, "losses": losses, "seconds": seconds, "logits": z, "pred": pred,
            "results": results}


def reference_model(name: str = "example_cnn", path=None):
    """(model, results) of a network saved by Lab 4; trained and saved first if the file is missing."""
    from pathlib import Path
    from iomtlab.io import load_model
    from iomtlab.spec import active
    p = Path(path or active().work() / "models" / f"{name}.pt")
    if not p.exists():
        r = fit_reference(name, save=True)
        return r["model"].eval(), r["results"]
    d = load_model(p)
    model = build_model(name, **d["results"].get("config", {}))
    model.load_state_dict(d["state_dict"])
    return model.eval(), d["results"]
