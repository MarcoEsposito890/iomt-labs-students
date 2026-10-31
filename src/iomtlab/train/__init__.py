"""Training: the loop, the seeds, the class weights (Labs 4, 5, 8), and the
reference split written by Lab 3 (train.compare).

Everything lives in iomtlab.train.loops and is re-exported here, so
notebooks write ``from iomtlab.train import fit``.
"""
from iomtlab.train.loops import (effective_num_weights, evaluate,
                                 first_batch_hash, fit, inverse_freq_weights,
                                 predict, set_seed, train_epoch)
from iomtlab.train.compare import (MODEL_NAMES, MODELS_DIR, RECIPE, REFERENCE_SPLIT,
                                   build_model, fit_reference, register_model, make_loader, reference_model,
                                   save_reference_split, train_windows)
from iomtlab.train.data import (WINDOWS_FILE, load_reference_windows, logits_of,
                                reference_split, to_input, train_test_masks)

__all__ = [
    "set_seed", "fit", "train_epoch", "predict", "evaluate",
    "first_batch_hash", "inverse_freq_weights", "effective_num_weights",
    "REFERENCE_SPLIT", "RECIPE", "save_reference_split", "train_windows",
    "MODEL_NAMES", "MODELS_DIR", "build_model", "register_model",
    "fit_reference", "make_loader", "reference_model", "WINDOWS_FILE",
    "load_reference_windows", "logits_of", "reference_split", "to_input", "train_test_masks",
]
