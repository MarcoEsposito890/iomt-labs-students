"""Constants of the course pipeline: paths, sampling, labels, subjects, energy.

The pipeline values (50 Hz, windows of 50 samples with step 25, full-scale
normalisation) are frozen: every lab and every number of the labs
assumes them.
"""

from __future__ import annotations

import os
from pathlib import Path


# --- paths -------------------------------------------------------------------
def _find_root() -> Path:
    """Project root: $IOMTLAB_ROOT, else the first folder holding pyproject.toml
    and src/iomtlab, going up from the working directory or from the package.

    parents[2] of this file would work only for an editable install: with
    pip install . the package sits in site-packages, away from the data.
    Falls back to the working directory.
    """
    env = os.environ.get("IOMTLAB_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # risalgo le cartelle, prima da quella corrente e poi da quella del pacchetto
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for d in (start, *start.parents):
            if (d / "pyproject.toml").is_file() and (d / "src" / "iomtlab").is_dir():
                return d
    return Path.cwd()


REPO_ROOT = _find_root()
#: Folder with the sNN subfolders. $IOMTLAB_DATA_DIR overrides it, for
#: example with the synthetic data of scripts/make_fake_data.py.
DATA_DIR = Path(os.environ.get("IOMTLAB_DATA_DIR", REPO_ROOT / "data" / "DATA"))
#: Results of the deployed TUG system and the manual video reference, 30
#: subjects. They are not in the Zenodo record: they are in data/chapter4/ of
#: the thesis repository, and scripts/get_data.py --from links them here.
TUG_RESULTS_DIR = Path(os.environ.get("IOMTLAB_TUG_DIR",
                                      REPO_ROOT / "data" / "tug-results"))

# --- signal pipeline --------------------------------------------------------
TARGET_HZ = 50.0
WINDOW_SIZE = 50   # campioni, cioè 1.0 s a 50 Hz
WINDOW_SHIFT = 25  # metà finestra: sovrapposizione del 50%

SENSOR_CHANNELS = ["x_acc", "y_acc", "z_acc", "x_gyro", "y_gyro", "z_gyro"]
N_CHANNELS = len(SENSOR_CHANNELS)

# Fondo scala dichiarato dei sensori [m/s², rad/s]: lo uso per normalizzare e per
# quantizzare in int16. Non uso statistiche dei dati, perché il fondo scala è noto
# sul dispositivo e la stessa normalizzazione vale anche quando il modello gira lì.
FULL_SCALE = {
    "sp": {"acc": 78.45317840576172, "gyro": 17.45274543762207},
    "sw": {"acc": 78.4532, "gyro": 34.906586},
}

# --- labels -----------------------------------------------------------------
LABEL_NAMES = ["SEATED", "STANDING_UP", "WALKING", "TURNING", "SITTING_DOWN"]
LABEL_MAP = {name: i for i, name in enumerate(LABEL_NAMES)}
N_CLASSES = len(LABEL_NAMES)

# Classi transitorie: brevi e sottorappresentate, sono loro a produrre le finestre
# ambigue quando etichetto per maggioranza.
TRANSIENT_LABELS = ["STANDING_UP", "SITTING_DOWN", "TURNING"]

# --- subjects ---------------------------------------------------------------
ALL_SUBJECTS = [f"s{i:02d}" for i in range(1, 24)]

# Orientamento del telefono in tasca, per soggetto (da executions_info.csv).
SUBJECT_ORIENTATIONS = {
    "s01": "back_inv", "s02": "front_inv", "s03": "back_inv", "s04": "back_inv",
    "s05": "front",    "s06": "back",      "s07": "front",    "s08": "back_inv",
    "s09": "back_inv", "s10": "back",      "s11": "back_inv", "s12": "front_inv",
    "s13": "back_inv", "s14": "back_inv",  "s15": "back_inv", "s16": "front",
    "s17": "front_inv","s18": "front",     "s19": "back",     "s20": "back",
    "s21": "front",    "s22": "back_inv",  "s23": "back_inv",
}


# --- energy and channel constants -----------------------------------------
# Sono i valori delle slide del blocco 5.1. Li tengo qui perché li usano più
# laboratori: ricopiati in ogni notebook, prima o poi divergono.
# e_mac_J non è stampata in nessun articolo: la ricavo come kappa * f^2 * gamma
# con i valori della Table II di QPART (arXiv:2506.23934).
ENERGY = {
    "e_mac_J": 0.6e-9,        # energia per moltiplicazione-accumulo sul nodo
    "e_bit_lowpower_J": 240e-9,   # 60 mW su 250 kbit/s, IEEE 802.15.4
    "e_bit_mobile_J": 70e-9,      # 700 mW su 10 Mbit/s, radio mobile
    "rate_lowpower_bps": 250e3,
    "rate_mobile_bps": 10e6,
    "usable_fraction": 0.60,      # quota del canale che si riesce davvero a usare
}
