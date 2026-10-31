"""Loading the raw recordings of the dataset.

One CSV per recording, in data_dir/<subject>/<subject>_<execution>_<device>.csv,
device being "sp" (smartphone) or "sw" (smartwatch). data_dir defaults to
config.DATA_DIR.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from iomtlab.config import DATA_DIR

_FNAME = re.compile(r"^(s\d{2})_(\d{2})_(sp|sw)\.csv$")


def execution_path(subject: str, execution: str | int, device: str,
                   data_dir: Path | str = DATA_DIR) -> Path:
    """Path of the CSV of one recording; execution may be 1 or "01"."""
    exec_id = f"{int(execution):02d}"        # nel nome del file la prova ha sempre due cifre: 1 diventa "01"
    return Path(data_dir) / subject / f"{subject}_{exec_id}_{device}.csv"


def load_execution(subject: str, execution: str | int, device: str,
                   data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load one recording as a DataFrame, one row per sample.

    Columns: x_acc, y_acc, z_acc (m/s^2), x_gyro, y_gyro, z_gyro (rad/s),
    timestamp (UNIX time, ms), label. Raises FileNotFoundError if the file
    is missing.
    """
    path = execution_path(subject, execution, device, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"prova non trovata: {path}")
    return pd.read_csv(path)


def list_executions(subject: str, device: str = "sp",
                    data_dir: Path | str = DATA_DIR) -> list[str]:
    """Ids of the recordings of one subject on one device ("01", "02", ...); [] if none."""
    subject_dir = Path(data_dir) / subject
    if not subject_dir.is_dir():
        return []
    out = []
    for p in sorted(subject_dir.glob("*.csv")):
        m = _FNAME.match(p.name)             # dal nome del file ricavo soggetto, prova e dispositivo
        if m and m.group(3) == device:       # e tengo solo i file del dispositivo richiesto
            out.append(m.group(2))
    return out


def load_subjects_info(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """subjects_info.csv as a DataFrame: one row per subject (age, gender, height, weight, hand)."""
    return pd.read_csv(Path(data_dir) / "subjects_info.csv")


def load_executions_info(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """executions_info.csv as a DataFrame: one row per recording (phone orientation, turns)."""
    return pd.read_csv(Path(data_dir) / "executions_info.csv")
