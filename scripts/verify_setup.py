#!/usr/bin/env python3
"""Check of the repository on a student's or lecturer's machine.

Runs, in order: the environment report, the library import, the broker
check, pytest, then (without --quick) check_course.py on synthetic data, the
Zenodo download and check_course.py on the real data. Each step is isolated,
so one failure does not stop the others.

    python scripts/verify_setup.py --quick         # environment, broker and pytest (Lab 0)
    python scripts/verify_setup.py                 # everything
    python scripts/verify_setup.py --skip-zenodo   # everything but the download

Writes verifica.txt in the repository root: the file to send to the lecturer.
"""
from __future__ import annotations

import argparse
import importlib
import io
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Check this copy, not an iomtlab installed elsewhere (for example an
#: editable install from another folder), which would otherwise be imported.
sys.path.insert(0, str(ROOT / "src"))

REPORT: list[str] = []
RESULTS: list[tuple[str, str, float]] = []


def say(line: str = "") -> None:
    print(line)
    REPORT.append(line)


def rule(title: str) -> None:
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


def run(name: str, cmd: list[str], timeout: int = 1800, cwd: Path = ROOT) -> str:
    """Run cmd, print and record its output and its status; return the status."""
    rule(f"{name}\n$ {' '.join(cmd)}")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        say(out.strip()[-12000:] or "(nessun output)")
        status = "PASS" if p.returncode == 0 else f"FAIL (exit {p.returncode})"
    except subprocess.TimeoutExpired:
        say(f"(interrotto dopo {timeout} s)")
        status = "TIMEOUT"
    except FileNotFoundError as e:
        say(f"(comando non trovato: {e})")
        status = "ASSENTE"
    dt = time.time() - t0
    say(f"--> {name}: {status}  [{dt:.1f} s]")
    RESULTS.append((name, status, dt))
    return status


def environment() -> None:
    rule("Ambiente")
    say(f"python      {sys.version.split()[0]}  ({sys.executable})")
    say(f"piattaforma {platform.platform()}")
    say(f"repo        {ROOT}")
    say(f"venv        {'si' if sys.prefix != sys.base_prefix else 'NO — attivalo'}")
    say(f"spazio      {shutil.disk_usage(ROOT).free / 1e9:.1f} GB liberi")
    for var in ("IOMTLAB_ROOT", "IOMTLAB_DATA_DIR"):
        say(f"{var:11} {os.environ.get(var, '(non impostata)')}")
    say()
    say("Pacchetti:")
    wanted = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch",
              "paho.mqtt", "pytest", "nbmake", "jupyterlab", "iomtlab"]
    for mod in wanted:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "?")
            say(f"  {mod:14} {v}")
        except Exception as e:                       # noqa: BLE001
            say(f"  {mod:14} ASSENTE  ({type(e).__name__})")
    try:
        import torch
        say()
        say(f"torch: threads {torch.get_num_threads()}, "
            f"cuda {torch.cuda.is_available()}")
    except Exception:                                # noqa: BLE001
        say()
        say("torch: ASSENTE; i test dei modelli e i laboratori dal 4 in poi non possono girare")


def broker_state() -> None:
    """Labs 0 and 1 need a broker on localhost:1883: say whether one answers."""
    rule("Broker MQTT (Lab 0 e Lab 1)")
    import socket
    try:
        with socket.create_connection(("localhost", 1883), timeout=1.0):
            say("localhost:1883 risponde: broker presente")
    except OSError:
        say("NESSUN broker su localhost:1883. Installare Mosquitto (mosquitto.org; "
            "macOS: brew install mosquitto; Linux: pacchetto mosquitto) e avviarlo "
            "con `mosquitto -v`. Senza, i Lab 0 e 1 si fermano alla prima cella che lo usa.")


def dataset_state() -> None:
    """Where DATA_DIR points and how many CSV files it holds."""
    rule("Stato del dataset")
    try:
        from iomtlab.config import DATA_DIR                   # type: ignore
        d = Path(DATA_DIR)
    except Exception as e:                                    # noqa: BLE001
        say(f"impossibile leggere iomtlab.config.DATA_DIR: {e}")
        return
    say(f"DATA_DIR = {d}")
    if not d.exists():
        say("la cartella non esiste: il dataset non è stato scaricato")
        return
    csv = sorted(d.rglob("*.csv"))
    say(f"file CSV trovati: {len(csv)}")
    for f in csv[:5]:
        say(f"  {f.relative_to(d)}  {f.stat().st_size / 1024:.0f} KiB")
    if len(csv) > 5:
        say(f"  ... e altri {len(csv) - 5}")


def library_smoke() -> None:
    """Import every subpackage without pytest, to isolate environment errors."""
    rule("Import della libreria")
    buf = io.StringIO()
    ok = True
    with redirect_stdout(buf):
        for mod in ("iomtlab", "iomtlab.signal", "iomtlab.features",
                    "iomtlab.evaluation", "iomtlab.io",
                    "iomtlab.tug", "iomtlab.models", "iomtlab.train",
                    "iomtlab.edge", "iomtlab.uncertainty", "iomtlab.fl",
                    "iomtlab.net"):
            try:
                m = importlib.import_module(mod)
                print(f"  {mod:22} ok")
            except Exception as e:                            # noqa: BLE001
                print(f"  {mod:22} ERRORE  {type(e).__name__}: {e}")
                ok = False
    say(buf.getvalue().rstrip())
    # Which copy was imported: if not this one, the report is about another repo.
    try:
        import iomtlab
        where = Path(iomtlab.__file__).resolve().parent.parent
        say(f"  copia verificata: {where}")
        if where != (ROOT / "src").resolve():
            say(f"  ! attesa {ROOT / 'src'}: c'è un'altra iomtlab installata")
            ok = False
    except Exception:                                         # noqa: BLE001
        pass
    RESULTS.append(("import della libreria", "PASS" if ok else "FAIL", 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-zenodo", action="store_true",
                    help="non scaricare il dataset")
    ap.add_argument("--quick", action="store_true",
                    help="solo ambiente, import e pytest")
    a = ap.parse_args()

    say(f"verifica di iomt-har-labs — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    environment()
    library_smoke()
    broker_state()

    py = [sys.executable]
    run("pytest", py + ["-m", "pytest", "-q", "--no-header"])

    if a.quick:
        summary()
        return 0

    run("check_course.py (dati sintetici)", py + ["scripts/check_course.py"])

    if not a.skip_zenodo:
        run("get_data.py (Zenodo, ~29.5 MB)", py + ["scripts/get_data.py"],
            timeout=1800)
        dataset_state()
        run("check_course.py --real-data",
            py + ["scripts/check_course.py", "--real-data"], timeout=3600)
    else:
        dataset_state()

    summary()
    return 0


def summary() -> None:
    """Print the table of the steps and write verifica.txt."""
    rule("RIEPILOGO")
    for name, status, dt in RESULTS:
        say(f"{status:16} {name}  [{dt:.1f} s]")
    bad = [n for n, s, _ in RESULTS if s != "PASS"]
    say()
    say(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} passati.")
    if bad:
        say("Non passati: " + ", ".join(bad))
    out = ROOT / "verifica.txt"
    out.write_text("\n".join(REPORT), encoding="utf-8")
    print()
    print(f"Rapporto completo in {out}: è il file da mandare al docente.")


if __name__ == "__main__":
    raise SystemExit(main())
