#!/usr/bin/env python3
"""Run the tests, then every lab notebook end to end.

    python scripts/check_course.py                 # synthetic data
    python scripts/check_course.py --real-data     # data in data/DATA

Runs pytest on the repository, then each labs/labNN_*.ipynb with nbmake. By
default tests and notebooks read a synthetic dataset that make_fake_data.py
writes to build/check/. The notebooks that talk to MQTT (lab00, lab01) need a
broker on localhost:1883; without one they are skipped, with a message.
Writes nothing outside build/check/ and work/.

Exit status is 0 if nothing failed (a skipped notebook is not a failure),
1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "build" / "check"
LAB_RE = re.compile(r"^lab(\d{2})_.*\.ipynb$")
#: Notebooks that need an MQTT broker on localhost:1883.
NEEDS_BROKER = {0, 1}


def run(cmd, env=None) -> tuple[int, str]:
    """Run cmd in the repository root; return (exit status, stdout + stderr)."""
    p = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def last_line(out: str) -> str:
    """The last line of out, the pytest summary."""
    lines = out.strip().splitlines()
    return lines[-1] if lines else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-data", action="store_true",
                    help="usa data/DATA invece dei dati sintetici")
    ap.add_argument("--timeout", type=int, default=600,
                    help="tempo massimo per cella di notebook, in secondi")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from iomtlab.net import broker_available

    py = sys.executable
    if a.real_data:
        data = ROOT / "data" / "DATA"
        if not any(data.glob("s*/*.csv")):
            print("data/DATA è vuota: lancia prima scripts/get_data.py")
            return 1
    else:
        if WORK.exists():
            shutil.rmtree(WORK)
        WORK.mkdir(parents=True)
        data = WORK / "fake_data"
        code, out = run([py, str(ROOT / "scripts" / "make_fake_data.py"), str(data),
                          "--subjects", "8"])   # dal Lab 3: 5 pieghe per soggetto
        if code:
            print(out)
            return 1
    env = dict(os.environ, IOMTLAB_DATA_DIR=str(data))

    results = []                                   # (name, status, info)
    code, out = run([py, "-m", "pytest", "-q"], env=env)
    results.append(("pytest", "PASS" if code == 0 else "FAIL", last_line(out)))

    broker = broker_available()
    for nb in sorted(p for p in (ROOT / "labs").glob("lab*.ipynb") if LAB_RE.match(p.name)):
        k = int(LAB_RE.match(nb.name).group(1))
        name = str(nb.relative_to(ROOT))
        if k in NEEDS_BROKER and not broker:
            results.append((name, "SKIP", "nessun broker MQTT su localhost:1883: "
                                          "avvialo con `mosquitto -d -p 1883`"))
            continue
        code, out = run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--nbmake",
                         f"--nbmake-timeout={a.timeout}", str(nb)], env=env)
        results.append((name, "PASS" if code == 0 else "FAIL", last_line(out)))
        if code:
            print(out)

    # il template compilato su un dataset di forma diversa dal TUG: check_port e il notebook
    ex = ROOT / "projects" / "esempio_sintetico"
    if ex.is_dir():
        p = subprocess.run([py, "check_port.py"], cwd=ex, env=env, capture_output=True, text=True)
        results.append(("projects/esempio_sintetico/check_port.py", "PASS" if p.returncode == 0 else "FAIL",
                        p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""))
        if p.returncode:
            print(p.stdout + p.stderr)
        code, out = run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--nbmake",
                         f"--nbmake-timeout={a.timeout}", str(ex / "progetto.ipynb")], env=env)
        results.append(("projects/esempio_sintetico/progetto.ipynb", "PASS" if code == 0 else "FAIL",
                        last_line(out)))
        if code:
            print(out)

    width = max(len(r[0]) for r in results)
    print()
    for name, status, info in results:
        print(f"{status}  {name:<{width}}  {info}")
    return 0 if all(s != "FAIL" for _, s, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
