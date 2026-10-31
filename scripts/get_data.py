#!/usr/bin/env python3
"""Download the course dataset from Zenodo into data/DATA.

Matey-Sanz, Casteleyn, Granell (2023), Data in Brief 51.
DOI 10.5281/zenodo.8398688, CC-BY 4.0.

The archive is found through the Zenodo REST API instead of a fixed file
URL, so a new version of the record does not break the script. The checksum
published in the record is verified before extraction, so every student
works on byte-identical data and the numbers of the labs hold.

    python scripts/get_data.py                 # download and extract
    python scripts/get_data.py --check         # verify an existing copy
    python scripts/get_data.py --force         # download again
    python scripts/get_data.py --from PATH     # use a local copy (thesis repository)
"""
from __future__ import annotations

import argparse
import re
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD_ID = "8398688"
API = f"https://zenodo.org/api/records/{RECORD_ID}"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARCHIVE = DATA / "_download.zip"

#: Expected after extraction. Checked so that a partial or wrong archive is
#: caught here instead of surfacing as a confusing error in Lab 2.
EXPECTED_SUBJECTS = 23
EXPECTED_RECORDINGS = 446    # 223 recordings x 2 devices, inside sNN/
EXPECTED_INDEX = "executions_info.csv"


def _human(n: int) -> str:
    """n bytes as a readable size."""
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def fetch_metadata() -> dict:
    """Ask Zenodo for the record and return url, name, size, checksum, version of its largest file."""
    with urllib.request.urlopen(API, timeout=30) as r:
        record = json.load(r)
    files = record.get("files", [])
    if not files:
        raise RuntimeError("il record Zenodo non elenca file")
    f = max(files, key=lambda x: x.get("size", 0))
    return {
        "url": f["links"]["self"],
        "name": f.get("key", "dataset.zip"),
        "size": f.get("size", 0),
        "checksum": f.get("checksum", ""),      # e.g. "md5:0123abcd..."
        "version": record.get("metadata", {}).get("version", "unknown"),
    }


def download(url: str, dest: Path, size: int = 0) -> None:
    """Download url to dest, printing the progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    what = _human(size) if size else "l'archivio"
    print(f"  scarico {what} ...")
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as out:
        done = 0
        while chunk := r.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if size:
                pct = 100 * done / size
                print(f"\r  {pct:5.1f} %  {_human(done)}", end="", flush=True)
    print()


def verify(path: Path, checksum: str) -> bool:
    """True if the archive matches the checksum published in the record (or none is published)."""
    if not checksum:
        print("  ! il record non pubblica un checksum: verifica saltata")
        return True
    algo, _, expected = checksum.partition(":")
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    got = h.hexdigest()
    ok = got == expected
    print(f"  {algo}: {'ok' if ok else 'DIVERSO'}")
    if not ok:
        print(f"    atteso   {expected}\n    ottenuto {got}")
    return ok


def extract(archive: Path, dest: Path) -> None:
    """Unzip archive into dest."""
    print("  estraggo ...")
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)


def normalise(dest: Path) -> Path:
    """Move the folder that holds the sNN subfolders to dest/DATA and return it.

    Zenodo archives usually carry a top-level wrapper folder whose name
    changes between versions; the library expects data/DATA/sNN.
    """
    target = dest / "DATA"
    if target.is_dir() and any(target.glob("s*")):
        return target
    for cand in dest.rglob("*"):
        if cand.is_dir() and len(list(cand.glob("s[0-9][0-9]"))) >= 5:
            if cand != target:
                if target.exists():
                    shutil.rmtree(target)
                cand.rename(target)
            return target
    raise RuntimeError("nell'archivio non c'è una cartella con le sottocartelle sNN")


def check(data_dir: Path) -> bool:
    """Count subjects, recordings and index file in data_dir; True if all are as expected."""
    if not data_dir.is_dir():
        print(f"  ! {data_dir} non esiste")
        return False
    subjects = sorted(p for p in data_dir.glob("s[0-9][0-9]") if p.is_dir())
    # salto le cartelle nascoste: una copia presa da un repository con notebook
    # porta con sé .ipynb_checkpoints, che aggiungerebbe file al conteggio
    rec = [p for p in data_dir.rglob("*.csv")
           if p.parent != data_dir and not p.parent.name.startswith(".")]
    index = (data_dir / EXPECTED_INDEX).is_file()
    print(f"  soggetti: {len(subjects):3d}  (attesi {EXPECTED_SUBJECTS})")
    print(f"  file CSV: {len(rec):3d}  (attesi {EXPECTED_RECORDINGS})")
    print(f"  {EXPECTED_INDEX}: {'presente' if index else 'ASSENTE'}")
    ok = (len(subjects) == EXPECTED_SUBJECTS
          and len(rec) == EXPECTED_RECORDINGS and index)
    if not ok:
        print("  ! la copia è incompleta: rilancia con --force")
    return ok


# ══════════════════════════════════════════════════ local source
#: The Matey-Sanz thesis repository (github.com/matey97/thesis) holds the same
#: dataset in data/chapter2/, with the same sNN/ layout and executions_info.csv.
#: It is the way in on a network that blocks zenodo.org.
THESIS_SUBDIR = "data/chapter2"


def use_local(src: Path, link: bool = True) -> Path:
    """Point data/DATA to a local copy of the dataset, such as the thesis repository.

    The counts are checked first; then data/DATA becomes a symlink to it (a
    copy with link=False). Exits with a message if the copy is not the
    expected one. Returns data/DATA.

        python scripts/get_data.py --from /path/to/thesis-main
    """
    src = Path(src).expanduser().resolve()
    if (src / THESIS_SUBDIR).is_dir():
        src = src / THESIS_SUBDIR
    if not (src / EXPECTED_INDEX).is_file():
        raise SystemExit(f"{src} non contiene {EXPECTED_INDEX}: "
                         f"non è una copia del dataset")
    subs = sorted(d.name for d in src.iterdir()
                  if d.is_dir() and re.fullmatch(r"s\d{2}", d.name))
    recs = sum(1 for _ in src.glob("s*/*.csv"))
    if len(subs) != EXPECTED_SUBJECTS or recs != EXPECTED_RECORDINGS:
        raise SystemExit(
            f"conteggi diversi dall'atteso: {len(subs)} soggetti "
            f"(attesi {EXPECTED_SUBJECTS}), {recs} acquisizioni "
            f"(attese {EXPECTED_RECORDINGS}): non è la copia attesa")
    DATA.mkdir(parents=True, exist_ok=True)
    dest = DATA / "DATA"
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink():
            dest.unlink()
        else:
            raise SystemExit(f"{dest} esiste già: va rimosso prima")
    if link:
        dest.symlink_to(src)
    else:
        shutil.copytree(src, dest)
    print(f"{len(subs)} soggetti, {recs} acquisizioni -> {dest}"
          f"{' (collegamento)' if link else ''}")

    # i risultati del sistema TUG in esercizio non sono nel record Zenodo, ma
    # stanno nello stesso repository della tesi: se ci sono, collego anche quelli
    tug = src.parent / "chapter4" / "system-results"
    if tug.is_dir():
        tdest = DATA / "tug-results"
        if tdest.is_symlink():
            tdest.unlink()
        if not tdest.exists():
            if link:
                tdest.symlink_to(tug)
            else:
                shutil.copytree(tug, tdest)
            n = len([d for d in tug.iterdir() if d.is_dir()])
            print(f"{n} soggetti di risultati TUG -> {tdest}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="scarica di nuovo")
    ap.add_argument("--check", action="store_true", help="verifica soltanto")
    ap.add_argument("--keep-archive", action="store_true", help="non cancella lo zip")
    ap.add_argument("--from", dest="src", metavar="PATH",
                    help="usa una copia locale del dataset (per esempio il "
                         "repository della tesi di Matey-Sanz) invece di "
                         "Zenodo")
    ap.add_argument("--copy", action="store_true",
                    help="con --from, copia invece di collegare")
    args = ap.parse_args()

    if args.src:
        use_local(args.src, link=not args.copy)
        return 0

    if args.check:
        return 0 if check(DATA / "DATA") else 1

    if (DATA / "DATA").is_dir() and not args.force:
        print("data/DATA c'è già; la verifico")
        if check(DATA / "DATA"):
            return 0
        print("la scarico di nuovo")

    print(f"record {RECORD_ID} su Zenodo")
    meta = fetch_metadata()
    print(f"  file: {meta['name']}  versione: {meta['version']}")
    download(meta["url"], ARCHIVE, meta["size"])
    if not verify(ARCHIVE, meta["checksum"]):
        ARCHIVE.unlink(missing_ok=True)
        print("checksum diverso: archivio scartato, non ho estratto nulla")
        return 1
    extract(ARCHIVE, DATA)
    data_dir = normalise(DATA)
    if not args.keep_archive:
        ARCHIVE.unlink(missing_ok=True)
    print(f"dati in {data_dir}")
    return 0 if check(data_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
