"""Fetch benchmark datasets + reference code for the project.

Strategy (robust to a flaky network):
  1. Clone the two reference repos into ``reference/`` (git resumes better than
     many small HTTP GETs, and we want the code to study anyway):
       - (CVRPTW predecessor: Solomon data + pipeline)
       - AsishMandoi/VRP-explorations  (FQS/APS/CSPS reference solvers)
  2. Copy any Solomon (*.txt in Solomon format) and Christofides (CMT*) files
     found in those clones into data/solomon and data/christofides.
  3. Fall back to direct raw-URL mirrors for Solomon C101/R101/RC101 (N=100)
     if the clones yield no usable instances.

Run:  python scripts/fetch_data.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference"
SOL = ROOT / "data" / "solomon"
CMT = ROOT / "data" / "christofides"

REPOS = {
    "graph-coarsening": "https://github.com/.git",
    "VRP-explorations": "https://github.com/AsishMandoi/VRP-explorations.git",
}

# Raw-URL fallback mirrors (Solomon 100-customer, classic .txt format).
SOLOMON_RAW_FALLBACK = {
    # CVRPLIB text mirror on GitHub (well-formed Solomon format)
    "C101": "https://raw.githubusercontent.com/PyVRP/Instances/main/VRPTW/C101.txt",
    "R101": "https://raw.githubusercontent.com/PyVRP/Instances/main/VRPTW/R101.txt",
    "RC101": "https://raw.githubusercontent.com/PyVRP/Instances/main/VRPTW/RC101.txt",
}


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def clone_repos() -> None:
    REF.mkdir(parents=True, exist_ok=True)
    for name, url in REPOS.items():
        dest = REF / name
        if dest.exists():
            print(f"[clone] {name}: already present, skipping")
            continue
        print(f"[clone] {name} <- {url}")
        rc = run(["git", "clone", "--depth", "1", url, str(dest)])
        if rc != 0:
            print(f"  !! clone failed for {name} (rc={rc})")


def _looks_like_solomon(text: str) -> bool:
    up = text.upper()
    return "VEHICLE" in up and "CUSTOMER" in up and "DUE" in up


def harvest_from_clones() -> tuple[int, int]:
    SOL.mkdir(parents=True, exist_ok=True)
    CMT.mkdir(parents=True, exist_ok=True)
    n_sol = n_cmt = 0
    if not REF.exists():
        return 0, 0
    for path in REF.rglob("*"):
        if not path.is_file():
            continue
        low = path.name.lower()
        # Solomon: names like c101.txt / r1_10_1.txt etc., Solomon-format body
        if path.suffix.lower() == ".txt":
            try:
                txt = path.read_text(errors="ignore")
            except Exception:
                continue
            if _looks_like_solomon(txt):
                out = SOL / path.name
                if not out.exists():
                    shutil.copy2(path, out)
                    n_sol += 1
        # Christofides: CMT-prefixed files
        if low.startswith("cmt"):
            out = CMT / path.name
            if not out.exists():
                shutil.copy2(path, out)
                n_cmt += 1
    return n_sol, n_cmt


def fetch_solomon_fallback() -> int:
    SOL.mkdir(parents=True, exist_ok=True)
    got = 0
    for name, url in SOLOMON_RAW_FALLBACK.items():
        out = SOL / f"{name}.txt"
        if out.exists():
            continue
        try:
            print(f"[http] {name} <- {url}")
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read().decode("utf-8", errors="ignore")
            if _looks_like_solomon(data):
                out.write_text(data)
                got += 1
            else:
                print(f"  !! {name}: fetched but not Solomon format, discarding")
        except Exception as e:
            print(f"  !! {name}: {e}")
    return got


def main() -> int:
    print("=== fetch_data ===")
    clone_repos()
    n_sol, n_cmt = harvest_from_clones()
    print(f"[harvest] copied {n_sol} Solomon + {n_cmt} Christofides files from clones")
    if n_sol == 0:
        print("[fallback] no Solomon from clones; trying raw URLs")
        got = fetch_solomon_fallback()
        print(f"[fallback] fetched {got} Solomon instances")
    sol_files = sorted(SOL.glob("*.txt"))
    cmt_files = sorted(CMT.glob("*"))
    print(f"\nSolomon cached: {len(sol_files)}")
    for p in sol_files[:12]:
        print(f"   {p.name}")
    print(f"Christofides cached: {len(cmt_files)}")
    for p in cmt_files[:12]:
        print(f"   {p.name}")
    return 0 if (sol_files or cmt_files) else 1


if __name__ == "__main__":
    sys.exit(main())
