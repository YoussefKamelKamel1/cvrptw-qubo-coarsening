"""Data checks: malformed Solomon rows and instances that share customer maps."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.datasets import list_solomon, load_solomon, train_test_split_instances  # noqa: E402

# 1. rows that do not parse as seven numbers, in every file
print("== rows that do not parse ==")
bad = 0
for f in sorted((ROOT / "data" / "solomon").rglob("*.csv")):
    lines = f.read_text().splitlines()
    for k, line in enumerate(lines[1:], start=2):
        parts = line.split(",")
        try:
            assert len(parts) == 7
            [float(p) for p in parts]
        except Exception:
            bad += 1
            print(f"  {f.name} line {k}: {line!r}")
print(f"  total malformed rows: {bad}")

# 2. customer count as loaded
print("\n== loaded customer counts != 100 ==")
names = list_solomon()
for n in names:
    c = load_solomon(n).coords.shape[0] - 1
    if c != 100:
        print(f"  {n}: {c}")

# 3. which instances share coordinates+demands (full 100 customers)
print("\n== groups of instances with identical customer maps (coords+demand) ==")
key = {}
for n in names:
    i = load_solomon(n)
    key[n] = (np.round(i.coords, 3).tobytes(), np.round(i.demand, 3).tobytes())
groups = defaultdict(list)
for n, k in key.items():
    groups[k].append(n)
for g in sorted(groups.values(), key=lambda g: g[0]):
    fams = sorted({x.rstrip("0123456789") + x[len(x.rstrip("0123456789")):][:1] for x in g})
    print(f"  {len(g):2d} instances: {g[0]}..{g[-1]}  families: {sorted({''.join(c for c in x if not c.isdigit()) + x[len(''.join(c for c in x if not c.isdigit()))] for x in g})}")

# coordinates only
print("\n== coordinate-only sharing across families ==")
ck = defaultdict(set)
for n in names:
    i = load_solomon(n)
    fam = n[:-2]
    ck[np.round(i.coords, 3).tobytes()].add(fam)
for fams in ck.values():
    print("  shared map across:", sorted(fams))

_, test = train_test_split_instances()
print("\ntest:", test)
