"""OR-Tools reference (guided local search, 3 s, one run at a time) for every held-out
Solomon instance and size.
"""
import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from review_runner import load                        # noqa: E402
from src.baselines import solve_cvrptw_ortools         # noqa: E402
from src.datasets import train_test_split_instances    # noqa: E402

SIZES = [10, 20, 40, 50, 60, 80, 100]
OUT = HERE / "results" / "ortools_ref_test.csv"

rows = []
for name in train_test_split_instances()[1]:
    for N in SIZES:
        inst = load(name, "solomon", "fixed").truncate(N)
        t0 = time.perf_counter()
        s = solve_cvrptw_ortools(inst, time_limit_s=3.0)
        dt = time.perf_counter() - t0
        rows.append(dict(instance=name, N=N, feasible=bool(s.feasible),
                         cost=float(s.cost) if s.feasible else "",
                         vehicles=len([r for r in s.routes if len(r) > 0]) if s.feasible else "",
                         time_s=round(dt, 2)))
        print(rows[-1], flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", OUT)
