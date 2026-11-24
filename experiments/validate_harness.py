"""Check that harness.py reproduces the stored GNN results when no change is applied."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd                                          # noqa: E402
from harness import ROOT, run_job, variant, ortools_cache    # noqa: E402

p4 = pd.read_csv(ROOT / "results" / "phase4_full.csv")
cache = ortools_cache()
cases = [("C104", 10, 0), ("R104", 15, 1), ("RC108", 20, 2), ("R208", 10, 3),
         ("C208", 20, 0), ("RC204", 15, 4)]

ok = True
for name, N, seed in cases:
    ref = p4[(p4.instance == f"{name}[:{N}]") & (p4.N == N) & (p4.seed == seed)
             & (p4.condition == "gnn")].iloc[0]
    a = run_job(name, N, seed, ortools_cost=cache.get((name, N)))
    with variant(cont_factor=0.5):
        b = run_job(name, N, seed, ortools_cost=cache.get((name, N)))
    same_ref = (a["is_feasible"] == ref.is_feasible
                and abs(a["total_distance"] - ref.total_distance) < 1e-6
                and a["num_vehicles"] == ref.num_vehicles)
    same_patch = all(a[k] == b[k] for k in ("is_feasible", "total_distance",
                                             "num_vehicles", "pre_vehicles", "pre_total_viol"))
    ok &= same_ref and same_patch
    print(f"  {name:6s} N={N:2d} s={seed}  matches phase4_full: {same_ref}   "
          f"patch(0.5)==original: {same_patch}   "
          f"dist {a['total_distance']:.2f} vs {ref.total_distance:.2f}")
print("\nHARNESS FAITHFUL" if ok else "\nMISMATCH - do not trust results")
