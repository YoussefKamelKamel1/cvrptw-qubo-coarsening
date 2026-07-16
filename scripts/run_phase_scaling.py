"""Step 4: scaling to N=40. Shows uncoarsened blows up while coarsened stays usable.

- Coarsened (heuristic-tuned, gnn zero-shot) get the full held-out sweep at N=40.
- Uncoarsened `none` runs on a small subset with a reduced budget (it is ~2400
  QUBO vars with an O(n^3) build term) purely to document intractability.
- Combined with the N=10/15/20 numbers from phase4_full.csv for the scaling curve.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import train_test_split_instances     # noqa: E402
from src.refcache import build_cache                     # noqa: E402
from src.runner_parallel import build_jobs, run_grid     # noqa: E402


def main():
    import numpy as np
    import pandas as pd

    _, test = train_test_split_instances()
    N = 40
    model = str(ROOT / "results" / "gnn_model_rl.pt")
    n_workers = min(6, os.cpu_count())     # leave cores for any concurrent job

    none_subset = test[:4]
    print(f"OR-Tools cache at N={N} for {len(test)} instances...", flush=True)
    cache = build_cache([(n, N) for n in test], n_workers=n_workers, time_limit=5.0)

    coarse = build_jobs(test, [N], seeds=[0, 1, 2], conditions=["heuristic", "gnn"],
                        model_path=model, ortools_cache=cache,
                        num_reads=200, num_sweeps=200)
    none = build_jobs(none_subset, [N], seeds=[0], conditions=["none"],
                      ortools_cache=cache, num_reads=100, num_sweeps=100)
    jobs = coarse + none
    print(f"Scaling N={N}: {len(coarse)} coarsened + {len(none)} none jobs on "
          f"{n_workers} workers", flush=True)

    t = time.perf_counter()
    rows = run_grid(jobs, n_workers=n_workers, progress_every=20)
    print(f"done in {time.perf_counter()-t:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "phase_scaling.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)\n", flush=True)

    order = ["none", "heuristic", "gnn"]
    print(f"=== N={N} summary ===")
    agg = (df.groupby("condition")
           .agg(feasible=("is_feasible", lambda s: round(100 * s.mean(), 1)),
                pre_viol=("pre_total_viol", "mean"),
                qubo_vars=("num_qubo_vars", "mean"),
                time_s=("computation_time", "mean")).round(2))
    print(agg.reindex([c for c in order if c in agg.index]).to_string())

    fe = df[df.is_feasible]
    print("\nOptimality gap (feasible, coarsened):")
    for c in ["heuristic", "gnn"]:
        v = fe[fe.condition == c]["opt_gap"].dropna().to_numpy() * 100
        if len(v):
            print(f"   {c:10s}: {v.mean():.1f}%  (n={len(v)})")

    # scaling curve: merge with phase4_full N=10/15/20
    p4 = ROOT / "results" / "phase4_full.csv"
    if p4.exists():
        prev = pd.read_csv(p4)
        both = pd.concat([prev, df], ignore_index=True)
        print("\n=== Scaling curve: mean QUBO vars by N x condition ===")
        v = both.groupby(["N", "condition"]).num_qubo_vars.mean().round(0).unstack("condition")
        print(v[[c for c in order if c in v.columns]].to_string())
        print("\nOptimality gap (%) by N x condition (feasible):")
        g = (both[both.is_feasible].groupby(["N", "condition"]).opt_gap.mean()
             .mul(100).round(1).unstack("condition"))
        print(g[[c for c in order if c in g.columns]].to_string())


if __name__ == "__main__":
    main()
