"""Larger-N run: does tuning-free coarsening scale better than fixed tuning?

Focus on the metrics where coarsening's value is real (QUBO size, pre-repair
quality, coarsened feasibility, compression) rather than post-repair cost. Key
question: does the GNN - heuristic separation GROW with N (the heuristic was tuned
at N=10 and may not transfer)?
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import train_test_split_instances       # noqa: E402
from src.refcache import build_cache                       # noqa: E402
from src.runner_parallel import build_jobs, run_grid       # noqa: E402

MODEL = str(ROOT / "results" / "gnn_model_rl.pt")


def main():
    import pandas as pd

    _, test = train_test_split_instances()
    Ns = [20, 30, 40]
    seeds = [0, 1, 2]
    conditions = ["none", "heuristic", "gnn"]
    nw = min(8, os.cpu_count())

    print("OR-Tools cache...", flush=True)
    cache = build_cache([(n, N) for N in Ns for n in test], n_workers=nw, time_limit=5.0)

    jobs = build_jobs(test, Ns, seeds, conditions, penalty_mode="adaptive",
                      num_reads=200, num_sweeps=200, model_path=MODEL,
                      ortools_cache=cache)
    print(f"Scaling-P4: {len(jobs)} jobs on {nw} workers", flush=True)
    t = time.perf_counter()
    rows = run_grid(jobs, n_workers=nw, progress_every=100)
    print(f"done in {time.perf_counter()-t:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "scaling_phase4.csv", index=False)

    print("\n=== coarsened feasibility %, pre-repair violations, QUBO vars by N ===")
    print(f"  {'N':>3} {'cond':>10} {'feas%':>6} {'pre_viol':>9} {'vars':>6} {'ratio':>6}")
    for N in Ns:
        for c in conditions:
            s = df[(df.N == N) & (df.condition == c)]
            print(f"  {N:>3} {c:>10} {100*s.is_feasible.mean():6.1f} "
                  f"{s.pre_total_viol.mean():9.1f} {s.num_qubo_vars.mean():6.0f} "
                  f"{s.coarsen_ratio.mean():6.2f}")

    print("\n=== GNN - heuristic separation vs N (does it grow?) ===")
    for N in Ns:
        h = df[(df.N == N) & (df.condition == "heuristic")]
        g = df[(df.N == N) & (df.condition == "gnn")]
        d_feas = 100 * (g.is_feasible.mean() - h.is_feasible.mean())
        d_viol = h.pre_total_viol.mean() - g.pre_total_viol.mean()
        d_vars = h.num_qubo_vars.mean() - g.num_qubo_vars.mean()
        print(f"  N={N}: feas +{d_feas:+.1f}pp   pre_viol reduced {d_viol:+.1f}   "
              f"vars smaller {d_vars:+.0f}")


if __name__ == "__main__":
    main()
