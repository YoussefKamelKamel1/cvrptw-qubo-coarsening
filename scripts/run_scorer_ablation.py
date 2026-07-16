"""Isolate the learned merge scorer's value.

heuristic_fixed uses the SAME coarsening settings as the GNN (P=0.5, rc=0.5) but
with the fixed spatio-temporal D_ij score; the GNN swaps only the scorer for the
learned one. So gnn vs heuristic_fixed isolates the learned policy with everything
else held constant. heuristic (per-family tuned) is the oracle-knowledge baseline.

Question: does the learned scorer beat the fixed scorer at equal settings, and
does it match the per-family-tuned heuristic without any tuning?
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
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    _, test = train_test_split_instances()
    Ns = [10, 20, 40]
    seeds = [0, 1, 2]
    conditions = ["heuristic", "heuristic_fixed", "gnn"]
    nw = min(8, os.cpu_count())

    cache = build_cache([(n, N) for N in Ns for n in test], n_workers=nw, time_limit=5.0)
    jobs = build_jobs(test, Ns, seeds, conditions, penalty_mode="adaptive",
                      num_reads=200, num_sweeps=200, model_path=MODEL, ortools_cache=cache)
    print(f"Scorer ablation: {len(jobs)} jobs", flush=True)
    t = time.perf_counter()
    rows = run_grid(jobs, n_workers=nw, progress_every=100)
    print(f"done in {time.perf_counter()-t:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "scorer_ablation.csv", index=False)

    print("\n=== by N x condition: pre-viol | coarsened feas% | pre-feas% | vars ===")
    for N in Ns:
        print(f"-- N={N} --")
        for c in conditions:
            s = df[(df.N == N) & (df.condition == c)]
            print(f"   {c:16s} pre_viol={s.pre_total_viol.mean():6.1f}  "
                  f"feas={100*s.is_feasible.mean():5.1f}%  "
                  f"pre_feas={100*s.pre_feasible.mean():5.1f}%  vars={s.num_qubo_vars.mean():4.0f}")

    # paired: gnn vs heuristic_fixed (same settings, only scorer differs)
    print("\n=== gnn vs heuristic_fixed (isolates learned scorer), pre-repair violations ===")
    for N in Ns + ["all"]:
        d = df if N == "all" else df[df.N == N]
        piv = d.pivot_table(index=["instance", "seed"], columns="condition",
                            values="pre_total_viol").dropna(subset=["gnn", "heuristic_fixed"])
        g, hf = piv["gnn"].to_numpy(), piv["heuristic_fixed"].to_numpy()
        agg = (d.groupby(["instance", "condition"])["pre_total_viol"].mean()
               .unstack("condition").dropna(subset=["gnn", "heuristic_fixed"]))
        gi, hfi = agg["gnn"].to_numpy(), agg["heuristic_fixed"].to_numpy()
        p = (wilcoxon(hfi, gi, alternative="greater")[1]
             if not (gi == hfi).all() else float("nan"))
        print(f"  N={str(N):>3}: fixed={hf.mean():6.1f} gnn={g.mean():6.1f}  "
              f"GNN lower on {int((gi<hfi).sum())}/{len(gi)} inst  p(inst)={p:.3f}")


if __name__ == "__main__":
    main()
