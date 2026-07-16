"""Step 2a: full-width Phase-2 ablation (static vs adaptive), parallel.

56 instances x 5 seeds x {none, heuristic} x {static, adaptive}, N=10. Includes
the RC1 binding-capacity fix. Reports pre-repair violation reduction with a paired
Wilcoxon (larger n than the original 432) and feasibility by family.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import list_solomon           # noqa: E402
from src.runner_parallel import build_jobs, run_grid  # noqa: E402


def main():
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    instances = list_solomon()          # 56
    Ns = [10]
    seeds = [0, 1, 2, 3, 4]
    conditions = ["none", "heuristic"]
    n_workers = min(8, os.cpu_count())

    jobs = []
    for mode in ["static", "adaptive"]:
        jobs += build_jobs(instances, Ns, seeds, conditions, penalty_mode=mode,
                           num_reads=200, num_sweeps=200)
    print(f"Phase-2 full: {len(jobs)} jobs on {n_workers} workers", flush=True)

    t = time.perf_counter()
    rows = run_grid(jobs, n_workers=n_workers, progress_every=200)
    print(f"done in {time.perf_counter()-t:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "phase2_full.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)\n", flush=True)

    # aggregate
    agg = (df.groupby("penalty_mode")
           .agg(pre_viol=("pre_total_viol", "mean"),
                pre_feas=("pre_feasible", lambda s: 100 * s.mean()),
                post_feas=("is_feasible", lambda s: 100 * s.mean()),
                vars=("num_qubo_vars", "mean")).round(2))
    print(agg.to_string())

    print("\nPost-repair feasibility % by family group x mode:")
    piv = (df.groupby(["family_group", "penalty_mode"])["is_feasible"]
           .mean().mul(100).round(1).unstack("penalty_mode"))
    print(piv.to_string())

    print("\nRC1 specifically (the fixed regression):")
    rc1 = df[df.family == "RC1"]
    print(rc1.groupby("penalty_mode")["is_feasible"].mean().mul(100).round(1).to_string())

    # paired Wilcoxon on pre-repair violations
    keys = ["instance", "N", "seed", "condition"]
    p = df.pivot_table(index=keys, columns="penalty_mode", values="pre_total_viol").dropna()
    s, a = p["static"].to_numpy(), p["adaptive"].to_numpy()
    print(f"\nPaired Wilcoxon (pre-repair violations), n={len(p)}")
    print(f"  mean static={s.mean():.2f}  adaptive={a.mean():.2f}")
    if not (s == a).all():
        stat, pval = wilcoxon(s, a, alternative="greater")
        print(f"  p={pval:.2e}  (static > adaptive)")


if __name__ == "__main__":
    main()
