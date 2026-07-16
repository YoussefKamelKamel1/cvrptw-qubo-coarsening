"""Step 2b: full-width Phase-4 evaluation (10 seeds) with optimality-gap CIs.

13 held-out test instances x 10 seeds x {none, heuristic, gnn} x N in {10,15,20},
adaptive penalties, reward-trained GNN. Reports the optimality gap vs OR-Tools
with bootstrap 95% CIs, a paired Wilcoxon (GNN vs heuristic), and feasibility.
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


def _boot_ci(vals, n=2000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    _, test = train_test_split_instances()
    Ns = [10, 15, 20]
    seeds = list(range(10))
    conditions = ["none", "heuristic", "gnn"]
    model = str(ROOT / "results" / "gnn_model_rl.pt")
    n_workers = min(8, os.cpu_count())

    print(f"Building OR-Tools cache for {len(test)}x{len(Ns)} pairs...", flush=True)
    cache = build_cache([(n, N) for N in Ns for n in test], n_workers=n_workers)

    jobs = build_jobs(test, Ns, seeds, conditions, penalty_mode="adaptive",
                      num_reads=200, num_sweeps=200, model_path=model,
                      ortools_cache=cache)
    print(f"Phase-4 full: {len(jobs)} jobs on {n_workers} workers", flush=True)

    t = time.perf_counter()
    rows = run_grid(jobs, n_workers=n_workers, progress_every=200)
    print(f"done in {time.perf_counter()-t:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "phase4_full.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)\n", flush=True)

    order = ["none", "heuristic", "gnn"]
    fe = df[df.is_feasible]
    print("=== Optimality gap vs OR-Tools (%, mean [95% CI], feasible runs) ===")
    for N in Ns:
        print(f"-- N={N} --")
        for c in order:
            v = fe[(fe.N == N) & (fe.condition == c)]["opt_gap"].dropna().to_numpy() * 100
            lo, hi = _boot_ci(v)
            print(f"   {c:10s}: {v.mean():6.1f}%  [{lo:5.1f}, {hi:5.1f}]  (n={len(v)})")

    print("\n=== Paired Wilcoxon: GNN vs heuristic optimality gap ===")
    piv = fe.pivot_table(index=["instance", "N", "seed"], columns="condition",
                         values="opt_gap").dropna(subset=["gnn", "heuristic"])
    g, h = piv["gnn"].to_numpy(), piv["heuristic"].to_numpy()
    print(f"  paired n={len(piv)}  mean gnn={g.mean()*100:.1f}%  heuristic={h.mean()*100:.1f}%")
    if not (g == h).all():
        stat, pval = wilcoxon(h, g, alternative="greater")
        print(f"  p={pval:.2e}  (heuristic gap > gnn gap)")

    print("\nN=10 feasibility % by family x condition:")
    n10 = df[df.N == 10]
    print(n10.groupby(["family_group", "condition"])["is_feasible"].mean().mul(100)
          .round(1).unstack("condition")[order].to_string())


if __name__ == "__main__":
    main()
