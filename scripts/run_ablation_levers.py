"""Ablation: isolate the two Phase-2 levers.
static | scale_only (weights only) | smartcap_only (drop non-binding cap) | adaptive.
56 instances x 5 seeds x {none, heuristic}, N=10. Shows which lever does the work.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import list_solomon          # noqa: E402
from src.runner_parallel import build_jobs, run_grid  # noqa: E402


def main():
    import pandas as pd
    instances = list_solomon()
    seeds = [0, 1, 2, 3, 4]
    conditions = ["none", "heuristic"]
    modes = ["static", "scale_only", "smartcap_only", "adaptive"]
    nw = min(8, os.cpu_count())

    jobs = []
    for m in modes:
        jobs += build_jobs(instances, [10], seeds, conditions, penalty_mode=m,
                           num_reads=200, num_sweeps=200)
    print(f"Ablation: {len(jobs)} jobs on {nw} workers", flush=True)
    t = time.perf_counter()
    rows = run_grid(jobs, n_workers=nw, progress_every=400)
    print(f"done in {time.perf_counter()-t:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "phase2_levers.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)\n", flush=True)

    agg = (df.groupby("penalty_mode")
           .agg(pre_viol=("pre_total_viol", "mean"),
                pre_feas=("pre_feasible", lambda s: 100 * s.mean()),
                post_feas=("is_feasible", lambda s: 100 * s.mean()),
                vars=("num_qubo_vars", "mean")).round(2))
    print(agg.reindex(modes).to_string())
    print("\nInterpretation: scale_only ~ static (scaling is a no-op); "
          "smartcap_only ~ adaptive (dropping non-binding capacity is the lever).")


if __name__ == "__main__":
    main()
