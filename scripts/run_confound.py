"""decouple dynamic-range conditioning from variable count.

The smart-capacity lever drops non-binding capacity constraints, which BOTH
collapses the QUBO's dynamic range AND removes the slack variables (131 -> 101),
so it cannot alone attribute the gain to conditioning. This adds a control,
``capcrush``: keep every capacity variable (variable count == static) but crush
the capacity penalty so its (2^m)^2 couplings no longer dominate the dynamic
range. If capcrush recovers the smart-capacity/adaptive gain at the static
variable count, dynamic-range conditioning is the mechanism, not the smaller
search space.

Modes: static, scale_only, smartcap_only, adaptive, capcrush. N=10, 56 instances
x 5 seeds x {none, heuristic}. Writes results/phase2_confound.csv.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import list_solomon                    # noqa: E402
from src.runner_parallel import build_jobs, run_grid     # noqa: E402


def main():
    import pandas as pd
    from scipy.stats import wilcoxon

    instances = list_solomon()
    seeds = [0, 1, 2, 3, 4]
    conditions = ["none", "heuristic"]
    modes = ["static", "scale_only", "smartcap_only", "adaptive", "capcrush"]
    nw = min(8, os.cpu_count())

    jobs = []
    for mode in modes:
        jobs += build_jobs(instances, [10], seeds, conditions, penalty_mode=mode,
                           num_reads=200, num_sweeps=200)
    print(f"Confound run: {len(jobs)} jobs on {nw} workers", flush=True)

    t = time.perf_counter()
    rows = run_grid(jobs, n_workers=nw, progress_every=400)
    print(f"done in {time.perf_counter()-t:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "phase2_confound.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)\n", flush=True)

    agg = (df.groupby("penalty_mode")
           .agg(pre_viol=("pre_total_viol", "mean"),
                pre_feas=("pre_feasible", lambda s: 100 * s.mean()),
                vars=("num_qubo_vars", "mean")).round(2))
    agg = agg.reindex(modes)
    print("mode            pre_viol  pre_feas%   vars")
    for m in modes:
        r = agg.loc[m]
        print(f"  {m:14s} {r.pre_viol:7.2f}   {r.pre_feas:6.1f}   {r.vars:5.1f}")

    # instance-aggregated paired Wilcoxon vs static
    print("\nInstance-level paired Wilcoxon on pre-repair violations (vs static):")
    piv = (df.groupby(["instance", "penalty_mode"])["pre_total_viol"]
           .median().unstack("penalty_mode"))
    base = piv["static"].to_numpy()
    for m in ["scale_only", "smartcap_only", "adaptive", "capcrush"]:
        v = piv[m].to_numpy()
        pair = piv[["static", m]].dropna()
        s, x = pair["static"].to_numpy(), pair[m].to_numpy()
        if (s == x).all():
            print(f"  static vs {m:14s}: identical")
        else:
            _, p = wilcoxon(s, x, alternative="greater")
            print(f"  static vs {m:14s}: static={s.mean():5.2f} {m}={x.mean():5.2f}  p={p:.2e}")


if __name__ == "__main__":
    main()
