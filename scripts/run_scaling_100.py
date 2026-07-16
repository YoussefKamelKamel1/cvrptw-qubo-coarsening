"""Solve the coarsened pipeline at N=80 and N=100 (extends the solve range).

'none' is not solved (15k vars at N=100 = hopeless; variable count already made
the point). Only coarsened conditions. Instances with fewer than N customers are
excluded per N (a few Solomon instances have 99).
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import train_test_split_instances, load_solomon    # noqa: E402
from src.runner_parallel import build_jobs, run_grid                 # noqa: E402

MODEL = str(ROOT / "results" / "gnn_model_rl.pt")


def main():
    import pandas as pd
    from scipy.stats import wilcoxon

    _, test = train_test_split_instances()
    seeds = [0, 1, 2]
    conditions = ["heuristic", "heuristic_fixed", "gnn"]
    nw = min(8, os.cpu_count())

    jobs = []
    for N in [80, 100]:
        insts = [n for n in test if load_solomon(n).n_customers >= N]
        jobs += build_jobs(insts, [N], seeds, conditions, penalty_mode="adaptive",
                           num_reads=200, num_sweeps=200, model_path=MODEL)
    print(f"Solving coarsened at N=80/100: {len(jobs)} jobs on {nw} workers", flush=True)
    t = time.perf_counter()
    rows = run_grid(jobs, n_workers=nw, progress_every=50)
    print(f"done in {time.perf_counter()-t:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "scaling_100_solve.csv", index=False)

    print("\n=== coarsened feasibility %, pre-viol, vars at N=80/100 ===")
    for N in [80, 100]:
        for c in conditions:
            s = df[(df.N == N) & (df.condition == c)]
            if len(s):
                print(f"  N={N} {c:16s} feas={100*s.is_feasible.mean():5.1f}%  "
                      f"pre_viol={s.pre_total_viol.mean():7.1f}  vars={s.num_qubo_vars.mean():5.0f}"
                      f"  (n_inst={s.instance.nunique()})")

    print("\n=== GNN vs tuned heuristic feasibility (paired, N=80/100 pooled) ===")
    piv = (df.groupby(["instance", "N", "condition"])["is_feasible"].mean()
           .unstack("condition").dropna(subset=["gnn", "heuristic"]))
    x, y = piv["gnn"].to_numpy(), piv["heuristic"].to_numpy()
    if len(x) and not (x == y).all():
        _, p = wilcoxon(x, y, alternative="greater")
        print(f"  gnn={x.mean()*100:.1f}% heur={y.mean()*100:.1f}%  "
              f"GNN>=heur on {int((x>=y).sum())}/{len(x)} (strictly {int((x>y).sum())}, "
              f"worse {int((x<y).sum())})  p={p:.3f}")


if __name__ == "__main__":
    main()
