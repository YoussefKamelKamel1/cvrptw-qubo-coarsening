"""Learned stopping (Phase-5): does a learned stop threshold beat fixed depth P?

On TRAIN instances at N=40, sweep the fixed-P frontier vs the learned-threshold
frontier (P kept low so tau controls depth), each as (feasibility, opt-gap,
coarsen ratio). Pick the best of each by a balanced score, compare, then confirm
the chosen tau on the HELD-OUT test at N=10/15/20/40 vs fixed P=0.5.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import train_test_split_instances     # noqa: E402
from src.refcache import build_cache, load_cache         # noqa: E402
from src.runner_parallel import build_jobs, run_grid     # noqa: E402

MODEL = str(ROOT / "results" / "gnn_model_rl.pt")


def frontier(rows_df, label):
    fe = rows_df[rows_df.is_feasible]
    feas = 100 * rows_df.is_feasible.mean()
    gap = fe.opt_gap.mean() * 100 if len(fe) else float("nan")
    ratio = rows_df.coarsen_ratio.mean()
    vars_ = rows_df.num_qubo_vars.mean()
    score = feas - 0.3 * (gap if gap == gap else 300)
    print(f"  {label:16s} feas={feas:5.1f}%  gap={gap:6.1f}%  ratio={ratio:.2f}  "
          f"vars={vars_:4.0f}  score={score:6.1f}")
    return dict(label=label, feas=feas, gap=gap, ratio=ratio, vars=vars_, score=score)


def main():
    import pandas as pd
    train, test = train_test_split_instances()
    nw = min(8, os.cpu_count())
    seeds = [0, 1, 2]
    Ps = [0.4, 0.5, 0.6, 0.7]
    taus = [0.5, 0.7, 0.8, 0.9, 0.95]

    print("OR-Tools cache (train + test @ N=40)...", flush=True)
    build_cache([(n, 40) for n in train + test], n_workers=nw, time_limit=5.0)
    cache = load_cache()

    # ---- TRAIN frontiers at N=40 ----
    print("\n=== Fixed-P frontier (train, N=40) ===")
    fixedP = []
    for P in Ps:
        jobs = build_jobs(train, [40], seeds, ["gnn"], model_path=MODEL,
                          ortools_cache=cache, gnn_P=P, gnn_rc=0.5)
        df = pd.DataFrame(run_grid(jobs, n_workers=nw, progress_every=0))
        fixedP.append(frontier(df, f"P={P}"))

    print("\n=== Learned-threshold frontier (train, N=40, P=0.1) ===")
    learned = []
    for tau in taus:
        jobs = build_jobs(train, [40], seeds, ["gnn"], model_path=MODEL,
                          ortools_cache=cache, gnn_P=0.1, gnn_rc=0.5,
                          gnn_stop_threshold=tau)
        df = pd.DataFrame(run_grid(jobs, n_workers=nw, progress_every=0))
        learned.append(frontier(df, f"tau={tau}"))

    bestP = max(fixedP, key=lambda d: d["score"])
    bestT = max(learned, key=lambda d: d["score"])
    print(f"\nBest fixed-P: {bestP['label']} (score {bestP['score']:.1f})")
    print(f"Best learned: {bestT['label']} (score {bestT['score']:.1f})")
    win = "learned stopping" if bestT["score"] > bestP["score"] else "fixed P"
    print(f"=> better balanced score on train: {win}")

    # ---- confirm chosen tau on TEST across N ----
    tau = float(bestT["label"].split("=")[1])
    print(f"\n=== TEST confirmation: learned tau={tau} vs fixed P=0.5 ===")
    cache_all = load_cache()
    for N in [10, 15, 20, 40]:
        build_cache([(n, N) for n in test], n_workers=nw, time_limit=5.0)
    cache_all = load_cache()
    rows = []
    for label, kw in [("gnn_P0.5", dict(gnn_P=0.5)),
                      (f"gnn_tau{tau}", dict(gnn_P=0.1, gnn_stop_threshold=tau))]:
        jobs = build_jobs(test, [10, 15, 20, 40], seeds, ["gnn"], model_path=MODEL,
                          ortools_cache=cache_all, gnn_rc=0.5, **kw)
        df = pd.DataFrame(run_grid(jobs, n_workers=nw, progress_every=0))
        df["variant"] = label
        rows.append(df)
    allt = pd.concat(rows, ignore_index=True)
    allt.to_csv(ROOT / "results" / "learned_stop.csv", index=False)
    print("feasibility % and gap % by N x variant:")
    for N in [10, 15, 20, 40]:
        sub = allt[allt.N == N]
        for v in ["gnn_P0.5", f"gnn_tau{tau}"]:
            s = sub[sub.variant == v]
            fe = s[s.is_feasible]
            print(f"  N={N} {v:12s}: feas={100*s.is_feasible.mean():5.1f}%  "
                  f"gap={fe.opt_gap.mean()*100 if len(fe) else float('nan'):6.1f}%  "
                  f"ratio={s.coarsen_ratio.mean():.2f}")


if __name__ == "__main__":
    main()
