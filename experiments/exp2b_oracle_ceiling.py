"""Pipeline with a coarsening that merges only consecutive customers of the OR-Tools
routes, against GNN coarsening.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NS = (10, 15, 20, 40)
SEEDS = range(5)
ARMS = [("gnn", 2), ("oracle", 2), ("oracle", 4), ("gnn", 4)]


def _reference(args):
    name, N = args
    import harness  # noqa: F401
    from src.baselines import solve_cvrptw_ortools
    from src.datasets import load_solomon
    s = solve_cvrptw_ortools(load_solomon(name).truncate(N), time_limit_s=3.0)
    return (name, N, s.routes if s.feasible else None, s.cost if s.feasible else None)


def _job(args):
    name, N, seed, cond, div, routes, cost = args
    from harness import run_job, variant
    with variant(fleet_divisor=div):
        row = run_job(name, N, seed, condition=cond, ortools_cost=cost,
                      oracle_routes=routes)
    row.update(arm=f"{cond}/fleet{div}", fleet_divisor=div, ref_routes=len(routes))
    return row


def main():
    import pandas as pd
    from harness import OUT
    from src.datasets import train_test_split_instances
    _, test = train_test_split_instances()
    workers = min(8, os.cpu_count())
    with ProcessPoolExecutor(workers) as ex:
        refs = [r for r in ex.map(_reference, [(n, N) for N in NS for n in test]) if r[2]]
    print(f"{len(refs)} reference solutions", flush=True)
    jobs = [(n, N, s, c, div, routes, cost)
            for (n, N, routes, cost) in refs for c, div in ARMS for s in SEEDS]
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}  {time.perf_counter()-t0:.0f}s", flush=True)
    out = OUT / "exp2b_oracle_ceiling.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows) in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
