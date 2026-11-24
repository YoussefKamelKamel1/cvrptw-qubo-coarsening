"""Continuity term with factor 1/2 (as published) against factor 1: number of routes
and gap to OR-Tools.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FACTORS = (0.5, 1.0)
CONDITIONS = ("gnn", "none")
NS = (10, 15, 20)
SEEDS = range(5)


def _job(args):
    name, N, seed, cond, factor, ref = args
    from harness import run_job, variant
    with variant(cont_factor=factor):
        row = run_job(name, N, seed, condition=cond, ortools_cost=ref)
    row["cont_factor"] = factor
    return row


def _ortools_vehicles(args):
    name, N = args
    from harness import ROOT  # noqa: F401  (sets sys.path)
    from src.baselines import solve_cvrptw_ortools
    from src.datasets import load_solomon
    s = solve_cvrptw_ortools(load_solomon(name).truncate(N), time_limit_s=3.0)
    return dict(instance=name, N=N, ort_cost=s.cost if s.feasible else float("nan"),
                ort_vehicles=s.num_vehicles_used if s.feasible else -1)


def main():
    import pandas as pd
    from harness import OUT, ortools_cache
    from src.datasets import train_test_split_instances
    _, test = train_test_split_instances()
    cache = ortools_cache()
    workers = min(8, os.cpu_count())

    ov_path = OUT / "ortools_vehicles.csv"
    if not ov_path.exists():
        with ProcessPoolExecutor(workers) as ex:
            ov = list(ex.map(_ortools_vehicles, [(n, N) for N in NS for n in test]))
        pd.DataFrame(ov).to_csv(ov_path, index=False)
        print(f"OR-Tools vehicle counts -> {ov_path}", flush=True)

    jobs = [(n, N, s, c, f, cache.get((n, N)))
            for f in FACTORS for c in CONDITIONS for N in NS for n in test for s in SEEDS]
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}  {time.perf_counter()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = OUT / "exp1_vehicle_bias.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows) in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
