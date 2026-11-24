"""Fleet size of the position encoding: default n_v = max(2, n_c // 2) against
alternatives.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# (label, fleet_divisor, cont_factor); divisor 2 with factor 0.5 is the published setup ("submitted")
VARIANTS = [("submitted", 2, 0.5), ("fleet/3", 3, 0.5),
            ("fleet/4", 4, 0.5), ("fleet/4+cont1", 4, 1.0)]
CONDITIONS = ("gnn", "none")
NS = (10, 15, 20)
SEEDS = range(5)


def _job(args):
    name, N, seed, cond, label, div, cf, ref = args
    from harness import run_job, variant
    with variant(cont_factor=cf, fleet_divisor=div):
        row = run_job(name, N, seed, condition=cond, ortools_cost=ref)
    row.update(variant=label, fleet_divisor=div, cont_factor=cf)
    return row


def main():
    import pandas as pd
    from harness import OUT, ortools_cache
    from src.datasets import train_test_split_instances
    _, test = train_test_split_instances()
    cache = ortools_cache()
    jobs = [(n, N, s, c, lab, div, cf, cache.get((n, N)))
            for lab, div, cf in VARIANTS for c in CONDITIONS
            for N in NS for n in test for s in SEEDS]
    workers = min(8, os.cpu_count())
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}  {time.perf_counter()-t0:.0f}s", flush=True)
    out = OUT / "exp1b_fleet_size.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows) in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
