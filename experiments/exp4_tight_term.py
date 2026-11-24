"""Pipeline without the graded time-window term."""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# (label, tight_scale, fleet_divisor)
VARIANTS = [("submitted", 1.0, None), ("tight=0.1", 0.1, None),
            ("tight=0", 0.0, None), ("tight=0+fleet/4", 0.0, 4)]
CONDITIONS = ("gnn", "none")
NS = (10, 15, 20, 40)
SEEDS = range(5)


def _job(args):
    name, N, seed, cond, label, ts, div, ref = args
    from harness import run_job, variant
    with variant(tight_scale=ts, fleet_divisor=div):
        row = run_job(name, N, seed, condition=cond, ortools_cost=ref)
    row.update(variant=label, tight_scale=ts, fleet_divisor=div or 2)
    return row


def main():
    import pandas as pd
    from harness import OUT, ortools_cache
    from src.datasets import train_test_split_instances
    _, test = train_test_split_instances()
    cache = ortools_cache()
    jobs = [(n, N, s, c, lab, ts, div, cache.get((n, N)))
            for lab, ts, div in VARIANTS for c in CONDITIONS
            for N in NS for n in test for s in SEEDS]
    workers = min(8, os.cpu_count())
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            if i % 250 == 0:
                print(f"  {i}/{len(jobs)}  {time.perf_counter()-t0:.0f}s", flush=True)
    out = OUT / "exp4_tight_term.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows) in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
