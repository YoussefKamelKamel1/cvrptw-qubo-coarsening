"""Adaptive calibration against static penalties at N = 20 and 40 (56 instances,
five seeds).
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.datasets import list_solomon                    # noqa: E402
from src.runner_parallel import build_jobs, run_grid     # noqa: E402

NS = (20, 40)
MODES = ("static", "adaptive", "capcrush")


def main():
    import pandas as pd
    jobs = []
    for mode in MODES:
        jobs += build_jobs(list_solomon(), list(NS), [0, 1, 2, 3, 4], ["none", "heuristic"],
                           penalty_mode=mode, num_reads=200, num_sweeps=200)
    nw = min(10, os.cpu_count())
    print(f"{len(jobs)} jobs on {nw} workers", flush=True)
    t = time.perf_counter()
    rows = run_grid(jobs, n_workers=nw, progress_every=300)
    out = HERE / "results" / "exp9_table1_largerN.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows) in {time.perf_counter()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
