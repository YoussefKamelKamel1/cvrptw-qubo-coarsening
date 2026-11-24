"""Pipeline without the self-loop term."""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCALES = (1.0, 0.0)
CONDITIONS = ("gnn", "none")
NS = (10, 15, 20)
SEEDS = range(5)


def _job(args):
    name, N, seed, cond, scale, ref = args
    from harness import run_job, variant
    with variant(self_scale=scale):
        row = run_job(name, N, seed, condition=cond, ortools_cost=ref)
    row["self_scale"] = scale
    return row


def main():
    import pandas as pd
    from harness import OUT, ortools_cache, run_job
    from src.datasets import train_test_split_instances
    _, test = train_test_split_instances()
    cache = ortools_cache()

    # fidelity: scale 1 must equal the unpatched pipeline
    from harness import variant
    for n, N, s, c in [("C104", 10, 0, "gnn"), ("RC108", 20, 2, "none")]:
        a = run_job(n, N, s, c, ortools_cost=cache.get((n, N)))
        with variant(self_scale=1.0):
            b = run_job(n, N, s, c, ortools_cost=cache.get((n, N)))
        assert all(a[k] == b[k] for k in ("total_distance", "pre_total_viol",
                                          "is_feasible", "pre_vehicles")), "scale 1 differs"
    print("scale 1.0 reproduces the published pipeline", flush=True)

    jobs = [(n, N, s, c, sc, cache.get((n, N)))
            for sc in SCALES for c in CONDITIONS for N in NS for n in test for s in SEEDS]
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(min(8, os.cpu_count())) as ex:
        for fu in as_completed([ex.submit(_job, j) for j in jobs]):
            rows.append(fu.result())
    out = OUT / "exp8_selfloop.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows) in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
