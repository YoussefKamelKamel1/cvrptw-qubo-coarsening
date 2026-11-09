"""Can a better SA temperature schedule rescue the static penalties?

Static's failure is attributed to coefficient conditioning. An alternative
explanation is that the sampler's default temperature range, derived from the
largest and smallest effective fields, is badly placed for a QUBO with a 1e10
dynamic range. Here the default range is shifted in memory, per sample call:

  hot_x  : start colder, beta_hot multiplied by x (cold end unchanged)
  cold_x : end colder, beta_cold multiplied by x
  sweeps : ten times more sweeps at the default range

Scale (1, 1) with an explicit default range must reproduce the standard pipeline.
N = 10, 56 instances x 3 seeds x {none, heuristic}, 200 reads.

  python scripts/run_sa_schedule.py   -> results/sa_schedule.csv
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SETTINGS = {                       # name: (hot multiplier, cold multiplier, sweeps)
    "default": (1.0, 1.0, 200),
    "hot_x10": (10.0, 1.0, 200),
    "hot_x100": (100.0, 1.0, 200),
    "hot_x1e3": (1e3, 1.0, 200),
    "hot_x1e4": (1e4, 1.0, 200),
    "cold_x10": (1.0, 10.0, 200),
    "sweeps_x10": (1.0, 1.0, 2000),
}
MODES = ("static", "adaptive")


def _patch(hot, cold):
    import src.solvers as sv
    orig = sv.solve_qubo

    def solve_qubo(qubo, cfg, limit=1):
        if cfg.backend != "sa":
            return orig(qubo, cfg, limit)
        import dimod
        from dwave.samplers import SimulatedAnnealingSampler
        from dwave.samplers.sa.sampler import default_beta_range
        bqm = dimod.BQM.from_qubo(qubo.dict)
        b0, b1 = default_beta_range(bqm)
        kw = dict(num_reads=cfg.num_reads, num_sweeps=cfg.num_sweeps,
                  beta_range=[b0 * hot, max(b1 * cold, b0 * hot)])
        if cfg.seed is not None:
            kw["seed"] = cfg.seed
        resp = SimulatedAnnealingSampler().sample(bqm, **kw)
        return [s for s in resp.lowest()][:limit]

    sv.solve_qubo = solve_qubo


def _job(args):
    spec, setting = args
    from src.runner_parallel import run_one_job
    hot, cold, sweeps = SETTINGS[setting]
    _patch(hot, cold)
    spec = dict(spec, num_sweeps=sweeps)
    row = run_one_job(spec)
    row["setting"] = setting
    return row


def main():
    import pandas as pd
    from src.datasets import list_solomon
    from src.runner_parallel import build_jobs, run_one_job

    # fidelity: explicit default range == the unpatched sampler
    for spec in build_jobs(["R101", "C101"], [10], [0], ["none"], penalty_mode="static"):
        a = run_one_job(spec)
        b = _job((spec, "default"))
        assert all(a[k] == b[k] for k in ("pre_total_viol", "total_distance", "is_feasible")), \
            "explicit default range differs from the sampler default"
    print("explicit default range reproduces the standard pipeline", flush=True)

    base = []
    for mode in MODES:
        base += build_jobs(list_solomon(), [10], [0, 1, 2], ["none", "heuristic"],
                           penalty_mode=mode, num_reads=200, num_sweeps=200)
    jobs = [(s, st) for st in SETTINGS for s in base]
    nw = min(10, os.cpu_count())
    print(f"{len(jobs)} jobs on {nw} workers", flush=True)
    t, rows = time.perf_counter(), []
    with ProcessPoolExecutor(nw) as ex:
        for i, fu in enumerate(as_completed([ex.submit(_job, j) for j in jobs]), 1):
            rows.append(fu.result())
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}  {time.perf_counter()-t:.0f}s", flush=True)
    out = ROOT / "results" / "sa_schedule.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows) in {time.perf_counter()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
