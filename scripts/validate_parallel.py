"""Validate the parallel runner: parallel results == serial, and measure speedup."""
import os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.refcache import build_cache
from src.runner_parallel import build_jobs, run_grid


def main():
    from src.datasets import list_solomon
    instances = list_solomon()[:24]
    Ns = [10]
    seeds = [0, 1, 2]
    conditions = ["none", "heuristic", "gnn"]
    model = str(ROOT / "results" / "gnn_model_rl.pt")
    ncpu = os.cpu_count()
    n_workers = min(8, ncpu)
    print(f"CPUs: {ncpu}  workers: {n_workers}")

    # OR-Tools cache
    pairs = [(n, N) for N in Ns for n in instances]
    t = time.perf_counter()
    cache = build_cache(pairs, n_workers=n_workers)
    print(f"OR-Tools cache built ({len(cache)} entries) in {time.perf_counter()-t:.1f}s")

    jobs = build_jobs(instances, Ns, seeds, conditions, model_path=model,
                      ortools_cache=cache, num_reads=200, num_sweeps=200)
    print(f"{len(jobs)} jobs")

    t = time.perf_counter()
    serial = run_grid(jobs, serial=True, progress_every=0)
    t_serial = time.perf_counter() - t
    print(f"serial:   {t_serial:.1f}s")

    t = time.perf_counter()
    par = run_grid(jobs, n_workers=n_workers, progress_every=0)
    t_par = time.perf_counter() - t
    print(f"parallel: {t_par:.1f}s  ->  speedup {t_serial/max(t_par,1e-9):.1f}x")

    # compare
    mism = 0
    for a, b in zip(serial, par):
        if (a["is_feasible"] != b["is_feasible"]
                or round(a["total_distance"], 4) != round(b["total_distance"], 4)
                or a["num_qubo_vars"] != b["num_qubo_vars"]):
            mism += 1
            print("MISMATCH", a["instance"], a["condition"], a["seed"],
                  a["total_distance"], b["total_distance"])
    print(f"parallel == serial: {'YES' if mism == 0 else f'NO ({mism} mismatches)'}")
    # sanity: show a couple of rows with gaps
    for r in par[:3]:
        print(f"  {r['instance']} {r['condition']}: feas={r['is_feasible']} "
              f"dist={r['total_distance']:.1f} gap={r['opt_gap']:.3f} vars={r['num_qubo_vars']}")


if __name__ == "__main__":
    main()
