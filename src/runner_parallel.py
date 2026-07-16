"""Parallel experiment runner.

One *job* = (instance, N, seed, condition, penalty_mode, solver). ``run_one_job``
builds the instance, coarsens (none / heuristic-tuned / heuristic-fixed / gnn),
solves the QUBO on SA, and returns a metrics row (optionally with the OR-Tools
optimality gap). ``run_grid`` maps jobs over a process pool.

Design:
  * Jobs are plain picklable dicts (Windows 'spawn' safe).
  * The GNN model is loaded lazily *inside* each worker on CPU (cached per
    process) -- GNN inference on tens-of-node graphs is sub-millisecond, so CPU
    avoids CUDA-in-subprocess overhead. The GPU is used only for training.
  * Per-job determinism (seeded) means parallel results equal serial results.
"""
from __future__ import annotations

import os
# keep BLAS single-threaded so parallel SA processes don't oversubscribe cores
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# workers run GNN inference on CPU; hide CUDA so each process does not reserve a
# CUDA context (12 x CUDA imports exhaust the Windows paging file). The GPU is
# used only for training, which runs in its own (non-parallel) process.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import time
from typing import Optional

GNN_FIXED_P = 0.5
GNN_FIXED_RC = 0.5
HEUR_FIXED = {"alpha": 1.0, "beta": 1.0, "P": GNN_FIXED_P, "radiusCoeff": GNN_FIXED_RC}

_MODEL_CACHE: dict = {}


def _get_model(path):
    m = _MODEL_CACHE.get(path)
    if m is None:
        from .gnn_coarsening import load_model
        m = load_model(path, device="cpu")
        _MODEL_CACHE[path] = m
    return m


def run_one_job(spec: dict) -> dict:
    from .datasets import load_solomon, SOLOMON_FAMILY_HYPERPARAMS
    from .coarsening import graph_from_instance, SpatioTemporalGraphCoarsener
    from .experiment import solve_on_graph
    from .solvers import SolverConfig
    from .utils import set_global_seed
    # torch/GNN imported lazily below only for 'gnn' jobs, so non-GNN jobs never
    # pay the torch import cost.

    name, N, seed = spec["name"], spec["N"], spec["seed"]
    condition = spec["condition"]
    solver = spec.get("solver", "FQS")
    penalty_mode = spec.get("penalty_mode", "adaptive")
    set_global_seed(seed)

    inst = load_solomon(name).truncate(N)
    cfg = SolverConfig("sa", spec.get("num_reads", 200),
                       spec.get("num_sweeps", 200), seed=seed)

    graph, depot = graph_from_instance(inst)
    n0 = len(graph.nodes)
    coarsener = None
    if condition == "heuristic":
        coarsener = SpatioTemporalGraphCoarsener(
            graph=graph, depot_id=depot, **SOLOMON_FAMILY_HYPERPARAMS[inst.family])
        graph, _ = coarsener.coarsen()
    elif condition == "heuristic_fixed":
        coarsener = SpatioTemporalGraphCoarsener(graph=graph, depot_id=depot, **HEUR_FIXED)
        graph, _ = coarsener.coarsen()
    elif condition == "gnn":
        from .gnn_coarsening import GNNCoarsener
        model = _get_model(spec["model_path"])
        coarsener = GNNCoarsener(graph, P=spec.get("gnn_P", GNN_FIXED_P),
                                 radiusCoeff=spec.get("gnn_rc", GNN_FIXED_RC),
                                 depot_id=depot, model=model, device="cpu",
                                 stop_threshold=spec.get("gnn_stop_threshold"))
        graph, _ = coarsener.coarsen()
    # 'none' -> solve the full graph

    _, m, _ = solve_on_graph(graph, depot, inst.capacity, solver, cfg,
                             penalty_mode=penalty_mode, coarsener=coarsener)

    row = {
        "instance": inst.name, "family": inst.family, "family_group": inst.family_group,
        "N": N, "seed": seed, "condition": condition, "solver": solver,
        "penalty_mode": penalty_mode,
        "is_feasible": bool(m["is_feasible"]), "is_valid": bool(m["is_valid"]),
        "pre_feasible": bool(m["pre_feasible"]), "pre_total_viol": int(m["pre_total_viol"]),
        "total_distance": float(m["total_distance"]), "num_vehicles": int(m["num_vehicles"]),
        "cap_violations": int(m["capacity_violations"]),
        "tw_violations": int(m["time_window_violations"]),
        "coarsen_ratio": round(len(graph.nodes) / max(1, n0), 3),
        "num_qubo_vars": int(m["num_qubo_vars"]),
        "computation_time": float(m["computation_time"]),
    }
    ref = spec.get("ortools_cost")
    row["ortools_cost"] = float(ref) if ref is not None else float("nan")
    row["opt_gap"] = ((row["total_distance"] - ref) / ref
                      if (ref is not None and ref > 0 and m["is_feasible"])
                      else float("nan"))
    return row


def build_jobs(instances, Ns, seeds, conditions, solvers=("FQS",),
               penalty_mode="adaptive", num_reads=200, num_sweeps=200,
               model_path=None, ortools_cache=None,
               gnn_P=GNN_FIXED_P, gnn_rc=GNN_FIXED_RC,
               gnn_stop_threshold=None) -> list:
    jobs = []
    for N in Ns:
        for name in instances:
            ref = ortools_cache.get((name, N)) if ortools_cache else None
            for seed in seeds:
                for solver in solvers:
                    for cond in conditions:
                        jobs.append({
                            "name": name, "N": N, "seed": seed, "condition": cond,
                            "solver": solver, "penalty_mode": penalty_mode,
                            "num_reads": num_reads, "num_sweeps": num_sweeps,
                            "model_path": model_path, "ortools_cost": ref,
                            "gnn_P": gnn_P, "gnn_rc": gnn_rc,
                            "gnn_stop_threshold": gnn_stop_threshold,
                        })
    return jobs


def _worker_init():
    """Warm the heavy imports once per worker so per-job latency is just compute."""
    import numpy  # noqa: F401
    from . import datasets, coarsening, experiment, solvers, qubo, metrics  # noqa
    try:
        from dwave.samplers import SimulatedAnnealingSampler  # noqa: F401
    except Exception:
        pass
    try:
        from . import gnn_coarsening  # noqa: F401  (CPU-only torch)
    except Exception:
        pass


def run_grid(jobs, n_workers: Optional[int] = None, serial: bool = False,
             progress_every: int = 50, chunksize: int = 4) -> list:
    t0 = time.perf_counter()
    rows = []
    if serial or n_workers == 1:
        for i, j in enumerate(jobs):
            rows.append(run_one_job(j))
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  [{i+1}/{len(jobs)}] {time.perf_counter()-t0:.0f}s", flush=True)
        return rows
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as ex:
        for i, r in enumerate(ex.map(run_one_job, jobs, chunksize=chunksize)):
            rows.append(r)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  [{i+1}/{len(jobs)}] {time.perf_counter()-t0:.0f}s", flush=True)
    return rows
