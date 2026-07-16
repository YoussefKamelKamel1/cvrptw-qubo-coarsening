"""Phase-1 experiment runner: reproduce the CVRPTW coarsening baseline.

Runs, per instance and seed, the 2x2 grid
    {no coarsening, heuristic coarsening} x {FQS, APS}
on the SA simulator, decoding + repairing + (for coarsened) inflating, then
scoring validity / feasibility / cost. Config- and seed-driven; every number is
regenerable from (instance, N, seed, SolverConfig, PenaltyWeights).

This is the control. Phase 2 swaps PenaltyWeights for adaptive calibration;
Phase 3 swaps the coarsener's merge scorer for the GNN -- both reuse this runner.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Optional

from .coarsening import SpatioTemporalGraphCoarsener, graph_from_instance
from .datasets import SOLOMON_FAMILY_HYPERPARAMS, load_solomon
from .metrics import calculate_route_metrics
from .qubo import (PenaltyCalibration, PenaltyWeights, adaptive_penalties,
                   vrp_problem_from_graph)
from .solvers import SolverConfig, solve_vrp
from .utils import set_global_seed


def _routes_to_strings(solution_int_routes, int_to_id, depot_id):
    """Map int-indexed customer routes to string ids, closed at the depot."""
    formatted = []
    for route in solution_int_routes:
        if not route:
            continue
        r = [int_to_id[i] for i in route]
        closed = [depot_id] + r + [depot_id]
        if len(closed) > 2:
            formatted.append(closed)
    return formatted


def solve_on_graph(graph, depot_id, capacity, solver, cfg,
                   penalty_mode="static", penalties=None, penalty_cal=None,
                   coarsener=None):
    """Solve one (graph, solver) condition. Returns (routes, metrics, sol).

    penalty_mode: 'static' uses ``penalties`` (default PenaltyWeights);
    'adaptive' calibrates penalties to *this* problem's objective scale
    (Phase 2), which also re-scales per coarsening level.
    """
    t0 = time.perf_counter()
    problem, int_to_id = vrp_problem_from_graph(graph, depot_id, capacity)
    # penalty modes (Phase-2 + its ablation levers):
    #   static        : static weights, keep all capacity constraints
    #   adaptive      : objective-scaled weights + drop non-binding capacity (both)
    #   scale_only    : objective-scaled weights, keep all capacity  (scaling lever)
    #   smartcap_only : static weights, drop non-binding capacity     (capacity lever)
    #   capcrush      : keep ALL capacity variables (var count == static) but crush
    #                   the capacity penalty so its (2^m)^2 couplings no longer
    #                   dominate the dynamic range -- isolates dynamic-range
    #                   conditioning from the variable-count reduction of smartcap.
    if penalty_mode in ("adaptive", "scale_only", "capcrush"):
        pw = adaptive_penalties(problem, penalty_cal)
    else:
        pw = penalties or PenaltyWeights()
    smart_capacity = penalty_mode in ("adaptive", "smartcap_only")
    if penalty_mode == "capcrush":
        import math
        from dataclasses import replace
        Q = problem.capacities[0] if problem.capacities else 0
        if Q > 0:
            m_max = math.floor(math.log2(max(1.0, Q)))
            # largest capacity coupling ~ time-window penalty (no longer dominant),
            # while every capacity slack variable is retained.
            pw = replace(pw, capacity_penalty=pw.time_window_penalty / (2 ** m_max) ** 2)

    sol = solve_vrp(problem, solver, cfg, pw, smart_capacity=smart_capacity)
    pre = sol.report(sol.raw_solution)  # pre-repair SA quality (int space)
    routes = _routes_to_strings(sol.solution, int_to_id, depot_id)

    metrics_graph = graph
    if coarsener is not None:
        routes = coarsener.inflate_route(routes)
        metrics_graph = coarsener.graph  # original (uncoarsened) subgraph
    metrics = calculate_route_metrics(metrics_graph, routes, depot_id, capacity)
    metrics["computation_time"] = time.perf_counter() - t0
    metrics["num_qubo_vars"] = _count_vars(problem, solver, smart_capacity)
    metrics["pre_tw_viol"] = pre["tw_violations"]
    metrics["pre_cap_viol"] = pre["cap_violations"]
    metrics["pre_total_viol"] = pre["total_violations"]
    metrics["pre_feasible"] = pre["feasible"]
    return routes, metrics, sol


def _count_vars(problem, solver, smart_capacity: bool = False) -> int:
    """Approx QUBO variable count = vehicles * customers * k_max + slack bits."""
    import math
    nc = len(problem.dests)
    nv = len(problem.capacities)
    avg = math.ceil(nc / nv) if nv else 0
    k_max = min(avg + 1, nc)
    total_demand = sum(problem.weights.get(j, 0) for j in problem.dests)
    slack = 0
    for cap in problem.capacities:
        if cap <= 0:
            continue
        if smart_capacity and total_demand <= cap:
            continue  # non-binding capacity dropped
        slack_cap = max(1.0, min(cap, total_demand) if smart_capacity else cap)
        slack += math.floor(math.log2(slack_cap)) + 1
    return nv * nc * k_max + slack


@dataclass
class RunResult:
    instance: str
    family: str
    family_group: str
    N: int
    seed: int
    condition: str          # 'none' | 'heuristic'
    solver: str             # 'FQS' | 'APS'
    penalty_mode: str       # 'static' | 'adaptive'
    is_valid: bool
    is_feasible: bool
    total_distance: float
    num_vehicles: int
    num_customers_visited: int
    cap_violations: int
    tw_violations: int
    # pre-repair (raw SA output) quality
    pre_feasible: bool
    pre_tw_viol: int
    pre_cap_viol: int
    pre_total_viol: int
    num_qubo_vars: int
    computation_time: float

    def as_row(self) -> dict:
        return asdict(self)


def run_instance(name: str, N: int, cfg: SolverConfig,
                 penalties: Optional[PenaltyWeights] = None,
                 penalty_mode: str = "static",
                 penalty_cal: Optional[PenaltyCalibration] = None,
                 coarsen_params: Optional[dict] = None) -> list[RunResult]:
    """Run the 2x2 grid for one instance/N/seed. Returns a list of RunResult."""
    penalties = penalties or PenaltyWeights()
    set_global_seed(cfg.seed if cfg.seed is not None else 0)

    inst = load_solomon(name).truncate(N)
    fam = inst.family
    if coarsen_params is None:
        coarsen_params = SOLOMON_FAMILY_HYPERPARAMS.get(fam, {
            "alpha": 1.0, "beta": 1.0, "P": 0.5, "radiusCoeff": 2.0})

    results = []
    for solver in ("FQS", "APS"):
        # --- no coarsening ---
        graph, depot = graph_from_instance(inst)
        _, m, _ = solve_on_graph(graph, depot, inst.capacity, solver, cfg,
                                 penalty_mode=penalty_mode, penalties=penalties,
                                 penalty_cal=penalty_cal)
        results.append(_mk(name, inst, N, cfg, "none", solver, penalty_mode, m))

        # --- heuristic coarsening ---
        graph, depot = graph_from_instance(inst)
        coarsener = SpatioTemporalGraphCoarsener(
            graph=graph, depot_id=depot, **coarsen_params)
        coarsened, _layers = coarsener.coarsen()
        _, m, _ = solve_on_graph(coarsened, depot, inst.capacity, solver, cfg,
                                 penalty_mode=penalty_mode, penalties=penalties,
                                 penalty_cal=penalty_cal, coarsener=coarsener)
        results.append(_mk(name, inst, N, cfg, "heuristic", solver, penalty_mode, m))
    return results


def _mk(name, inst, N, cfg, condition, solver, penalty_mode, m) -> RunResult:
    return RunResult(
        instance=inst.name, family=inst.family, family_group=inst.family_group,
        N=N, seed=cfg.seed, condition=condition, solver=solver,
        penalty_mode=penalty_mode,
        is_valid=bool(m["is_valid"]), is_feasible=bool(m["is_feasible"]),
        total_distance=float(m["total_distance"]), num_vehicles=int(m["num_vehicles"]),
        num_customers_visited=int(m["num_customers_visited"]),
        cap_violations=int(m["capacity_violations"]),
        tw_violations=int(m["time_window_violations"]),
        pre_feasible=bool(m["pre_feasible"]), pre_tw_viol=int(m["pre_tw_viol"]),
        pre_cap_viol=int(m["pre_cap_viol"]), pre_total_viol=int(m["pre_total_viol"]),
        num_qubo_vars=int(m["num_qubo_vars"]),
        computation_time=float(m["computation_time"]),
    )


def run_phase1(instances: list[str], Ns: list[int], seeds: list[int],
               cfg: Optional[SolverConfig] = None,
               penalties: Optional[PenaltyWeights] = None):
    """Run the full Phase-1 grid; return a pandas DataFrame of RunResults."""
    import pandas as pd
    base = cfg or SolverConfig()
    rows = []
    for name in instances:
        for N in Ns:
            for seed in seeds:
                run_cfg = SolverConfig(backend=base.backend, num_reads=base.num_reads,
                                       num_sweeps=base.num_sweeps, seed=seed)
                for r in run_instance(name, N, run_cfg, penalties):
                    rows.append(r.as_row())
    return pd.DataFrame(rows)


def run_ablation(instances: list[str], Ns: list[int], seeds: list[int],
                 cfg: Optional[SolverConfig] = None,
                 penalties: Optional[PenaltyWeights] = None,
                 penalty_cal: Optional[PenaltyCalibration] = None):
    """Phase-2 ablation: static vs adaptive penalties at *equal* solver budget.

    Returns a DataFrame with a ``penalty_mode`` column so the two arms can be
    compared on pre-repair violations and feasibility.
    """
    import pandas as pd
    base = cfg or SolverConfig()
    rows = []
    for mode in ("static", "adaptive"):
        for name in instances:
            for N in Ns:
                for seed in seeds:
                    run_cfg = SolverConfig(backend=base.backend, num_reads=base.num_reads,
                                           num_sweeps=base.num_sweeps, seed=seed)
                    for r in run_instance(name, N, run_cfg, penalties=penalties,
                                          penalty_mode=mode, penalty_cal=penalty_cal):
                        rows.append(r.as_row())
    return pd.DataFrame(rows)
