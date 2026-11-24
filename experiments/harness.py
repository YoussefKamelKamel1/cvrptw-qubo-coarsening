"""Run the pipeline with small in-memory changes to its terms (used by exp1, exp4, exp8)."""
from __future__ import annotations

import contextlib
import inspect
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(ROOT))

import src.qubo as qubo_mod                                        # noqa: E402
from src.coarsening import graph_from_instance, SpatioTemporalGraphCoarsener  # noqa: E402
from src.datasets import load_solomon, SOLOMON_FAMILY_HYPERPARAMS  # noqa: E402
from src.experiment import solve_on_graph                          # noqa: E402
from src.solvers import SolverConfig                               # noqa: E402
from src.utils import set_global_seed                              # noqa: E402
from src.runner_parallel import GNN_FIXED_P, GNN_FIXED_RC          # noqa: E402

_ORIG_GET_QUBO = qubo_mod.VRPProblem.get_qubo
_MODEL = {}


def _model():
    if "m" not in _MODEL:
        from src.gnn_coarsening import load_model
        _MODEL["m"] = load_model(ROOT / "results" / "gnn_model_rl.pt", device="cpu")
        _MODEL["m"].eval()
    return _MODEL["m"]


def _patched_get_qubo(cont_factor: float | None = None,
                      tight_scale: float | None = None,
                      self_scale: float | None = None):
    """The original get_qubo with only the named expressions changed.

    cont_factor : replaces the 0.5 in the continuity pair term
    tight_scale : multiplies the graded ("tight") time-window weight
    Each target expression is asserted to occur exactly once.
    """
    src = textwrap.dedent(inspect.getsource(_ORIG_GET_QUBO))
    ns = dict(vars(qubo_mod))
    if cont_factor is not None:
        t = "-cont * 0.5"
        assert src.count(t) == 1, t
        src = src.replace(t, "-cont * _CONT_FACTOR")
        ns["_CONT_FACTOR"] = float(cont_factor)
    if tight_scale is not None:
        t = "tight = twp * frac * 0.5"
        assert src.count(t) == 1, t
        src = src.replace(t, "tight = twp * frac * 0.5 * _TIGHT_SCALE")
        ns["_TIGHT_SCALE"] = float(tight_scale)
    if self_scale is not None:
        t = "selfloop = pw.selfloop_penalty"
        assert src.count(t) == 1, t
        src = src.replace(t, "selfloop = pw.selfloop_penalty * _SELF_SCALE")
        ns["_SELF_SCALE"] = float(self_scale)
    exec(compile(src, "<variant>", "exec"), ns)
    return ns["get_qubo"]


import src.experiment as experiment_mod                            # noqa: E402

_ORIG_VRP_FROM_GRAPH = experiment_mod.vrp_problem_from_graph


def _fleet_wrapper(divisor: int):
    """vrp_problem_from_graph with fleet n_v = max(2, n_c // divisor).

    The published default is divisor 2; k_max = ceil(n_c / n_v) + 1 follows
    automatically, so a smaller fleet means longer allowed routes.
    """
    def wrapped(graph, depot_id, capacity, num_vehicles=None):
        n_c = len(graph.nodes) - 1
        return _ORIG_VRP_FROM_GRAPH(graph, depot_id, capacity,
                                    num_vehicles=max(2, n_c // divisor))
    return wrapped


@contextlib.contextmanager
def variant(cont_factor: float | None = None, fleet_divisor: int | None = None,
            tight_scale: float | None = None, self_scale: float | None = None):
    """Apply a QUBO variant in memory; always restore the originals afterwards."""
    try:
        if cont_factor is not None or tight_scale is not None or self_scale is not None:
            qubo_mod.VRPProblem.get_qubo = _patched_get_qubo(cont_factor, tight_scale,
                                                             self_scale)
        if fleet_divisor is not None:
            experiment_mod.vrp_problem_from_graph = _fleet_wrapper(fleet_divisor)
        yield
    finally:
        qubo_mod.VRPProblem.get_qubo = _ORIG_GET_QUBO
        experiment_mod.vrp_problem_from_graph = _ORIG_VRP_FROM_GRAPH


def run_job(name: str, N: int, seed: int, condition: str = "gnn",
            num_reads: int = 200, num_sweeps: int = 200,
            ortools_cost: float | None = None, oracle_routes=None) -> dict:
    """One pipeline run, matching run_one_job, plus pre/post-repair vehicle counts."""
    set_global_seed(seed)
    inst = load_solomon(name).truncate(N)
    cfg = SolverConfig("sa", num_reads, num_sweeps, seed=seed)
    graph, depot = graph_from_instance(inst)
    n0 = len(graph.nodes)
    coarsener = None
    if condition == "heuristic":
        coarsener = SpatioTemporalGraphCoarsener(
            graph=graph, depot_id=depot, **SOLOMON_FAMILY_HYPERPARAMS[inst.family])
        graph, _ = coarsener.coarsen()
    elif condition == "gnn":
        from src.gnn_coarsening import GNNCoarsener
        coarsener = GNNCoarsener(graph, P=GNN_FIXED_P, radiusCoeff=GNN_FIXED_RC,
                                 depot_id=depot, model=_model(), device="cpu",
                                 stop_threshold=None)
        graph, _ = coarsener.coarsen()
    elif condition == "oracle":
        from oracle_coarsener import OracleCoarsener
        coarsener = OracleCoarsener(graph=graph, depot_id=depot, alpha=1.0, beta=1.0,
                                    P=GNN_FIXED_P, radiusCoeff=GNN_FIXED_RC,
                                    routes=oracle_routes)
        graph, _ = coarsener.coarsen()

    _, m, sol = solve_on_graph(graph, depot, inst.capacity, "FQS", cfg,
                               penalty_mode="adaptive", coarsener=coarsener)
    row = dict(instance=inst.name, family=inst.family, family_group=inst.family_group,
               N=N, seed=seed, condition=condition,
               is_feasible=bool(m["is_feasible"]), pre_feasible=bool(m["pre_feasible"]),
               pre_total_viol=int(m["pre_total_viol"]),
               total_distance=float(m["total_distance"]),
               num_vehicles=int(m["num_vehicles"]),
               pre_vehicles=sum(1 for r in sol.raw_solution if r),
               post_vehicles=sum(1 for r in sol.solution if r),
               num_qubo_vars=int(m["num_qubo_vars"]),
               coarsen_ratio=round(len(graph.nodes) / max(1, n0), 3))
    ref = ortools_cost
    row["ortools_cost"] = float(ref) if ref else float("nan")
    row["opt_gap"] = ((row["total_distance"] - ref) / ref
                      if (ref and ref > 0 and m["is_feasible"]) else float("nan"))
    return row


def ortools_cache() -> dict:
    """The project's cached OR-Tools reference costs, read-only."""
    from src.refcache import load_cache
    return load_cache(ROOT / "results" / "ortools_cache.csv")
