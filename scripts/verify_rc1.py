"""Verify the RC1 binding-capacity fix: old (flat 8B) vs new (binding-aware) cap.

Runs RC1 (binding) and a C1/R1 control (non-binding) with heuristic coarsening +
adaptive penalties, comparing post-repair feasibility.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import load_solomon, SOLOMON_FAMILY_HYPERPARAMS
from src.coarsening import graph_from_instance, SpatioTemporalGraphCoarsener
from src.qubo import vrp_problem_from_graph, adaptive_penalties, PenaltyWeights
from src.solvers import SolverConfig, solve_vrp, VRPSolution
from src.experiment import _routes_to_strings
from src.metrics import calculate_route_metrics
from src.utils import set_global_seed


def old_penalties(problem):
    """Reconstruct the pre-fix adaptive penalties (flat cap = 8B)."""
    n = len(problem.costs)
    max_edge = max((problem.costs[i][j] for i in range(n) for j in range(n) if i != j),
                   default=1.0)
    max_edge = max(max_edge, 1.0)
    B = 100.0 * max_edge * max(1, len(problem.dests))
    return PenaltyWeights(only_one=15 * B, capacity_penalty=8 * B,
                          time_window_penalty=5 * B,
                          vehicle_start_cost=100.0 * max_edge, order=100.0)


def feas(inst, penalties_fn, seeds=(0, 1, 2)):
    ok = 0
    for seed in seeds:
        set_global_seed(seed)
        cfg = SolverConfig("sa", 200, 200, seed=seed)
        g, depot = graph_from_instance(inst)
        co = SpatioTemporalGraphCoarsener(graph=g, depot_id=depot,
                                          **SOLOMON_FAMILY_HYPERPARAMS[inst.family])
        cg, _ = co.coarsen()
        problem, int_to_id = vrp_problem_from_graph(cg, depot, inst.capacity)
        pw = penalties_fn(problem)
        sol = solve_vrp(problem, "FQS", cfg, pw, smart_capacity=True)
        routes = _routes_to_strings(sol.solution, int_to_id, depot)
        routes = co.inflate_route(routes)
        m = calculate_route_metrics(co.graph, routes, depot, inst.capacity)
        ok += int(m["is_feasible"])
    return 100.0 * ok / len(seeds)


def main():
    N = 10
    rc1 = ["RC101", "RC102", "RC103", "RC104", "RC105", "RC106", "RC107", "RC108"]
    control = ["C101", "C102", "R101", "R102"]
    for label, names in [("RC1 (binding)", rc1), ("C1/R1 control (non-binding)", control)]:
        old_f, new_f = [], []
        for name in names:
            inst = load_solomon(name).truncate(N)
            old_f.append(feas(inst, old_penalties))
            new_f.append(feas(inst, adaptive_penalties))
        print(f"{label}: old(flat 8B) {sum(old_f)/len(old_f):.1f}%  "
              f"-> new(binding-aware) {sum(new_f)/len(new_f):.1f}%")


if __name__ == "__main__":
    main()
