"""Does graph coarsening transfer to the route-activation (edge-based) encoding?

For each coarsening (none / heuristic / gnn), build the route-activation QUBO,
solve it with simulated annealing, decode + repair, inflate, and score. Reports
QUBO size and feasibility so coarsening's effect on a different encoding is visible.
Uses only the reference pipeline's public pieces plus src/route_activation.py.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import (load_solomon, train_test_split_instances,        # noqa: E402
                          SOLOMON_FAMILY_HYPERPARAMS)
from src.coarsening import (graph_from_instance,                            # noqa: E402
                            SpatioTemporalGraphCoarsener)
from src.gnn_coarsening import GNNCoarsener, load_model                     # noqa: E402
from src.qubo import vrp_problem_from_graph                                # noqa: E402
from src.solvers import VRPSolution, SolverConfig, solve_qubo, _k_max_fqs   # noqa: E402
from src.experiment import _routes_to_strings                             # noqa: E402
from src.metrics import calculate_route_metrics                           # noqa: E402
from src.utils import set_global_seed                                     # noqa: E402
from src.route_activation import build_ras_qubo, decode_ras               # noqa: E402


def solve_one(name, N, coarsen, model, device, seed, reads, sweeps):
    inst = load_solomon(name).truncate(N)
    graph, depot = graph_from_instance(inst)
    coarsener = None
    if coarsen == "gnn":
        coarsener = GNNCoarsener(graph, P=0.5, radiusCoeff=0.5, depot_id=depot,
                                 model=model, device=device)
        graph, _ = coarsener.coarsen()
    elif coarsen == "heuristic":
        coarsener = SpatioTemporalGraphCoarsener(
            graph=graph, depot_id=depot, **SOLOMON_FAMILY_HYPERPARAMS[inst.family])
        graph, _ = coarsener.coarsen()

    problem, int_to_id = vrp_problem_from_graph(graph, depot, inst.capacity)
    qubo = build_ras_qubo(problem)
    n_vars = len({v for key in qubo.dict for v in key})

    sample = (solve_qubo(qubo, SolverConfig("sa", reads, sweeps, seed)) or [{}])[0]
    routes_int = decode_ras(sample, problem)

    nv = len(problem.capacities)
    binder = VRPSolution(problem, {}, [_k_max_fqs(len(problem.dests), nv)] * nv, solution=[])
    repaired = binder._repair_time_windows(binder._repair_solution(routes_int))
    routes = _routes_to_strings(repaired, int_to_id, depot)

    metrics_graph = graph
    if coarsener is not None:
        routes = coarsener.inflate_route(routes)
        metrics_graph = coarsener.graph
    m = calculate_route_metrics(metrics_graph, routes, depot, inst.capacity)
    return n_vars, bool(m["is_feasible"]), float(m["total_distance"])


def main():
    import numpy as np
    import pandas as pd

    ap_reads, ap_sweeps = 200, 200
    _, test = train_test_split_instances()
    device = "cpu"
    model = load_model(ROOT / "results" / "gnn_model_rl.pt", device=device)
    Ns = [10, 15, 20]
    seeds = [0, 1, 2]
    conditions = ["none", "heuristic", "gnn"]
    instances = test[:6]

    print(f"Route-activation encoding | {len(instances)} instances x {seeds} seeds")
    t0 = time.perf_counter()
    rows = []
    for N in Ns:
        print(f"-- N={N} --")
        for cond in conditions:
            feas, vars_, dists = [], [], []
            for name in instances:
                for seed in seeds:
                    set_global_seed(seed)
                    nv, f, d = solve_one(name, N, cond, model, device, seed,
                                         ap_reads, ap_sweeps)
                    vars_.append(nv)
                    feas.append(f)
                    if f:
                        dists.append(d)
            print(f"   {cond:10s} vars={np.mean(vars_):5.0f}  feas={100*np.mean(feas):5.1f}%  "
                  f"dist={np.mean(dists) if dists else float('nan'):6.1f}")
            rows.append(dict(N=N, coarsen=cond, vars=round(float(np.mean(vars_)), 1),
                             feasible_pct=round(100 * float(np.mean(feas)), 1)))
    pd.DataFrame(rows).to_csv(ROOT / "results" / "route_activation.csv", index=False)
    print(f"done in {time.perf_counter()-t0:.0f}s -> results/route_activation.csv")


if __name__ == "__main__":
    main()
