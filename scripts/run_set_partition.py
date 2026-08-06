"""Does coarsening help the set-partitioning (route-selection) encoding?

For each coarsening (none / heuristic / gnn): coarsen the graph, generate a pool of
feasible routes, build the set-partition QUBO, solve it with simulated annealing,
decode the selected routes, then inflate and score on the original graph. Reports
QUBO size (pool), exact-cover rate before repair, post-repair feasibility, and cost.
Uses only the reference pipeline's public pieces plus src/set_partition.py.

  python scripts/run_set_partition.py --N 10 15 20 --n-instances 3
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                     # noqa: E402

from src.datasets import (load_solomon, train_test_split_instances,   # noqa: E402
                          SOLOMON_FAMILY_HYPERPARAMS)
from src.coarsening import (graph_from_instance,                       # noqa: E402
                            SpatioTemporalGraphCoarsener)
from src.gnn_coarsening import GNNCoarsener, load_model               # noqa: E402
from src.qubo import vrp_problem_from_graph                           # noqa: E402
from src.solvers import SolverConfig, solve_qubo                      # noqa: E402
from src.experiment import _routes_to_strings                        # noqa: E402
from src.metrics import calculate_route_metrics                      # noqa: E402
from src.utils import set_global_seed                                # noqa: E402
from src.set_partition import (generate_pool, build_setpartition_qubo,  # noqa: E402
                               decode_setpartition, cover_status, repair_cover)


def solve_one(name, N, coarsen, model, device, seed, reads, sweeps, n_perms):
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
    pool = generate_pool(problem, n_perms=n_perms, seed=seed)
    qubo = build_setpartition_qubo(problem, pool)
    n_vars = len(pool)

    # Full sample distribution, so the exact-cover rate is comparable with the
    # position encoding's pre-repair feasibility (which is also over all samples)
    # rather than being a best-of-N number.
    from dwave.samplers import SimulatedAnnealingSampler
    ss = SimulatedAnnealingSampler().sample_qubo(qubo.dict, num_reads=reads,
                                                 num_sweeps=sweeps, seed=seed)
    covered = total = 0
    for d in ss.aggregate().data(["sample", "num_occurrences"]):
        occ = int(d.num_occurrences)
        total += occ
        if cover_status(decode_setpartition(dict(d.sample), pool), problem)[0]:
            covered += occ
    raw_cover_frac = covered / total if total else 0.0

    sample = dict(next(iter(ss.lowest().samples())))
    selected = decode_setpartition(sample, pool)
    exact, n_missing, n_dup = cover_status(selected, problem)

    routes_int = repair_cover(selected, problem)
    routes = _routes_to_strings(routes_int, int_to_id, depot)
    metrics_graph = graph
    if coarsener is not None:
        routes = coarsener.inflate_route(routes)
        metrics_graph = coarsener.graph
    m = calculate_route_metrics(metrics_graph, routes, depot, inst.capacity)
    return dict(n_vars=n_vars, exact_cover=exact, raw_cover_frac=raw_cover_frac, n_missing=n_missing, n_dup=n_dup,
                feasible=bool(m["is_feasible"]), dist=float(m["total_distance"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, nargs="+", default=[10, 15, 20])
    ap.add_argument("--coarsen", nargs="+", default=["none", "heuristic", "gnn"])
    ap.add_argument("--instances", nargs="+", default=None)
    ap.add_argument("--n-instances", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--num-reads", type=int, default=500)
    ap.add_argument("--num-sweeps", type=int, default=200)
    ap.add_argument("--n-perms", type=int, default=60)
    ap.add_argument("--model", default="results/gnn_model_rl.pt")
    ap.add_argument("--out", default="results/set_partition.csv")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ROOT / args.model, device=device)
    _, test = train_test_split_instances()
    names = args.instances or test[:args.n_instances]
    print(f"Set-partition encoding | instances={names} | N={args.N} | "
          f"coarsen={args.coarsen} | seeds={args.seeds}\n", flush=True)

    rows = []
    t0 = time.perf_counter()
    for N in args.N:
        print(f"-- N={N} --", flush=True)
        for coarsen in args.coarsen:
            vars_, exact, feas, dists, rawc = [], [], [], [], []
            for name in names:
                for seed in args.seeds:
                    set_global_seed(seed)
                    r = solve_one(name, N, coarsen, model, device, seed,
                                  args.num_reads, args.num_sweeps, args.n_perms)
                    vars_.append(r["n_vars"])
                    exact.append(r["exact_cover"])
                    rawc.append(r["raw_cover_frac"])
                    feas.append(r["feasible"])
                    if r["feasible"]:
                        dists.append(r["dist"])
                    rows.append(dict(N=N, coarsen=coarsen, instance=name, seed=seed, **r))
            print(f"   {coarsen:10s} pool={np.mean(vars_):5.0f}  "
                  f"exact_cover={100*np.mean(exact):5.1f}%  raw_cover={100*np.mean(rawc):5.1f}%  feas={100*np.mean(feas):5.1f}%  "
                  f"dist={np.mean(dists) if dists else float('nan'):6.1f}", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\ndone in {time.perf_counter()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
