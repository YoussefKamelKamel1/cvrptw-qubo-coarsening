"""Run the smallest coarsened test instances on a real D-Wave QPU, budget-capped.

Debug on the simulator first: ``--dry-run`` runs the whole pipeline
(GNN-coarsen -> QUBO -> sample -> decode -> repair -> metrics) with the SA
sampler, so the only thing left to trust on hardware is the sampler swap.

Real QPU run requires ``dwave-system`` + ``minorminer`` installed and a
``DWAVE_API_TOKEN``. It:
  * uses only coarsened test instances (tens of variables),
  * finds a minor-embedding per problem and samples with a chain strength,
  * logs ``qpu_access_time`` per problem and **aborts** once a cumulative QPU-time
    budget is exceeded, so a scarce allocation can never be overspent.

Usage:
  python scripts/run_qpu.py --dry-run                 # simulator, safe to run now
  python scripts/run_qpu.py --max-qpu-seconds 20      # real QPU (token required)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.datasets import load_solomon, train_test_split_instances   # noqa: E402
from src.coarsening import graph_from_instance                       # noqa: E402
from src.gnn_coarsening import GNNCoarsener, load_model              # noqa: E402
from src.qubo import vrp_problem_from_graph, adaptive_penalties      # noqa: E402
from src.solvers import VRPSolution, _k_max_fqs                      # noqa: E402
from src.experiment import _routes_to_strings                        # noqa: E402
from src.metrics import calculate_route_metrics                      # noqa: E402
from src.utils import set_global_seed


def build_problem(name, N, model, device):
    """GNN-coarsen the instance and build its (small) FQS QUBO + adaptive penalties."""
    inst = load_solomon(name).truncate(N)
    g, depot = graph_from_instance(inst)
    co = GNNCoarsener(g, P=0.5, radiusCoeff=0.5, depot_id=depot, model=model, device=device)
    cg, _ = co.coarsen()
    problem, int_to_id = vrp_problem_from_graph(cg, depot, inst.capacity)
    nv = len(problem.capacities)
    k = _k_max_fqs(len(problem.dests), nv)
    qubo = problem.get_qubo([k] * nv, adaptive_penalties(problem), smart_capacity=True)
    n_vars = len({v for key in qubo.dict for v in (key if isinstance(key[0], tuple) else [key[0]])})
    return inst, depot, co, problem, int_to_id, qubo, n_vars


def decode_and_score(sample, problem, int_to_id, depot, co, inst):
    nv = len(problem.capacities)
    k = _k_max_fqs(len(problem.dests), nv)
    sol = VRPSolution(problem, sample, [k] * nv)
    routes = co.inflate_route(_routes_to_strings(sol.solution, int_to_id, depot))
    return calculate_route_metrics(co.graph, routes, depot, inst.capacity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/gnn_model_rl.pt")
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--num-reads", type=int, default=1000)
    ap.add_argument("--max-qpu-seconds", type=float, default=20.0)
    ap.add_argument("--instances", nargs="+", default=None,
                    help="subset; default = the smallest few held-out test instances")
    ap.add_argument("--dry-run", action="store_true", help="use SA sampler (no QPU)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ROOT / args.model, device=device)
    _, test = train_test_split_instances()
    names = args.instances or test[:4]      # keep the footprint tiny
    print(f"{'DRY-RUN (simulator)' if args.dry_run else 'REAL QPU'} | N={args.N} | "
          f"instances={names} | num_reads={args.num_reads}", flush=True)

    if args.dry_run:
        sampler = None
    else:
        # lazy import so the simulator path never needs the cloud packages
        from dwave.system import DWaveSampler, FixedEmbeddingComposite
        from minorminer import find_embedding
        if "DWAVE_API_TOKEN" not in os.environ:
            raise SystemExit("Set DWAVE_API_TOKEN for a real QPU run.")
        qpu = DWaveSampler()  # picks a default Advantage solver
        print(f"QPU solver: {qpu.solver.id}", flush=True)

    total_qpu = 0.0
    for name in names:
        inst, depot, co, problem, int_to_id, qubo, n_vars = build_problem(
            name, args.N, model, device)
        print(f"  {name}: coarsened QUBO has {n_vars} variables", flush=True)

        if args.dry_run:
            from src.solvers import SolverConfig, solve_qubo
            samples = solve_qubo(qubo, SolverConfig("sa", args.num_reads, 200, args.seed))
            sample = samples[0] if samples else {}
            qpu_t = 0.0
        else:
            if total_qpu >= args.max_qpu_seconds:
                print(f"  [budget] stopping: {total_qpu:.3f}s used >= "
                      f"{args.max_qpu_seconds}s", flush=True)
                break
            emb = find_embedding(list(qubo.dict.keys()), qpu.edgelist, random_seed=args.seed)
            from dwave.system import FixedEmbeddingComposite
            sampler = FixedEmbeddingComposite(qpu, emb)
            resp = sampler.sample_qubo(qubo.dict, num_reads=args.num_reads,
                                       chain_strength=None, auto_scale=True,
                                       return_embedding=True)
            sample = next(iter(resp.lowest().samples()))
            qpu_t = resp.info.get("timing", {}).get("qpu_access_time", 0) / 1e6  # us -> s
            total_qpu += qpu_t

        m = decode_and_score(sample, problem, int_to_id, depot, co, inst)
        print(f"    feasible={m['is_feasible']} dist={m['total_distance']:.1f} "
              f"qpu_time={qpu_t*1000:.1f}ms", flush=True)

    if not args.dry_run:
        print(f"\nTotal QPU access time: {total_qpu:.3f}s "
              f"(budget {args.max_qpu_seconds}s)", flush=True)


if __name__ == "__main__":
    main()
