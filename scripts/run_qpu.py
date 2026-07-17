"""Run coarsened CVRPTW QUBOs on a D-Wave quantum annealer (budget-capped).

This is the turnkey hardware script. The scientific question is NOT "beat
classical" (a classical repair is cheaper); it is:

  1. Does penalty conditioning matter MORE on hardware than in simulation?
     A physical annealer has limited coupler precision, so a QUBO with a huge
     internal dynamic range (static penalties, dominated by the (2^m)^2 capacity
     slack couplings) loses its objective/time-window couplings into the analog
     noise floor after auto-scaling. Adaptive conditioning should therefore help
     more on the QPU than on simulated annealing. We run static vs adaptive at
     equal budget and compare the delta.
  2. Does coarsening make the problem embeddable at all (chain length, qubits)?
  3. Does the tuning-free GNN coarsening transfer to real hardware?

Reproducibility / device pinning
--------------------------------
The solver is selected explicitly and every chip property is logged, so a run is
never ambiguous about WHICH device produced it (Advantage/Pegasus vs
Advantage2/Zephyr differ in topology, qubit count and noise). Pass --solver or
--topology; if neither is given the script prints the available solvers and stops
rather than silently picking one.

Usage
-----
  # simulator, safe to run now (validates the whole pipeline):
  python scripts/run_qpu.py --dry-run

  # list the QPU solvers available to your Leap account, then stop:
  python scripts/run_qpu.py --list-solvers

  # real QPU run (needs dwave-system, minorminer, DWAVE_API_TOKEN):
  python scripts/run_qpu.py --solver Advantage_system6.4 \
      --arms static adaptive --coarsen gnn --N 10 --max-qpu-seconds 20
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.datasets import (load_solomon, train_test_split_instances,        # noqa: E402
                          SOLOMON_FAMILY_HYPERPARAMS)
from src.coarsening import (graph_from_instance,                            # noqa: E402
                            SpatioTemporalGraphCoarsener)
from src.gnn_coarsening import GNNCoarsener, load_model                     # noqa: E402
from src.qubo import (vrp_problem_from_graph, adaptive_penalties,           # noqa: E402
                      PenaltyWeights)
from src.solvers import VRPSolution, _k_max_fqs                             # noqa: E402
from src.experiment import _routes_to_strings                              # noqa: E402
from src.metrics import calculate_route_metrics                            # noqa: E402
from src.utils import set_global_seed


# --------------------------------------------------------------------------- #
# problem construction
# --------------------------------------------------------------------------- #
def build_problem(name, N, coarsen, penalty_mode, model, device):
    """Return (inst, depot, coarsener|None, problem, int_to_id, qubo).

    coarsen      : 'gnn' | 'heuristic' | 'none'
    penalty_mode : 'adaptive' (objective-scaled + drop non-binding capacity)
                   'static'   (fixed penalties, all capacity constraints kept)
    """
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
    nv = len(problem.capacities)
    k = _k_max_fqs(len(problem.dests), nv)
    if penalty_mode == "adaptive":
        qubo = problem.get_qubo([k] * nv, adaptive_penalties(problem), smart_capacity=True)
    else:
        qubo = problem.get_qubo([k] * nv, PenaltyWeights(), smart_capacity=False)
    return inst, depot, coarsener, problem, int_to_id, qubo


def qubo_stats(qubo, floor: float):
    """Variable count and internal dynamic range of the QUBO.

    ``n_below_floor`` = couplings whose magnitude, relative to the largest, falls
    below ``floor`` -- a proxy for terms the QPU cannot resolve once the whole
    problem is auto-scaled into the coupler range. This is why static penalties
    (huge dynamic range) are expected to fare worse on hardware.
    """
    variables = set()
    coeffs = []
    for key, c in qubo.dict.items():
        u, v = key
        variables.add(u)
        variables.add(v)
        if c != 0:
            coeffs.append(abs(c))
    if not coeffs:
        return len(variables), 1.0, 0
    hi, lo = max(coeffs), min(coeffs)
    n_below = sum(1 for c in coeffs if c / hi < floor)
    return len(variables), hi / lo, n_below


def source_graph(qubo):
    """networkx graph of the QUBO (nodes = variables, edges = quadratic terms)."""
    import networkx as nx
    g = nx.Graph()
    for (u, v), c in qubo.dict.items():
        g.add_node(u)
        g.add_node(v)
        if u != v and c != 0:
            g.add_edge(u, v)
    return g


def decode_and_score(sample, problem, int_to_id, depot, coarsener, inst):
    nv = len(problem.capacities)
    k = _k_max_fqs(len(problem.dests), nv)
    sol = VRPSolution(problem, sample, [k] * nv)
    routes = _routes_to_strings(sol.solution, int_to_id, depot)
    if coarsener is not None:
        routes = coarsener.inflate_route(routes)
        metrics_graph = coarsener.graph
    else:
        g, _ = graph_from_instance(inst)
        metrics_graph = g
    return calculate_route_metrics(metrics_graph, routes, depot, inst.capacity)


# --------------------------------------------------------------------------- #
# device selection (explicit + logged, so runs are never version-ambiguous)
# --------------------------------------------------------------------------- #
def list_solvers():
    from dwave.cloud import Client
    with Client.from_config() as client:
        rows = []
        for s in client.get_solvers(qpu=True):
            props = s.properties
            topo = props.get("topology", {}).get("type", "?")
            rows.append((s.id, topo, props.get("num_qubits", len(props.get("qubits", [])))))
    print("Available QPU solvers (name | topology | qubits):")
    for sid, topo, nq in rows:
        print(f"  {sid} | {topo} | {nq}")
    return rows


def make_qpu(args):
    from dwave.system import DWaveSampler
    if args.solver:
        qpu = DWaveSampler(solver=args.solver)
    elif args.topology:
        qpu = DWaveSampler(solver=dict(topology__type=args.topology))
    else:
        raise SystemExit(
            "Refusing to pick a solver implicitly. Pass --solver <name> or "
            "--topology {pegasus,zephyr}; use --list-solvers to see the options.")
    p = qpu.properties
    print("=" * 66)
    print(f"DEVICE      : {qpu.solver.id}")
    print(f"topology    : {p.get('topology', {}).get('type', '?')}")
    print(f"qubits      : {p.get('num_qubits', len(qpu.nodelist))}  "
          f"couplers: {len(qpu.edgelist)}")
    print(f"coupler J   : {p.get('j_range')}   bias h: {p.get('h_range')}")
    print(f"anneal (us) : {p.get('default_annealing_time')}  "
          f"range {p.get('annealing_time_range')}")
    print("=" * 66, flush=True)
    return qpu, qpu.solver.id, p.get("topology", {}).get("type", "?")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="results/gnn_model_rl.pt")
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--arms", nargs="+", default=["adaptive"],
                    choices=["adaptive", "static"],
                    help="penalty modes to compare (use both for the precision study)")
    ap.add_argument("--coarsen", default="gnn", choices=["gnn", "heuristic", "none"])
    ap.add_argument("--instances", nargs="+", default=None,
                    help="subset; default = the smallest held-out test instances")
    ap.add_argument("--num-reads", type=int, default=1000)
    ap.add_argument("--annealing-time", type=float, default=None, help="microseconds")
    ap.add_argument("--chain-strength", type=float, default=None,
                    help="fixed chain strength; default = auto (torque compensation)")
    ap.add_argument("--spin-reversals", type=int, default=0,
                    help="gauge-averaging spin-reversal transforms (0 = off)")
    ap.add_argument("--precision-floor", type=float, default=0.03,
                    help="relative coupler-resolution floor for the diagnostic")
    ap.add_argument("--max-qpu-seconds", type=float, default=20.0)
    ap.add_argument("--solver", default=None, help="explicit D-Wave solver name")
    ap.add_argument("--topology", default=None, choices=["pegasus", "zephyr"])
    ap.add_argument("--list-solvers", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="use the SA simulator")
    ap.add_argument("--out", default="results/qpu_runs.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.list_solvers:
        list_solvers()
        return

    set_global_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ROOT / args.model, device=device)
    _, test = train_test_split_instances()
    names = args.instances or test[:4]

    qpu = solver_id = topology = None
    if not args.dry_run:
        from minorminer import find_embedding  # noqa: F401  (checked here early)
        from dwave.system import FixedEmbeddingComposite  # noqa: F401
        if "DWAVE_API_TOKEN" not in os.environ:
            raise SystemExit("Set DWAVE_API_TOKEN for a real QPU run.")
        qpu, solver_id, topology = make_qpu(args)

    print(f"{'DRY-RUN (simulator)' if args.dry_run else 'REAL QPU'} | N={args.N} | "
          f"coarsen={args.coarsen} | arms={args.arms} | instances={names}", flush=True)

    rows = []
    total_qpu = 0.0
    stop = False
    for name in names:
        if stop:
            break
        for mode in args.arms:
            inst, depot, coarsener, problem, int_to_id, qubo = build_problem(
                name, args.N, args.coarsen, mode, model, device)
            n_vars, dyn_range, n_below = qubo_stats(qubo, args.precision_floor)
            row = dict(instance=name, N=args.N, coarsen=args.coarsen, penalty_mode=mode,
                       n_vars=n_vars, dynamic_range=round(dyn_range, 1),
                       couplings_below_floor=n_below, solver=solver_id, topology=topology,
                       num_reads=args.num_reads)
            print(f"  {name} [{mode:8s}] vars={n_vars} dyn_range={dyn_range:.1f} "
                  f"below_floor={n_below}", flush=True)

            if args.dry_run:
                from src.solvers import SolverConfig, solve_qubo
                s = solve_qubo(qubo, SolverConfig("sa", args.num_reads, 200, args.seed))
                sample = s[0] if s else {}
                row.update(max_chain=0, phys_qubits=0, chain_break_frac=0.0, qpu_time_s=0.0)
            else:
                if total_qpu >= args.max_qpu_seconds:
                    print(f"  [budget] {total_qpu:.3f}s used >= {args.max_qpu_seconds}s; "
                          "stopping.", flush=True)
                    stop = True
                    break
                from minorminer import find_embedding
                from dwave.system import FixedEmbeddingComposite
                emb = find_embedding(source_graph(qubo), qpu.edgelist, random_seed=args.seed)
                if not emb:
                    print("    embedding FAILED (problem too large for this chip)", flush=True)
                    row.update(max_chain=-1, phys_qubits=-1, chain_break_frac=None,
                               qpu_time_s=0.0, feasible=None, total_distance=None)
                    rows.append(row)
                    continue
                row["max_chain"] = max(len(c) for c in emb.values())
                row["phys_qubits"] = sum(len(c) for c in emb.values())
                sampler = FixedEmbeddingComposite(qpu, emb)
                kw = dict(num_reads=args.num_reads, auto_scale=True, return_embedding=True)
                if args.chain_strength is not None:
                    kw["chain_strength"] = args.chain_strength
                if args.annealing_time is not None:
                    kw["annealing_time"] = args.annealing_time
                if args.spin_reversals > 0:
                    kw["num_spin_reversal_transforms"] = args.spin_reversals
                try:
                    resp = sampler.sample_qubo(qubo.dict, **kw)
                except TypeError:
                    kw.pop("num_spin_reversal_transforms", None)
                    resp = sampler.sample_qubo(qubo.dict, **kw)
                sample = next(iter(resp.lowest().samples()))
                cbf = getattr(resp.record, "chain_break_fraction", None)
                row["chain_break_frac"] = round(float(cbf.mean()), 4) if cbf is not None else None
                qpu_t = resp.info.get("timing", {}).get("qpu_access_time", 0) / 1e6
                row["qpu_time_s"] = round(qpu_t, 6)
                total_qpu += qpu_t
                print(f"    max_chain={row['max_chain']} qubits={row['phys_qubits']} "
                      f"chain_breaks={row['chain_break_frac']} qpu={qpu_t*1000:.1f}ms",
                      flush=True)

            m = decode_and_score(sample, problem, int_to_id, depot, coarsener, inst)
            row["feasible"] = bool(m["is_feasible"])
            row["total_distance"] = round(float(m["total_distance"]), 2)
            print(f"    feasible={row['feasible']} dist={row['total_distance']}", flush=True)
            rows.append(row)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = list(rows[0].keys())
        for r in rows:
            for k in r:
                keys.append(k) if k not in keys else None
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} rows)")
    if not args.dry_run:
        print(f"Total QPU access time: {total_qpu:.3f}s (budget {args.max_qpu_seconds}s)")


if __name__ == "__main__":
    main()
