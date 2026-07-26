"""Run coarsened CVRPTW QUBOs on a D-Wave annealer via Leap.

Experiments:
  1. GNN vs heuristic coarsening (--coarsen): does the tuning-free feasibility
     advantage transfer to hardware?
  2. conditioning arms (--arms static capcrush adaptive): static vs cap-crush
     isolates the dynamic-range effect at a FIXED logical-variable count; adaptive
     is the complete method.
  3. embeddability (--embedding-only): chain length and qubit count vs coarsening,
     measured from the solver graph with NO QPU sampling.

Reporting is over the full sample distribution (occurrence-weighted): pre-repair raw
feasibility and violations (the conditioning signal the strong repair would hide),
post-repair feasibility, best/mean feasible distance, best energy, occurrence-
weighted chain-break fraction, and an Ising-space precision diagnostic (the QUBO is
converted to Ising and scaled into the selected solver's h/J ranges). Rows are
checkpointed after every arm. --max-qpu-seconds is a cumulative pre-call guard
(a final arm can overshoot by one batch), not a strict hard cap.

Usage:
  python scripts/run_qpu.py --dry-run
  python scripts/run_qpu.py --list-solvers
  python scripts/run_qpu.py --solver <NAME> --embedding-only --coarsen none --N 40
  python scripts/run_qpu.py --solver <NAME> --arms static capcrush adaptive \
      --coarsen gnn --N 10 --instances R101 --num-reads 100 --max-qpu-seconds 5
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
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
from src.experiment import _routes_to_strings                             # noqa: E402
from src.metrics import calculate_route_metrics                           # noqa: E402
from src.utils import set_global_seed                                     # noqa: E402

FIELDS = [
    "timestamp", "commit", "solver", "topology", "instance", "N", "coarsen",
    "penalty_mode", "seed", "n_vars", "qubo_dynamic_range", "ising_scale",
    "max_h_scaled", "max_j_scaled", "n_J_below_floor", "n_h_below_floor",
    "embed_ok", "max_chain", "phys_qubits", "num_reads_req", "num_reads_ret",
    "gauges", "chain_strength", "annealing_time", "best_energy", "n_distinct",
    "raw_feas_frac", "mean_raw_viol", "min_raw_viol", "post_feas_frac",
    "best_dist", "mean_dist", "chain_break_frac", "qpu_access_time_s",
]
DEFAULT_H_RANGE = (-4.0, 4.0)   # Advantage-like; overridden by the real solver
DEFAULT_J_RANGE = (-1.0, 1.0)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# problem construction (static / cap-crush / adaptive)
# --------------------------------------------------------------------------- #
def build_problem(name, N, coarsen, penalty_mode, model, device):
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
    k_limits = [k] * nv

    if penalty_mode == "adaptive":
        pw, smart = adaptive_penalties(problem), True
    elif penalty_mode == "capcrush":
        # keep every capacity variable (fixed count) but crush the capacity penalty
        # so its (2^m)^2 couplings no longer dominate -- isolates conditioning.
        pw, smart = adaptive_penalties(problem), False
        Q = problem.capacities[0] if problem.capacities else 0
        if Q > 0:
            m_max = math.floor(math.log2(max(1.0, Q)))
            pw = replace(pw, capacity_penalty=pw.time_window_penalty / (2 ** m_max) ** 2)
    else:  # static
        pw, smart = PenaltyWeights(), False

    qubo = problem.get_qubo(k_limits, pw, smart_capacity=smart)
    return inst, depot, coarsener, problem, int_to_id, qubo, k_limits


# --------------------------------------------------------------------------- #
# Ising-space precision diagnostic
# --------------------------------------------------------------------------- #
def ising_stats(qubo, h_range, j_range, floor):
    """Convert the QUBO to Ising, scale into the device h/J ranges (as auto_scale
    does), and count h/J terms that fall below the assumed precision floor."""
    import dimod
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.dict)
    h, J, _ = bqm.to_ising()
    hv = [abs(x) for x in h.values() if x != 0]
    jv = [abs(x) for x in J.values() if x != 0]
    raw = [abs(x) for x in qubo.dict.values() if x != 0]
    qdr = (max(raw) / min(raw)) if raw else 1.0
    h_dev = max(abs(x) for x in h_range) if h_range else 4.0
    j_dev = max(abs(x) for x in j_range) if j_range else 1.0
    max_h, max_j = (max(hv) if hv else 0.0), (max(jv) if jv else 0.0)
    scale = max(max_h / h_dev if h_dev else 0.0, max_j / j_dev if j_dev else 0.0) or 1.0
    n_j = sum(1 for x in jv if x / scale < floor * j_dev)
    n_h = sum(1 for x in hv if x / scale < floor * h_dev)
    return dict(n_vars=bqm.num_variables, qubo_dynamic_range=round(qdr, 1),
                ising_scale=round(scale, 5), max_h_scaled=round(max_h / scale, 4),
                max_j_scaled=round(max_j / scale, 4),
                n_J_below_floor=n_j, n_h_below_floor=n_h)


def source_graph(qubo):
    import networkx as nx
    g = nx.Graph()
    for (u, v), c in qubo.dict.items():
        g.add_node(u)
        g.add_node(v)
        if u != v and c != 0:
            g.add_edge(u, v)
    return g


# --------------------------------------------------------------------------- #
# full-distribution evaluation (pre- and post-repair, occurrence-weighted)
# --------------------------------------------------------------------------- #
def evaluate_samples(sampleset, problem, coarsener, inst, int_to_id, depot, k_limits):
    if coarsener is not None:
        metrics_graph = coarsener.graph
    else:
        metrics_graph, _ = graph_from_instance(inst)
    agg = sampleset.aggregate()
    total = raw_feas = post_feas = 0
    n_distinct = 0
    best_energy = None
    raw_pairs, feas_dists = [], []
    for datum in agg.data(fields=["sample", "energy", "num_occurrences"], sorted_by="energy"):
        occ = int(datum.num_occurrences)
        n_distinct += 1
        total += occ
        best_energy = float(datum.energy) if best_energy is None else min(best_energy, float(datum.energy))
        sol = VRPSolution(problem, dict(datum.sample), k_limits)
        pre = sol.report(sol.raw_solution)
        raw_pairs.append((pre["total_violations"], occ))
        if pre["feasible"]:
            raw_feas += occ
        routes = _routes_to_strings(sol.solution, int_to_id, depot)
        if coarsener is not None:
            routes = coarsener.inflate_route(routes)
        m = calculate_route_metrics(metrics_graph, routes, depot, inst.capacity)
        if m["is_feasible"]:
            post_feas += occ
            feas_dists.append(float(m["total_distance"]))

    def wmean(pairs):
        den = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / den if den else float("nan")

    return dict(
        n_distinct=n_distinct, num_reads_ret=total,
        best_energy=round(best_energy, 3) if best_energy is not None else None,
        raw_feas_frac=round(raw_feas / total, 3) if total else 0.0,
        mean_raw_viol=round(wmean(raw_pairs), 2) if raw_pairs else None,
        min_raw_viol=min(v for v, _ in raw_pairs) if raw_pairs else None,
        post_feas_frac=round(post_feas / total, 3) if total else 0.0,
        best_dist=round(min(feas_dists), 2) if feas_dists else None,
        mean_dist=round(statistics.mean(feas_dists), 2) if feas_dists else None,
    )


def weighted_chain_break(resp) -> float | None:
    rec = getattr(resp, "record", None)
    if rec is None or "chain_break_fraction" not in rec.dtype.names:
        return None
    occ = rec.num_occurrences.astype(float)
    cbf = rec.chain_break_fraction.astype(float)
    return round(float((cbf * occ).sum() / occ.sum()), 4) if occ.sum() else None


# --------------------------------------------------------------------------- #
# device selection
# --------------------------------------------------------------------------- #
def list_solvers():
    from dwave.cloud import Client
    with Client.from_config() as client:
        print("Available QPU solvers (name | topology | qubits):")
        for s in client.get_solvers(qpu=True):
            p = s.properties
            print(f"  {s.id} | {p.get('topology', {}).get('type', '?')} | "
                  f"{p.get('num_qubits', len(p.get('qubits', [])))}")


def make_qpu(args):
    from dwave.system import DWaveSampler
    if args.solver:
        qpu = DWaveSampler(solver=args.solver)
    elif args.topology:
        qpu = DWaveSampler(solver=dict(topology__type=args.topology))
    else:
        raise SystemExit("Pass --solver <name> or --topology {pegasus,zephyr}; "
                         "use --list-solvers to see the options.")
    p = qpu.properties
    h_range = tuple(p.get("h_range", DEFAULT_H_RANGE))
    j_range = tuple(p.get("j_range", DEFAULT_J_RANGE))
    print("=" * 66)
    print(f"DEVICE   : {qpu.solver.id}  topology={p.get('topology', {}).get('type', '?')}")
    print(f"qubits   : {p.get('num_qubits', len(qpu.nodelist))}  couplers: {len(qpu.edgelist)}")
    print(f"h_range  : {h_range}   j_range: {j_range}")
    print("=" * 66, flush=True)
    return qpu, qpu.solver.id, p.get("topology", {}).get("type", "?"), h_range, j_range


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="results/gnn_model_rl.pt")
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--arms", nargs="+", default=["adaptive"],
                    choices=["static", "capcrush", "adaptive"],
                    help="use 'static capcrush adaptive' for the conditioning study")
    ap.add_argument("--coarsen", default="gnn", choices=["gnn", "heuristic", "none"])
    ap.add_argument("--instances", nargs="+", default=None)
    ap.add_argument("--num-reads", type=int, default=1000)
    ap.add_argument("--annealing-time", type=float, default=None, help="microseconds")
    ap.add_argument("--chain-strength", type=float, default=None)
    ap.add_argument("--spin-reversals", type=int, default=0,
                    help="gauge-averaging transforms via SpinReversalTransformComposite")
    ap.add_argument("--precision-floor", type=float, default=0.03,
                    help="assumed coupler-resolution floor for the Ising diagnostic")
    ap.add_argument("--max-qpu-seconds", type=float, default=20.0)
    ap.add_argument("--solver", default=None)
    ap.add_argument("--topology", default=None, choices=["pegasus", "zephyr"])
    ap.add_argument("--list-solvers", action="store_true")
    ap.add_argument("--embedding-only", action="store_true",
                    help="measure embedding chains/qubits only, no QPU sampling")
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
    h_range, j_range = DEFAULT_H_RANGE, DEFAULT_J_RANGE
    if not args.dry_run:
        from minorminer import find_embedding                     # noqa: F401
        from dwave.system import FixedEmbeddingComposite          # noqa: F401
        if "DWAVE_API_TOKEN" not in os.environ:
            raise SystemExit("Set DWAVE_API_TOKEN for a real QPU/embedding run.")
        qpu, solver_id, topology, h_range, j_range = make_qpu(args)
    elif args.embedding_only:
        raise SystemExit("--embedding-only needs a real solver (pass --solver).")

    arms = ["adaptive"] if args.embedding_only else args.arms
    meta = dict(timestamp=datetime.now().isoformat(timespec="seconds"),
                commit=git_commit(), solver=solver_id, topology=topology,
                num_reads_req=args.num_reads, gauges=args.spin_reversals,
                chain_strength=args.chain_strength, annealing_time=args.annealing_time,
                seed=args.seed)
    print(f"{'DRY-RUN' if args.dry_run else ('EMBED-ONLY' if args.embedding_only else 'QPU')} "
          f"| N={args.N} | coarsen={args.coarsen} | arms={arms} | {names}", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    total_qpu, stop = 0.0, False
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for name in names:
            if stop:
                break
            for mode in arms:
                inst, depot, coarsener, problem, int_to_id, qubo, k_limits = build_problem(
                    name, args.N, args.coarsen, mode, model, device)
                row = dict(meta, instance=name, N=args.N, coarsen=args.coarsen,
                           penalty_mode=mode, **ising_stats(qubo, h_range, j_range,
                                                            args.precision_floor))
                print(f"  {name} [{mode:8s}] vars={row['n_vars']} "
                      f"dyn_range={row['qubo_dynamic_range']} "
                      f"J<floor={row['n_J_below_floor']}", flush=True)

                if not args.dry_run:
                    from minorminer import find_embedding
                    emb = find_embedding(source_graph(qubo), qpu.edgelist, random_seed=args.seed)
                    row["embed_ok"] = bool(emb)
                    if emb:
                        row["max_chain"] = max(len(c) for c in emb.values())
                        row["phys_qubits"] = sum(len(c) for c in emb.values())
                    if args.embedding_only or not emb:
                        if not emb:
                            print("    embedding FAILED", flush=True)
                        else:
                            print(f"    max_chain={row['max_chain']} qubits={row['phys_qubits']}",
                                  flush=True)
                        writer.writerow(row)
                        f.flush()
                        continue
                    if total_qpu >= args.max_qpu_seconds:
                        print(f"  [budget] {total_qpu:.2f}s used >= {args.max_qpu_seconds}s; stop",
                              flush=True)
                        stop = True
                        break

                # ---- sample (full distribution) ----
                if args.dry_run:
                    from dwave.samplers import SimulatedAnnealingSampler
                    sampleset = SimulatedAnnealingSampler().sample_qubo(
                        qubo.dict, num_reads=args.num_reads, num_sweeps=200, seed=args.seed)
                else:
                    from dwave.system import FixedEmbeddingComposite
                    base = FixedEmbeddingComposite(qpu, emb)
                    kw = dict(num_reads=args.num_reads, auto_scale=True, return_embedding=True)
                    if args.chain_strength is not None:
                        kw["chain_strength"] = args.chain_strength
                    if args.annealing_time is not None:
                        kw["annealing_time"] = args.annealing_time
                    sampler = base
                    if args.spin_reversals > 0:
                        from dwave.system import SpinReversalTransformComposite
                        sampler = SpinReversalTransformComposite(base)
                        kw["num_spin_reversal_transforms"] = args.spin_reversals
                    sampleset = sampler.sample_qubo(qubo.dict, **kw)
                    row["chain_break_frac"] = weighted_chain_break(sampleset)
                    qpu_t = sampleset.info.get("timing", {}).get("qpu_access_time", 0) / 1e6
                    row["qpu_access_time_s"] = round(qpu_t, 6)
                    total_qpu += qpu_t

                row.update(evaluate_samples(sampleset, problem, coarsener, inst,
                                            int_to_id, depot, k_limits))
                print(f"    raw_feas={row['raw_feas_frac']} mean_raw_viol={row['mean_raw_viol']} "
                      f"post_feas={row['post_feas_frac']} best_dist={row['best_dist']}", flush=True)
                writer.writerow(row)
                f.flush()

    print(f"\nwrote {out}")
    if not args.dry_run and not args.embedding_only:
        print(f"cumulative QPU access time: {total_qpu:.3f}s (guard {args.max_qpu_seconds}s)")


if __name__ == "__main__":
    main()
