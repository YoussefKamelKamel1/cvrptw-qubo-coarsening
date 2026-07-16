"""Phase-3 label generation: per-instance ORACLE coarsening -> edge labels.

For each training instance (truncated to N) we sweep coarsening hyperparameters,
run the full Phase-2 pipeline (heuristic coarsen -> adaptive QUBO -> SA -> inflate)
for each, and pick the *oracle* setting: feasible with lowest cost (fallback:
fewest violations). The oracle's first-level merges become positive edge labels
on the level-0 graph; all other candidate pairs are negatives.

Labelling by DOWNSTREAM feasibility+cost (not by imitating one family's tuned
params) is what lets the GNN generalise across C/R/RC without per-family tuning.

Usage: python scripts/gen_labels.py --N 10 --out results/gnn_labels_N10.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.coarsening import SpatioTemporalGraphCoarsener, graph_from_instance  # noqa: E402
from src.datasets import load_solomon, train_test_split_instances            # noqa: E402
from src.gnn_coarsening import build_features                                 # noqa: E402
from src.metrics import calculate_route_metrics                              # noqa: E402
from src.qubo import adaptive_penalties, vrp_problem_from_graph              # noqa: E402
from src.solvers import SolverConfig, solve_vrp                             # noqa: E402
from src.experiment import _routes_to_strings                               # noqa: E402

# Hyperparameter sweep: the 3 distinct per-family tuned settings + variations,
# so every instance has at least one good setting available to become its oracle.
SWEEP = [
    {"alpha": 1.0, "beta": 1.0, "P": 0.5, "radiusCoeff": 2.0},   # C-type tuned
    {"alpha": 1.0, "beta": 0.6, "P": 0.4, "radiusCoeff": 0.5},   # R-type tuned
    {"alpha": 0.7, "beta": 0.8, "P": 0.7, "radiusCoeff": 2.0},   # RC-type tuned
    {"alpha": 1.0, "beta": 1.0, "P": 0.4, "radiusCoeff": 1.0},
    {"alpha": 0.5, "beta": 0.5, "P": 0.5, "radiusCoeff": 1.0},
    {"alpha": 1.0, "beta": 0.4, "P": 0.6, "radiusCoeff": 1.0},
    {"alpha": 0.8, "beta": 0.6, "P": 0.5, "radiusCoeff": 0.5},
    {"alpha": 1.0, "beta": 1.0, "P": 0.6, "radiusCoeff": 2.0},
]


def evaluate_setting(inst, params, cfg):
    """Coarsen with params, solve (adaptive QUBO), inflate; return metrics+layers."""
    G, depot = graph_from_instance(inst)
    coarsener = SpatioTemporalGraphCoarsener(graph=G, depot_id=depot, **params)
    coarsened, layers = coarsener.coarsen()
    problem, int_to_id = vrp_problem_from_graph(coarsened, depot, inst.capacity)
    pw = adaptive_penalties(problem)
    sol = solve_vrp(problem, "FQS", cfg, pw, smart_capacity=True)
    routes = _routes_to_strings(sol.solution, int_to_id, depot)
    routes = coarsener.inflate_route(routes)
    m = calculate_route_metrics(coarsener.graph, routes, depot, inst.capacity)
    return m, layers


def oracle_for_instance(inst, cfg):
    """Pick the oracle setting (feasible+min cost; fallback fewest violations)."""
    best = None
    for params in SWEEP:
        m, _layers = evaluate_setting(inst, params, cfg)
        feasible = m["is_feasible"]
        viol = m["capacity_violations"] + m["time_window_violations"]
        key = (0 if feasible else 1,
               m["total_distance"] if feasible else viol)
        if best is None or key < best[0]:
            best = (key, feasible, params, m)
    _, feasible, params, m = best
    return {"params": params, "feasible": bool(feasible),
            "cost": float(m["total_distance"])}


def make_examples(inst, oracle):
    """Build labelled examples from EVERY level of the oracle coarsening run.

    Re-runs the oracle coarsening with per-level snapshots; each snapshot
    (graph state + merges chosen at that level) yields one training graph. This
    multiplies data for free and includes super-node contexts that match how the
    GNN coarsener is applied at inference (recursively, per level).
    """
    G, depot = graph_from_instance(inst)
    coarsener = SpatioTemporalGraphCoarsener(graph=G, depot_id=depot, **oracle["params"])
    coarsener.coarsen(record_levels=True)
    examples = []
    for level, (gsnap, merges) in enumerate(coarsener.level_snapshots):
        fb = build_features(gsnap, depot)
        if fb.cand_edge_index.numel() == 0:
            continue
        labels = torch.tensor(
            [1.0 if frozenset((e.u_id, e.v_id)) in merges else 0.0
             for e in fb.cand_edges], dtype=torch.float32)
        examples.append({
            "instance": inst.name, "level": level,
            "node_feat": fb.node_feat, "mp_edge_index": fb.mp_edge_index,
            "cand_edge_index": fb.cand_edge_index, "cand_edge_feat": fb.cand_edge_feat,
            "labels": labels, "oracle_params": oracle["params"],
            "oracle_feasible": oracle["feasible"], "oracle_cost": oracle["cost"],
        })
    return examples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--num_reads", type=int, default=200)
    ap.add_argument("--num_sweeps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    train, test = train_test_split_instances()
    cfg = SolverConfig(backend="sa", num_reads=args.num_reads,
                       num_sweeps=args.num_sweeps, seed=args.seed)
    print(f"Train instances: {len(train)}  Test (held out): {len(test)}", flush=True)
    print(f"Test set: {test}", flush=True)

    examples = []
    n_feasible = 0
    for k, name in enumerate(train):
        inst = load_solomon(name).truncate(args.N)
        oracle = oracle_for_instance(inst, cfg)
        exs = make_examples(inst, oracle)
        examples.extend(exs)
        n_feasible += int(oracle["feasible"])
        pos = sum(int(e["labels"].sum().item()) for e in exs)
        tot = sum(int(e["labels"].numel()) for e in exs)
        print(f"  [{k+1:2d}/{len(train)}] {name:6s} oracle P={oracle['params']['P']} "
              f"feas={oracle['feasible']} cost={oracle['cost']:.1f} "
              f"levels={len(exs)} pos_edges={pos}/{tot}", flush=True)

    out = Path(args.out) if args.out else ROOT / "results" / f"gnn_labels_N{args.N}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"examples": examples, "N": args.N,
                "train_instances": train, "test_instances": test,
                "sweep": SWEEP}, out)
    print(f"\nWrote {len(examples)} labelled graphs ({n_feasible} with feasible "
          f"oracle) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
