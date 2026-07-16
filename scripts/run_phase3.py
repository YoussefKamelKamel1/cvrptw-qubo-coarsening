"""Phase-3 evaluation: GNN-guided coarsening vs heuristic, on held-out instances.

Compares three coarsening conditions x {FQS, APS} on the SA simulator with the
Phase-2 adaptive QUBO, over the TEST instances the GNN never saw:

  * none       -- no coarsening
  * heuristic  -- spatio-temporal coarsening with PER-FAMILY tuned params
  * gnn        -- learned merge scorer with a SINGLE fixed (P, radiusCoeff) for
                  all families (the headline: no per-family tuning)

Headline question: does 'gnn' match/beat 'heuristic' feasibility -- especially
R-type -- without per-family tuning?

Usage: python scripts/run_phase3.py --model results/gnn_model_N10.pt --N 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.coarsening import SpatioTemporalGraphCoarsener, graph_from_instance  # noqa: E402
from src.datasets import (SOLOMON_FAMILY_HYPERPARAMS, load_solomon,             # noqa: E402
                          train_test_split_instances)
from src.experiment import solve_on_graph                                     # noqa: E402
from src.gnn_coarsening import GNNCoarsener, load_model                        # noqa: E402
from src.solvers import SolverConfig                                          # noqa: E402
from src.utils import set_global_seed                                         # noqa: E402

# Fixed coarsening params for the GNN condition -- identical for every family.
# Selected on TRAIN validation (rc=0.5 clearly best; P among ties chosen for
# maximal coarsening). No per-family tuning, no test-set leakage.
GNN_FIXED_P = 0.5
GNN_FIXED_RCOEFF = 0.5


def coarsen_ratio(coarsened, original_n) -> float:
    return len(coarsened.nodes) / max(1, original_n)


# Fixed heuristic params (no per-family tuning) -- the fair control for the GNN.
HEUR_FIXED = {"alpha": 1.0, "beta": 1.0, "P": GNN_FIXED_P, "radiusCoeff": GNN_FIXED_RCOEFF}


def run_condition(inst, condition, solver, cfg, model, device):
    graph, depot = graph_from_instance(inst)
    n0 = len(graph.nodes)
    coarsener = None
    if condition == "heuristic":  # per-family TUNED heuristic (published upper baseline)
        params = SOLOMON_FAMILY_HYPERPARAMS[inst.family]
        coarsener = SpatioTemporalGraphCoarsener(graph=graph, depot_id=depot, **params)
        graph, _ = coarsener.coarsen()
    elif condition == "heuristic_fixed":  # heuristic scorer, SAME fixed params as GNN
        coarsener = SpatioTemporalGraphCoarsener(graph=graph, depot_id=depot, **HEUR_FIXED)
        graph, _ = coarsener.coarsen()
    elif condition == "gnn":  # learned scorer, fixed params (no per-family tuning)
        coarsener = GNNCoarsener(graph=graph, P=GNN_FIXED_P, radiusCoeff=GNN_FIXED_RCOEFF,
                                 depot_id=depot, model=model, device=device)
        graph, _ = coarsener.coarsen()
    _, m, _ = solve_on_graph(graph, depot, inst.capacity, solver, cfg,
                             penalty_mode="adaptive", coarsener=coarsener)
    # coarsen ratio is measured on the (possibly coarsened) graph vs original
    return m, len(graph.nodes) / max(1, n0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="results/gnn_model_N10.pt")
    ap.add_argument("--Ns", type=int, nargs="+", default=[10, 15, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--solvers", type=str, nargs="+", default=["FQS"])
    ap.add_argument("--num_reads", type=int, default=200)
    ap.add_argument("--num_sweeps", type=int, default=200)
    ap.add_argument("--out", type=str, default="results/phase3_eval.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ROOT / args.model, device=device)
    _, test = train_test_split_instances()
    print(f"Held-out test instances ({len(test)}): {test}")
    print(f"GNN trained at N=10; evaluated (scaling) at Ns={args.Ns}, solvers={args.solvers}")
    print(f"GNN fixed params: P={GNN_FIXED_P}, radiusCoeff={GNN_FIXED_RCOEFF} "
          f"(same for all families); heuristic uses per-family tuned params.\n", flush=True)

    rows = []
    for N in args.Ns:
        for name in test:
            for seed in args.seeds:
                cfg = SolverConfig(backend="sa", num_reads=args.num_reads,
                                   num_sweeps=args.num_sweeps, seed=seed)
                set_global_seed(seed)
                inst = load_solomon(name).truncate(N)
                for solver in args.solvers:
                    for condition in ("none", "heuristic", "heuristic_fixed", "gnn"):
                        m, ratio = run_condition(inst, condition, solver, cfg, model, device)
                        rows.append({
                            "instance": inst.name, "family": inst.family,
                            "family_group": inst.family_group, "N": N, "seed": seed,
                            "solver": solver, "condition": condition,
                            "is_feasible": bool(m["is_feasible"]),
                            "is_valid": bool(m["is_valid"]),
                            "pre_feasible": bool(m["pre_feasible"]),
                            "pre_total_viol": int(m["pre_total_viol"]),
                            "total_distance": float(m["total_distance"]),
                            "num_vehicles": int(m["num_vehicles"]),
                            "coarsen_ratio": round(ratio, 3),
                            "num_qubo_vars": int(m["num_qubo_vars"]),
                            "computation_time": float(m["computation_time"]),
                        })
        print(f"  done N={N}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = ROOT / args.out
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}\n", flush=True)
    _summary(df)
    return 0


def _summary(df) -> None:
    import pandas as pd
    pd.set_option("display.width", 200)
    order = [c for c in ["none", "heuristic", "heuristic_fixed", "gnn"]
             if c in df["condition"].unique()]

    print("=" * 72)
    print("SCALING: feasibility % by N x condition  (heuristic=per-family tuned, gnn=fixed)")
    print("=" * 72)
    piv = (df.groupby(["N", "condition"])["is_feasible"].mean().mul(100).round(1)
           .unstack("condition")[order])
    print(piv.to_string())

    print("\n" + "=" * 72)
    print("PRE-REPAIR feasibility % by N x condition  (isolates raw SA/QUBO quality)")
    print("   — this is where coarsening should help: smaller QUBOs SA can solve raw")
    print("=" * 72)
    pre = (df.groupby(["N", "condition"])["pre_feasible"].mean().mul(100).round(1)
           .unstack("condition")[order])
    print(pre.to_string())
    print("\nMean pre-repair violations by N x condition (lower = better raw solution):")
    pv = df.groupby(["N", "condition"])["pre_total_viol"].mean().round(1).unstack("condition")[order]
    print(pv.to_string())

    print("\nMean QUBO variables by N x condition (coarsening shrinks the problem):")
    v = df.groupby(["N", "condition"])["num_qubo_vars"].mean().round(0).unstack("condition")[order]
    print(v.to_string())

    print("\nMean solve time (s) by N x condition:")
    t = df.groupby(["N", "condition"])["computation_time"].mean().round(2).unstack("condition")[order]
    print(t.to_string())

    print("\n" + "=" * 72)
    print("Feasibility % by family group x condition, per N")
    print("=" * 72)
    for N in sorted(df["N"].unique()):
        sub = df[df["N"] == N]
        piv2 = (sub.groupby(["family_group", "condition"])["is_feasible"]
                .mean().mul(100).round(1).unstack("condition")[order])
        print(f"\n--- N={N} ---")
        print(piv2.to_string())

    print("\nValidity % by N x condition (should stay ~100%):")
    val = df.groupby(["N", "condition"])["is_valid"].mean().mul(100).round(1).unstack("condition")[order]
    print(val.to_string())


if __name__ == "__main__":
    raise SystemExit(main())
