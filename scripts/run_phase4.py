"""Phase-4 evaluation: reward-trained GNN vs tuned heuristic, with OR-Tools gap.

Compares {none, heuristic (per-family tuned), gnn (reward-trained, fixed params)}
on held-out test instances at multiple N, reporting BOTH feasibility (pre- and
post-repair) AND solution-cost quality as an **optimality gap vs OR-Tools**
(the classical reference). Answers: does the reward-trained GNN now clearly
beat the tuned heuristic, and how far is the whole pipeline from optimal?

Usage: python scripts/run_phase4.py --model results/gnn_model_rl.pt --Ns 10 15 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.baselines import solve_cvrptw_ortools                                   # noqa: E402
from src.coarsening import SpatioTemporalGraphCoarsener, graph_from_instance     # noqa: E402
from src.datasets import (SOLOMON_FAMILY_HYPERPARAMS, load_solomon,              # noqa: E402
                          train_test_split_instances)
from src.experiment import solve_on_graph                                       # noqa: E402
from src.gnn_coarsening import GNNCoarsener, load_model                          # noqa: E402
from src.solvers import SolverConfig                                            # noqa: E402
from src.utils import set_global_seed                                           # noqa: E402

GNN_P, GNN_RC = 0.5, 0.5


def run_condition(inst, condition, cfg, model, device):
    graph, depot = graph_from_instance(inst)
    coarsener = None
    if condition == "heuristic":
        coarsener = SpatioTemporalGraphCoarsener(
            graph=graph, depot_id=depot, **SOLOMON_FAMILY_HYPERPARAMS[inst.family])
        graph, _ = coarsener.coarsen()
    elif condition == "gnn":
        coarsener = GNNCoarsener(graph=graph, P=GNN_P, radiusCoeff=GNN_RC,
                                 depot_id=depot, model=model, device=device)
        graph, _ = coarsener.coarsen()
    _, m, _ = solve_on_graph(graph, depot, inst.capacity, "FQS", cfg,
                             penalty_mode="adaptive", coarsener=coarsener)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="results/gnn_model_rl.pt")
    ap.add_argument("--Ns", type=int, nargs="+", default=[10, 15, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--num_reads", type=int, default=200)
    ap.add_argument("--num_sweeps", type=int, default=200)
    ap.add_argument("--ortools_time", type=float, default=3.0)
    ap.add_argument("--out", type=str, default="results/phase4_eval.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ROOT / args.model, device=device)
    _, test = train_test_split_instances()
    print(f"Phase 4: reward-trained GNN vs tuned heuristic on {len(test)} held-out "
          f"instances, Ns={args.Ns}\n", flush=True)

    rows = []
    for N in args.Ns:
        for name in test:
            inst = load_solomon(name).truncate(N)
            ref = solve_cvrptw_ortools(inst, time_limit_s=args.ortools_time)
            ref_cost = ref.cost if ref.feasible else None
            for seed in args.seeds:
                set_global_seed(seed)
                cfg = SolverConfig("sa", args.num_reads, args.num_sweeps, seed=seed)
                for condition in ("none", "heuristic", "gnn"):
                    m = run_condition(inst, condition, cfg, model, device)
                    gap = (((m["total_distance"] - ref_cost) / ref_cost)
                           if (m["is_feasible"] and ref_cost) else float("nan"))
                    rows.append({
                        "instance": inst.name, "family_group": inst.family_group,
                        "N": N, "seed": seed, "condition": condition,
                        "is_feasible": bool(m["is_feasible"]),
                        "pre_feasible": bool(m["pre_feasible"]),
                        "pre_total_viol": int(m["pre_total_viol"]),
                        "total_distance": float(m["total_distance"]),
                        "ortools_cost": ref_cost, "opt_gap": gap,
                        "num_qubo_vars": int(m["num_qubo_vars"]),
                    })
        print(f"  done N={N}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = ROOT / args.out
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows -> {out}\n", flush=True)
    _summary(df)
    return 0


def _summary(df) -> None:
    import pandas as pd
    pd.set_option("display.width", 200)
    order = ["none", "heuristic", "gnn"]

    print("=" * 72)
    print("POST-REPAIR feasibility % by N x condition")
    print("=" * 72)
    print(df.groupby(["N", "condition"])["is_feasible"].mean().mul(100).round(1)
          .unstack("condition")[order].to_string())

    print("\nPRE-REPAIR feasibility % by N x condition (raw QUBO quality):")
    print(df.groupby(["N", "condition"])["pre_feasible"].mean().mul(100).round(1)
          .unstack("condition")[order].to_string())

    print("\n" + "=" * 72)
    print("OPTIMALITY GAP vs OR-Tools (mean over FEASIBLE runs; lower=better)")
    print("=" * 72)
    fe = df[df["is_feasible"]]
    print(fe.groupby(["N", "condition"])["opt_gap"].mean().mul(100).round(1)
          .unstack("condition")[order].to_string())

    print("\nN=10 feasibility % by family x condition:")
    n10 = df[df["N"] == 10]
    print(n10.groupby(["family_group", "condition"])["is_feasible"].mean().mul(100)
          .round(1).unstack("condition")[order].to_string())

    print("\nGNN vs heuristic - post-repair feasibility delta (gnn minus heuristic), by N:")
    piv = df.groupby(["N", "condition"])["is_feasible"].mean().mul(100).unstack("condition")
    for N in piv.index:
        d = piv.loc[N, "gnn"] - piv.loc[N, "heuristic"]
        print(f"  N={N}: {d:+.1f} pp  (gnn {piv.loc[N,'gnn']:.1f} vs heuristic {piv.loc[N,'heuristic']:.1f})")


if __name__ == "__main__":
    raise SystemExit(main())
