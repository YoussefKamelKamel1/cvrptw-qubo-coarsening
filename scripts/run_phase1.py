"""Phase-1 driver: reproduce the heuristic-coarsening CVRPTW baseline.

Usage:
    python scripts/run_phase1.py [configs/phase1.yaml]

Writes a per-run CSV and prints the reproduction summary:
feasibility % and mean distance by family group x condition x solver, plus the
validity check and the coarsening variable/runtime reduction.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.qubo import PenaltyWeights          # noqa: E402
from src.solvers import SolverConfig         # noqa: E402
from src.experiment import run_phase1        # noqa: E402
from src.utils import load_config            # noqa: E402


def main(config_path: str) -> int:
    cfg_dict = load_config(config_path)
    solver = cfg_dict.get("solver", {})
    base_cfg = SolverConfig(
        backend=solver.get("backend", "sa"),
        num_reads=int(solver.get("num_reads", 1000)),
        num_sweeps=int(solver.get("num_sweeps", 1000)),
    )
    penalties = PenaltyWeights(**cfg_dict.get("penalties", {})) if cfg_dict.get("penalties") else None

    instances = cfg_dict["instances"]
    Ns = cfg_dict["Ns"]
    seeds = cfg_dict["seeds"]
    n_runs = len(instances) * len(Ns) * len(seeds) * 4
    print(f"Phase-1: {len(instances)} instances x {len(Ns)} Ns x {len(seeds)} seeds "
          f"x 4 conditions = {n_runs} solves")

    df = run_phase1(instances, Ns, seeds, cfg=base_cfg, penalties=penalties)

    out = ROOT / cfg_dict.get("output", "results/phase1_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows -> {out}")

    _summary(df)
    return 0


def _summary(df) -> None:
    import pandas as pd
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n" + "=" * 70)
    print("VALIDITY (should be ~100% across the board — quantum tier)")
    print("=" * 70)
    val = df.groupby(["N", "condition", "solver"])["is_valid"].mean().mul(100).round(1)
    print(val.to_string())

    print("\n" + "=" * 70)
    print("FEASIBILITY % by family group x condition (headline reproduction)")
    print("   expect: C ~100%, R degraded under coarsening, RC intermediate")
    print("=" * 70)
    for N in sorted(df["N"].unique()):
        sub = df[df["N"] == N]
        piv = (sub.groupby(["family_group", "condition"])["is_feasible"]
               .mean().mul(100).round(1).unstack("condition"))
        print(f"\n--- N={N} ---")
        print(piv.to_string())

    print("\n" + "=" * 70)
    print("MEAN DISTANCE (feasible-and-valid runs) and QUBO VARS / RUNTIME")
    print("=" * 70)
    for N in sorted(df["N"].unique()):
        sub = df[df["N"] == N]
        agg = (sub.groupby(["family_group", "condition"])
               .agg(mean_vars=("num_qubo_vars", "mean"),
                    mean_time=("computation_time", "mean"),
                    feas_pct=("is_feasible", lambda s: 100 * s.mean()))
               .round(2))
        print(f"\n--- N={N} ---")
        print(agg.to_string())

    # Coarsening variable-reduction ratio (proxy for the ~P^2 scaling claim)
    print("\n" + "=" * 70)
    print("COARSENING VARIABLE REDUCTION (heuristic / none), by family group, N=10")
    print("=" * 70)
    sub = df[(df["N"] == 10) & (df["solver"] == "FQS")]
    piv = sub.groupby(["family_group", "condition"])["num_qubo_vars"].mean().unstack("condition")
    if {"none", "heuristic"}.issubset(piv.columns):
        piv["reduction_ratio"] = (piv["heuristic"] / piv["none"]).round(3)
    print(piv.round(1).to_string())


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "configs" / "phase1.yaml")
    raise SystemExit(main(config))
