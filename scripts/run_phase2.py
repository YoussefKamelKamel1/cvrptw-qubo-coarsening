"""Phase-2 driver: static vs adaptive penalty calibration ablation.

Usage:
    python scripts/run_phase2.py [configs/phase2.yaml]

Shows, at equal solver budget, that adaptive calibration (objective-scaled
penalties + capacity-aware slack) reduces pre-repair constraint violations and
lifts feasibility vs the static penalty hierarchy, with a paired Wilcoxon test.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.solvers import SolverConfig          # noqa: E402
from src.experiment import run_ablation       # noqa: E402
from src.utils import load_config             # noqa: E402


def main(config_path: str) -> int:
    cfg_dict = load_config(config_path)
    solver = cfg_dict.get("solver", {})
    base_cfg = SolverConfig(
        backend=solver.get("backend", "sa"),
        num_reads=int(solver.get("num_reads", 200)),
        num_sweeps=int(solver.get("num_sweeps", 200)),
    )
    instances, Ns, seeds = cfg_dict["instances"], cfg_dict["Ns"], cfg_dict["seeds"]
    n = len(instances) * len(Ns) * len(seeds) * 4 * 2
    print(f"Phase-2 ablation: {len(instances)} inst x {len(Ns)} N x {len(seeds)} seeds "
          f"x 4 cond x 2 modes = {n} solves "
          f"(budget reads={base_cfg.num_reads}, sweeps={base_cfg.num_sweeps})",
          flush=True)

    df = run_ablation(instances, Ns, seeds, cfg=base_cfg)
    out = ROOT / cfg_dict.get("output", "results/phase2_ablation.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows -> {out}\n", flush=True)
    _summary(df)
    return 0


def _summary(df) -> None:
    import pandas as pd
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)

    for N in sorted(df["N"].unique()):
        sub = df[df["N"] == N]
        print("=" * 72)
        print(f"N={N}: static vs adaptive at equal budget")
        print("=" * 72)
        agg = (sub.groupby("penalty_mode")
               .agg(pre_violations=("pre_total_viol", "mean"),
                    pre_tw_viol=("pre_tw_viol", "mean"),
                    pre_feasible_pct=("pre_feasible", lambda s: 100 * s.mean()),
                    post_feasible_pct=("is_feasible", lambda s: 100 * s.mean()),
                    valid_pct=("is_valid", lambda s: 100 * s.mean()),
                    qubo_vars=("num_qubo_vars", "mean"))
               .round(2))
        print(agg.to_string())

        print(f"\nPost-repair feasibility % by family group (N={N}):")
        piv = (sub.groupby(["family_group", "penalty_mode"])["is_feasible"]
               .mean().mul(100).round(1).unstack("penalty_mode"))
        print(piv.to_string())
        print()

    _wilcoxon(df)


def _wilcoxon(df) -> None:
    """Paired Wilcoxon signed-rank test on pre-repair violations (static vs adaptive)."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("scipy not available; skipping Wilcoxon test")
        return
    keys = ["instance", "N", "seed", "condition", "solver"]
    piv = df.pivot_table(index=keys, columns="penalty_mode", values="pre_total_viol")
    piv = piv.dropna()
    if not {"static", "adaptive"}.issubset(piv.columns) or len(piv) == 0:
        return
    s, a = piv["static"].to_numpy(), piv["adaptive"].to_numpy()
    print("=" * 72)
    print("PAIRED WILCOXON — pre-repair violations (static vs adaptive)")
    print("=" * 72)
    print(f"  paired samples: {len(s)}")
    print(f"  mean static={s.mean():.2f}  mean adaptive={a.mean():.2f}  "
          f"mean reduction={s.mean()-a.mean():.2f}")
    if (s == a).all():
        print("  all pairs equal — test not applicable")
        return
    try:
        stat, p = wilcoxon(s, a, alternative="greater")  # H1: static > adaptive
        print(f"  Wilcoxon statistic={stat:.1f}  p-value={p:.2e}  "
              f"(H1: static has MORE violations than adaptive)")
    except ValueError as e:
        print(f"  Wilcoxon could not run: {e}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "configs" / "phase2.yaml")
    raise SystemExit(main(config))
