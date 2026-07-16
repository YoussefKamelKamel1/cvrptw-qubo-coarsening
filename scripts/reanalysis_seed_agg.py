"""seed-aggregated (instance-level) re-analysis.

Multiple seeds on the same benchmark instance are not independent problem
instances, so treating each seed as an independent Wilcoxon observation is
pseudo-replication. Here we aggregate to one value per instance first (median
across seeds), then run the paired Wilcoxon across independent instances. This
is the conservative, defensible version of the Phase-2 and Phase-4 tests.

Reads the existing full-width CSVs; no new SA runs.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    # ---------------- Phase 2: static vs adaptive, pre-repair violations -----
    p2 = pd.read_csv(ROOT / "results" / "phase2_full.csv")
    print("=== PHASE 2 (pre-repair violations, static vs adaptive) ===")

    # original per-(instance,seed,condition) pairing
    keys = ["instance", "N", "seed", "condition"]
    piv = p2.pivot_table(index=keys, columns="penalty_mode",
                         values="pre_total_viol").dropna()
    s, a = piv["static"].to_numpy(), piv["adaptive"].to_numpy()
    _, p_raw = wilcoxon(s, a, alternative="greater")
    print(f"  per-seed  n={len(piv):4d}  static={s.mean():5.2f} adaptive={a.mean():5.2f}"
          f"  p={p_raw:.2e}")

    # aggregated to one value per INSTANCE (median across seeds AND conditions)
    agg = (p2.groupby(["instance", "penalty_mode"])["pre_total_viol"]
           .median().unstack("penalty_mode").dropna())
    s, a = agg["static"].to_numpy(), agg["adaptive"].to_numpy()
    _, p_inst = wilcoxon(s, a, alternative="greater")
    print(f"  by-instance n={len(agg):4d}  static={s.mean():5.2f} adaptive={a.mean():5.2f}"
          f"  p={p_inst:.2e}")

    # aggregated per (instance, condition) - keeps the two coarsening arms distinct
    agg2 = (p2.groupby(["instance", "condition", "penalty_mode"])["pre_total_viol"]
            .median().unstack("penalty_mode").dropna())
    s, a = agg2["static"].to_numpy(), agg2["adaptive"].to_numpy()
    _, p_ic = wilcoxon(s, a, alternative="greater")
    print(f"  by-inst*cond n={len(agg2):4d} static={s.mean():5.2f} adaptive={a.mean():5.2f}"
          f"  p={p_ic:.2e}")

    # ---------------- Phase 4: GNN vs heuristic, optimality gap --------------
    p4 = pd.read_csv(ROOT / "results" / "phase4_full.csv")
    fe = p4[p4.is_feasible].copy()
    print("\n=== PHASE 4 (gap to OR-Tools, GNN vs heuristic) ===")

    # original per-(instance,N,seed) pairing over feasible-in-both
    piv = fe.pivot_table(index=["instance", "N", "seed"], columns="condition",
                         values="opt_gap").dropna(subset=["gnn", "heuristic"])
    g, h = piv["gnn"].to_numpy(), piv["heuristic"].to_numpy()
    _, p_raw = wilcoxon(h, g, alternative="greater")
    print(f"  per-seed    n={len(piv):4d}  gnn={g.mean()*100:5.1f}% heur={h.mean()*100:5.1f}%"
          f"  p={p_raw:.2e}")

    # aggregate to one value per (instance, N): mean gap across feasible seeds
    agg = (fe.groupby(["instance", "N", "condition"])["opt_gap"].mean()
           .unstack("condition").dropna(subset=["gnn", "heuristic"]))
    g, h = agg["gnn"].to_numpy(), agg["heuristic"].to_numpy()
    _, p_iN = wilcoxon(h, g, alternative="greater")
    print(f"  by-inst*N   n={len(agg):4d}  gnn={g.mean()*100:5.1f}% heur={h.mean()*100:5.1f}%"
          f"  p={p_iN:.2e}")

    # aggregate to one value per instance (pool N)
    aggi = (fe.groupby(["instance", "condition"])["opt_gap"].mean()
            .unstack("condition").dropna(subset=["gnn", "heuristic"]))
    g, h = aggi["gnn"].to_numpy(), aggi["heuristic"].to_numpy()
    _, p_i = wilcoxon(h, g, alternative="greater")
    print(f"  by-instance n={len(aggi):4d}  gnn={g.mean()*100:5.1f}% heur={h.mean()*100:5.1f}%"
          f"  p={p_i:.2e}")


if __name__ == "__main__":
    main()
