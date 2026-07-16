"""Diagnose Phase-4 power: how consistent is GNN < heuristic across instances?"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    df = pd.read_csv(ROOT / "results" / "phase4_full.csv")
    df["base"] = df["instance"].str.replace(r"\[:\d+\]", "", regex=True)
    fe = df[df.is_feasible].copy()

    # per (base instance, N): mean gap across feasible seeds
    g = (fe.groupby(["base", "N", "condition"])["opt_gap"].mean()
         .unstack("condition").dropna(subset=["gnn", "heuristic"]))
    print(f"per (instance,N): n={len(g)}")
    print(f"  GNN lower on {(g.gnn < g.heuristic).sum()}/{len(g)} pairs")
    _, p = wilcoxon(g.heuristic, g.gnn, alternative="greater")
    print(f"  Wilcoxon p={p:.3e}")

    # per base instance: mean gap pooled over N and seeds
    gi = (fe.groupby(["base", "condition"])["opt_gap"].mean()
          .unstack("condition").dropna(subset=["gnn", "heuristic"]))
    print(f"\nper base instance (pool N): n={len(gi)}")
    print(f"  GNN lower on {(gi.gnn < gi.heuristic).sum()}/{len(gi)} instances")
    _, p = wilcoxon(gi.heuristic, gi.gnn, alternative="greater")
    print(f"  Wilcoxon p={p:.3e}")

    # by N: is the effect concentrated at larger N?
    print("\nper (instance,N) split by N:")
    for N in sorted(fe.N.unique()):
        s = g.xs(N, level="N")
        _, p = (wilcoxon(s.heuristic, s.gnn, alternative="greater")
                if len(s) > 3 and not (s.heuristic.values == s.gnn.values).all()
                else (0, float("nan")))
        print(f"  N={N}: n={len(s)}  GNN lower {int((s.gnn<s.heuristic).sum())}/{len(s)}"
              f"  mean gnn={s.gnn.mean()*100:.1f}% heur={s.heuristic.mean()*100:.1f}%  p={p:.3f}")


if __name__ == "__main__":
    main()
