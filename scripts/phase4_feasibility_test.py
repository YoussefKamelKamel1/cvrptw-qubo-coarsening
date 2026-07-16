"""Is the GNN's coarsened-feasibility edge over the heuristic real (paired)?

Per instance, feasibility fraction across seeds for each condition; paired
Wilcoxon GNN vs tuned heuristic and GNN vs fixed heuristic, per N and pooled.
Uses scorer_ablation.csv (N=10/20/40, 13 test instances x 3 seeds).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    df = pd.read_csv(ROOT / "results" / "scorer_ablation.csv")
    Ns = sorted(df.N.unique())

    def test(sub, a, b, label):
        piv = (sub.groupby(["instance", "condition"])["is_feasible"].mean()
               .unstack("condition").dropna(subset=[a, b]))
        x, y = piv[a].to_numpy(), piv[b].to_numpy()
        if (x == y).all():
            print(f"  {label}: identical"); return
        _, p = wilcoxon(x, y, alternative="greater")
        print(f"  {label}: {a}={x.mean()*100:5.1f}% {b}={y.mean()*100:5.1f}%  "
              f"{a}>{b} on {int((x>y).sum())}/{len(x)}  =on {int((x==y).sum())}  p={p:.3f}")

    for N in Ns:
        print(f"-- N={N} --")
        s = df[df.N == N]
        test(s, "gnn", "heuristic", "GNN vs tuned")
        test(s, "gnn", "heuristic_fixed", "GNN vs fixed")
    print("-- pooled (all N) --")
    test(df, "gnn", "heuristic", "GNN vs tuned")
    test(df, "gnn", "heuristic_fixed", "GNN vs fixed")

    # also pool the earlier scaling_phase4 run for more seeds/power if compatible
    sp = ROOT / "results" / "scaling_phase4.csv"
    if sp.exists():
        d2 = pd.read_csv(sp)
        both = pd.concat([df[["instance", "N", "seed", "condition", "is_feasible"]],
                          d2[["instance", "N", "seed", "condition", "is_feasible"]]],
                         ignore_index=True).drop_duplicates(["instance", "N", "seed", "condition"])
        print("-- pooled with scaling_phase4 (dedup) --")
        test(both, "gnn", "heuristic", "GNN vs tuned")


if __name__ == "__main__":
    main()
