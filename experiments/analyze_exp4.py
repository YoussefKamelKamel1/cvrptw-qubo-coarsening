"""Effect of the graded time-window term on cost and feasibility."""
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "results" / "exp4_tight_term.csv")
order = ["submitted", "tight=0.1", "tight=0", "tight=0+fleet/4"]

for cond in ("gnn", "none"):
    s = d[d.condition == cond]
    print(f"\n================ condition = {cond} ================")
    t = s.groupby(["N", "variant"]).agg(
        feas=("is_feasible", "mean"), pre_feas=("pre_feasible", "mean"),
        pre_viol=("pre_total_viol", "mean"), veh=("pre_vehicles", "mean"),
        gap=("opt_gap", "mean"))
    t[["feas", "pre_feas", "gap"]] *= 100
    print(t.reindex(order, level="variant").round(2).to_string())

    base = s[s.variant == "submitted"].set_index(["instance", "N", "seed"])
    print("\n  paired vs published setup (cost on runs feasible in both; feasibility on all runs):")
    for v in order[1:]:
        o = s[s.variant == v].set_index(["instance", "N", "seed"])
        j = base.join(o, lsuffix="_a", rsuffix="_b")
        both = j[(j.is_feasible_a == 1) & (j.is_feasible_b == 1)]
        a, b = both.opt_gap_a, both.opt_gap_b
        p = wilcoxon(a, b, alternative="greater").pvalue if (a != b).any() else float("nan")
        print(f"    {v:16s} n={len(both):3d}  gap {a.mean()*100:6.1f}% -> {b.mean()*100:6.1f}%  "
              f"better {(b < a - 1e-9).sum():3d} worse {(b > a + 1e-9).sum():3d}  p={p:.1e}  "
              f"| feasibility {j.is_feasible_a.mean()*100:5.1f}% -> "
              f"{j.is_feasible_b.mean()*100:5.1f}%  "
              f"| pre-repair {j.pre_feasible_a.mean()*100:5.1f}% -> "
              f"{j.pre_feasible_b.mean()*100:5.1f}%")
