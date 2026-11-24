"""Fleet size against the default fleet size of the encoding."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "results" / "exp1b_fleet_size.csv")
ov = pd.read_csv(HERE / "results" / "ortools_vehicles.csv").rename(
    columns={"instance": "base"})
d["base"] = d.instance.str.replace(r"\[.*", "", regex=True)
d = d.merge(ov, on=["base", "N"], how="left")
order = ["submitted", "fleet/3", "fleet/4", "fleet/4+cont1"]

for cond in ("gnn", "none"):
    s = d[d.condition == cond]
    print(f"\n================ condition = {cond} ================")
    t = s.groupby(["N", "variant"]).agg(
        feas=("is_feasible", "mean"), pre_feas=("pre_feasible", "mean"),
        veh=("pre_vehicles", "mean"), ort_veh=("ort_vehicles", "mean"),
        vars=("num_qubo_vars", "mean"), gap=("opt_gap", "mean"))
    t[["feas", "pre_feas", "gap"]] *= 100
    t = t.reindex(order, level="variant")
    print(t.round(1).to_string())

    base = s[s.variant == "submitted"].set_index(["instance", "N", "seed"])
    print("\n  paired vs published setup, runs feasible under both:")
    for v in order[1:]:
        o = s[s.variant == v].set_index(["instance", "N", "seed"])
        j = base.join(o, lsuffix="_a", rsuffix="_b")
        both = j[(j.is_feasible_a == 1) & (j.is_feasible_b == 1)]
        a, b = both.opt_gap_a, both.opt_gap_b
        p = wilcoxon(a, b, alternative="greater").pvalue if (a != b).any() else float("nan")
        print(f"    {v:14s} n={len(both):3d}  gap {a.mean()*100:6.1f}% -> {b.mean()*100:6.1f}%  "
              f"better {(b < a - 1e-9).sum():3d} worse {(b > a + 1e-9).sum():3d}  p={p:.1e}  "
              f"| feasibility {j.is_feasible_a.mean()*100:5.1f}% -> {j.is_feasible_b.mean()*100:5.1f}%")
