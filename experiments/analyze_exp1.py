"""Continuity factor 1/2 against 1: routes, gap and feasibility."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "results" / "exp1_vehicle_bias.csv")
ov = pd.read_csv(HERE / "results" / "ortools_vehicles.csv")
d = d.merge(ov, on=["instance", "N"], how="left") if "instance" in ov else d
d["instance_base"] = d.instance.str.replace(r"\[.*", "", regex=True)
ov = ov.rename(columns={"instance": "instance_base"})
d = d.drop(columns=[c for c in ("ort_cost", "ort_vehicles") if c in d]).merge(
    ov, on=["instance_base", "N"], how="left")

for cond in ("gnn", "none"):
    s = d[d.condition == cond]
    print(f"\n================ condition = {cond} ================")
    g = s.groupby(["N", "cont_factor"]).agg(
        feasible=("is_feasible", "mean"), pre_feasible=("pre_feasible", "mean"),
        pre_veh=("pre_vehicles", "mean"), post_veh=("post_vehicles", "mean"),
        ort_veh=("ort_vehicles", "mean"), gap=("opt_gap", "mean"))
    g[["feasible", "pre_feasible"]] *= 100
    g["gap"] *= 100
    print(g.round(2).to_string())

    # paired comparison on runs feasible under BOTH factors
    piv = s.pivot_table(index=["instance", "N", "seed"], columns="cont_factor",
                        values=["opt_gap", "post_vehicles", "pre_vehicles", "is_feasible"])
    both = piv[(piv[("is_feasible", 0.5)] == 1) & (piv[("is_feasible", 1.0)] == 1)]
    ga, gb = both[("opt_gap", 0.5)], both[("opt_gap", 1.0)]
    print(f"\n  paired, feasible under both: n={len(both)}")
    print(f"    gap   0.5={ga.mean()*100:.2f}%   1.0={gb.mean()*100:.2f}%   "
          f"better on {(gb < ga - 1e-9).sum()}, worse on {(gb > ga + 1e-9).sum()}, "
          f"tied {(np.isclose(ga, gb)).sum()}")
    if (ga != gb).any():
        print(f"    Wilcoxon (1.0 lower gap) p={wilcoxon(ga, gb, alternative='greater').pvalue:.2e}")
    for v in ("pre_vehicles", "post_vehicles"):
        a, b = both[(v, 0.5)], both[(v, 1.0)]
        print(f"    {v:13s} 0.5={a.mean():.2f}  1.0={b.mean():.2f}")
    fa = piv[("is_feasible", 0.5)].mean() * 100
    fb = piv[("is_feasible", 1.0)].mean() * 100
    print(f"    feasibility over all runs: 0.5={fa:.1f}%  1.0={fb:.1f}%")
