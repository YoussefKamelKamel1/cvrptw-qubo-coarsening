"""Gap with a coarsening that merges only consecutive customers of the reference routes."""
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "results" / "exp2b_oracle_ceiling.csv")
order = ["gnn/fleet2", "oracle/fleet2", "oracle/fleet4", "gnn/fleet4"]

t = d.groupby(["N", "arm"]).agg(feas=("is_feasible", "mean"),
                                pre_feas=("pre_feasible", "mean"),
                                ratio=("coarsen_ratio", "mean"),
                                vars=("num_qubo_vars", "mean"),
                                veh=("pre_vehicles", "mean"),
                                ref_veh=("ref_routes", "mean"),
                                gap=("opt_gap", "mean"))
t[["feas", "pre_feas", "gap"]] *= 100
print(t.reindex(order, level="arm").round(2).to_string())

base = d[d.arm == "gnn/fleet2"].set_index(["instance", "N", "seed"])
print("\npaired against the published GNN pipeline (runs feasible in both):")
for arm in order[1:]:
    o = d[d.arm == arm].set_index(["instance", "N", "seed"])
    j = base.join(o, lsuffix="_a", rsuffix="_b")
    both = j[(j.is_feasible_a == 1) & (j.is_feasible_b == 1)]
    a, b = both.opt_gap_a, both.opt_gap_b
    p = wilcoxon(a, b, alternative="greater").pvalue if (a != b).any() else float("nan")
    print(f"  {arm:14s} n={len(both):3d}  gap {a.mean()*100:5.1f}% -> {b.mean()*100:5.1f}%  "
          f"better {(b < a - 1e-9).sum():3d} worse {(b > a + 1e-9).sum():3d}  p={p:.1e}  "
          f"feasibility {j.is_feasible_a.mean()*100:5.1f}% -> {j.is_feasible_b.mean()*100:5.1f}%")
