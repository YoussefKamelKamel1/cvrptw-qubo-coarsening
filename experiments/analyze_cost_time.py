"""Route cost and running time of the Solomon runs by size, against OR-Tools and the
best-known solutions (Supplementary Table S5).

  python experiments/analyze_cost_time.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / "results"
BKS = {  # instance: (vehicles, distance)
    "C104": (10, 824.78), "C108": (10, 828.94), "C204": (3, 590.60), "C208": (3, 588.32),
    "R104": (9, 1007.31), "R108": (9, 960.88), "R112": (9, 982.14), "R204": (2, 825.52),
    "R208": (2, 726.82), "RC104": (10, 1135.48), "RC108": (10, 1139.82),
    "RC204": (3, 798.46), "RC208": (3, 828.14)}

runs = pd.read_csv(R / "solomon_stored_rank_fixed" / "runs.csv")
tr = pd.DataFrame([json.loads(l) for l in open(R / "solomon_stored_rank_fixed" / "traces.jsonl")])
tr = tr[["instance", "N", "seed", "arm", "time_coarsen_s"]]
d = runs.merge(tr, on=["instance", "N", "seed", "arm"])
ort = pd.read_csv(R / "ortools_ref_test.csv")
assert ort.feasible.all(), "OR-Tools infeasible somewhere"
d = d.merge(ort[["instance", "N", "cost", "vehicles", "time_s"]].rename(
    columns={"cost": "ref", "vehicles": "ref_veh", "time_s": "ref_time"}), on=["instance", "N"])
d["gap"] = (d.final_distance / d.ref - 1) * 100

pd.set_option("display.width", 220)
f = d[d.final_feasible]
t = f.groupby(["N", "arm"]).gap.mean().unstack().round(1)
t.columns = [f"gap_{c}" for c in t.columns]
t["n_feas"] = f.groupby(["N", "arm"]).size().unstack().astype(str).agg("/".join, axis=1)
t["time_coarsen"] = d.groupby("N").time_coarsen_s.median().round(2)
for a in ["gnn", "heuristic", "heuristic_fixed"]:
    t[f"time_{a}"] = d[d.arm == a].groupby("N").time_total_s.median().round(2)
t["ortools_time"] = ort.groupby("N").time_s.median().round(2)
t["veh_gnn"] = f[f.arm == "gnn"].groupby("N").final_vehicles.mean().round(1)
t["veh_ref"] = ort.groupby("N").vehicles.mean().round(1)
print("gap (%) among feasible runs, median time per run (s), vehicles")
print(t.to_string())
print("\nall sizes, gap:", f.groupby("arm").gap.mean().round(1).to_dict())
print("feasible runs:", f.groupby("arm").size().to_dict())

# N = 100: pipeline, OR-Tools and BKS
b = pd.DataFrame([(k, *v) for k, v in BKS.items()], columns=["instance", "bks_veh", "bks"])
o100 = ort[ort.N == 100].merge(b, on="instance")
o100["ort_vs_bks"] = (o100.cost / o100.bks - 1) * 100
print("\nN=100 OR-Tools (3 s) vs BKS: mean %+.1f%%, vehicles %.1f vs %.1f" %
      (o100.ort_vs_bks.mean(), o100.vehicles.mean(), o100.bks_veh.mean()))
print(o100[["instance", "cost", "vehicles", "bks", "bks_veh", "ort_vs_bks"]].round(2).to_string(index=False))
p100 = f[f.N == 100].merge(b, on="instance")
p100["vs_bks"] = (p100.final_distance / p100.bks - 1) * 100
print("\nN=100 pipeline (feasible runs) vs BKS:")
print(p100.groupby("arm").agg(runs=("vs_bks", "size"), vs_bks=("vs_bks", "mean"),
                              veh=("final_vehicles", "mean"), bks_veh=("bks_veh", "mean")).round(1))
