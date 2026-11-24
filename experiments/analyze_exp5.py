"""Set partitioning with reference routes in the pool: exact optimum against simulated
annealing, by penalty weight.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "results" / "exp5_setpartition_seeded.csv")
d["exact_gap"] = 100 * (d.exact_cost - d.ort_cost) / d.ort_cost
d["sa_gap"] = 100 * (d.sa_cost - d.ort_cost) / d.ort_cost
d["sa_hits_exact"] = np.isclose(d.sa_cost, d.exact_cost, rtol=1e-6)
d["exact_beats_ort"] = d.exact_cost < d.ort_cost - 1e-6
d["sa_beats_ort"] = d.sa_cost < d.ort_cost - 1e-6
d["sa_matches_ort"] = d.sa_cost <= d.ort_cost + 1e-6
d["arm_scale"] = d.arm + " @" + d.penalty_scale.astype(str)

order = ["A_random @10.0", "B_+ortools @10.0", "C_+neighbourhood @10.0",
         "C_+neighbourhood @2.0", "C_+neighbourhood @1.1"]
t = d.groupby(["N", "arm_scale"]).agg(
    pool=("pool", "mean"), exact_gap=("exact_gap", "mean"),
    exact_beats=("exact_beats_ort", "mean"), sa_gap=("sa_gap", "mean"),
    sa_hits_exact=("sa_hits_exact", "mean"), sa_le_ort=("sa_matches_ort", "mean"),
    cover=("sa_cover_frac", "mean"), sa_time=("sa_time", "mean"))
for c in ("exact_beats", "sa_hits_exact", "sa_le_ort", "cover"):
    t[c] *= 100
print(t.reindex(order, level="arm_scale").round(1).to_string())

print("\nall sizes pooled:")
p = d.groupby("arm_scale").agg(exact_gap=("exact_gap", "mean"),
                               exact_beats=("exact_beats_ort", "mean"),
                               sa_gap=("sa_gap", "mean"),
                               sa_hits_exact=("sa_hits_exact", "mean"),
                               sa_le_ort=("sa_matches_ort", "mean"),
                               sa_beats_ort=("sa_beats_ort", "mean"),
                               cases=("instance", "size"))
for c in ("exact_beats", "sa_hits_exact", "sa_le_ort", "sa_beats_ort"):
    p[c] *= 100
print(p.reindex(order).round(1).to_string())

print("\nfor reference, the position-encoding pipeline gap at the same N (exp4, published setup):")
e4 = pd.read_csv(HERE / "results" / "exp4_tight_term.csv")
e4 = e4[(e4.variant == "submitted") & (e4.N <= 20)]
print((e4.groupby(["condition", "N"]).opt_gap.mean() * 100).round(1).unstack("N").to_string())
