"""Three classical samplers on the same set-partitioning QUBOs."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "results" / "exp6_samplers.csv")
d["gap"] = 100 * (d.cost - d.ort_cost) / d.ort_cost
d["hit_opt"] = np.isclose(d.cost, d.exact_cost, rtol=1e-6)
d["within5"] = d.cost <= 1.05 * d.exact_cost
order = ["SA 500x1000", "SA 100x10000", "tabu 100x100ms"]

t = d.groupby(["N", "sampler"]).agg(gap=("gap", "mean"), hit_opt=("hit_opt", "mean"),
                                    within5=("within5", "mean"),
                                    cover=("cover_frac", "mean"), wall=("wall", "mean"))
for c in ("hit_opt", "within5", "cover"):
    t[c] *= 100
print(t.reindex(order, level="sampler").round(1).to_string())

print("\npooled over N:")
p = d.groupby("sampler").agg(gap=("gap", "mean"), hit_opt=("hit_opt", "mean"),
                             within5=("within5", "mean"), wall=("wall", "mean"))
p[["hit_opt", "within5"]] *= 100
print(p.reindex(order).round(1).to_string())

base = d[d.sampler == order[0]].set_index(["instance", "N", "seed"]).gap
print("\npaired vs the experiment-5 SA setting:")
for s in order[1:]:
    o = d[d.sampler == s].set_index(["instance", "N", "seed"]).gap
    j = pd.concat([base, o], axis=1, keys=["a", "b"]).dropna()
    p_ = wilcoxon(j.a, j.b, alternative="greater").pvalue if (j.a != j.b).any() else float("nan")
    print(f"  {s:15s} n={len(j)}  gap {j.a.mean():.1f}% -> {j.b.mean():.1f}%  "
          f"better {(j.b < j.a - 1e-9).sum()} worse {(j.b > j.a + 1e-9).sum()}  p={p_:.1e}")
