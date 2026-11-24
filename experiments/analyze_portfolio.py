"""Several fleet sizes per instance, keeping the cheapest feasible solution, against a
control with the same number of runs.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "results" / "exp1b_fleet_size.csv")
PORT = ["submitted", "fleet/3", "fleet/4"]


def best(rows):
    f = rows[rows.is_feasible == 1]
    if f.empty:
        return np.nan, 0.0
    r = f.loc[f.total_distance.idxmin()]
    return r.opt_gap, 1.0


for cond in ("gnn", "none"):
    s = d[d.condition == cond]
    recs = []
    for (inst, N), g in s.groupby(["instance", "N"]):
        sub = g[g.variant == "submitted"]
        pg, pf = [], []
        for seed in range(5):
            gap, ok = best(g[(g.seed == seed) & g.variant.isin(PORT)])
            pg.append(gap); pf.append(ok)
        cg, cf = [], []
        for trio in combinations(range(5), 3):
            gap, ok = best(sub[sub.seed.isin(trio)])
            cg.append(gap); cf.append(ok)
        single = sub[sub.is_feasible == 1].opt_gap.mean()
        recs.append(dict(instance=inst, N=N,
                         single_gap=single, single_feas=sub.is_feasible.mean(),
                         ctrl_gap=np.nanmean(cg), ctrl_feas=np.mean(cf),
                         port_gap=np.nanmean(pg), port_feas=np.mean(pf)))
    r = pd.DataFrame(recs)
    print(f"\n================ condition = {cond} ================")
    t = r.groupby("N")[["single_feas", "single_gap", "ctrl_feas", "ctrl_gap",
                        "port_feas", "port_gap"]].mean()
    for c in t.columns:
        t[c] *= 100
    print(t.round(1).to_string())
    both = r.dropna(subset=["ctrl_gap", "port_gap"])
    a, b = both.ctrl_gap, both.port_gap
    print(f"\n  portfolio vs budget-matched control, {len(both)} instance-size pairs:")
    print(f"    gap {a.mean()*100:.1f}% -> {b.mean()*100:.1f}%   "
          f"portfolio better on {(b < a - 1e-9).sum()}, worse on {(b > a + 1e-9).sum()}   "
          f"p={wilcoxon(a, b, alternative='greater').pvalue:.1e}")
    print(f"    feasibility {r.ctrl_feas.mean()*100:.1f}% -> {r.port_feas.mean()*100:.1f}%")
