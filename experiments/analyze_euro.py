"""Summary of the EURO Meets NeurIPS 2022 runs: feasibility by size, QUBO size, and
the road-matrix re-check (Supplementary Table S4).

  python experiments/analyze_euro.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
R = HERE / "results" / "euro_stored_rank_asloaded"


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return np.quantile(m, [0.025, 0.975])


def per_instance(d, col):
    p = d.pivot_table(index="instance", columns="arm", values=col, aggfunc="mean")
    diff = (p["gnn"] - p["heuristic_fixed"]).values * 100
    nz = diff[diff != 0]
    pval = wilcoxon(nz).pvalue if len(nz) else 1.0
    lo, hi = boot_ci(diff)
    return dict(metric=col, gnn=p["gnn"].mean() * 100, fixed=p["heuristic_fixed"].mean() * 100,
                diff=diff.mean(), ci_lo=lo, ci_hi=hi, higher=(diff > 0).sum(),
                tied=(diff == 0).sum(), lower=(diff < 0).sum(), p_two_sided=pval)


def tightness(inst, N=100):
    """Mean customer-to-customer distance / mean window width, first N customers."""
    xy, e, l = inst.coords[1:N + 1], inst.ready_time[1:N + 1], inst.due_time[1:N + 1]
    D = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    off = ~np.eye(len(xy), dtype=bool)
    return D[off].mean() / (l - e).mean(), (l - e).mean() / inst.due_time[0]


def main():
    runs = pd.read_csv(R / "runs.csv")
    road = pd.read_csv(R / "road_check.csv")
    d = runs.merge(road, on=["instance", "N", "seed", "arm"])
    assert len(d) == len(runs) == len(road) == 2520

    cols = ["pre_feasible", "final_feasible", "final_rep_feasible",
            "final_road_feasible", "rep_road_feasible"]
    t = d.groupby(["N", "arm"])[cols].mean() * 100
    t["qubo"] = d.groupby(["N", "arm"]).num_qubo_vars.mean()
    pd.set_option("display.width", 200)
    print("By size (%; road = re-checked on the instance's road travel-time matrix)")
    print(t.round(1).to_string())
    allr = d.groupby("arm")[cols].mean() * 100
    allr["qubo"] = d.groupby("arm").num_qubo_vars.mean()
    print(allr.round(1).to_string())

    print("\nPer base instance (60), GNN minus heuristic-fixed, points")
    res = pd.DataFrame([per_instance(d, c) for c in cols])
    print(res.round(4).to_string(index=False))
    big = d[d.N >= 80]
    print("\nN=80,100 only")
    print(pd.DataFrame([per_instance(big, c) for c in cols]).round(4).to_string(index=False))

    ok = d[d.final_euclid_feasible & d.final_road_feasible == False]  # noqa: E712
    print(f"\nEuclid-feasible runs that fail on roads: {len(d[(d.final_euclid_feasible) & (~d.final_road_feasible)])}"
          f" of {int(d.final_euclid_feasible.sum())}; median late customers among them "
          f"{d[(d.final_euclid_feasible) & (~d.final_road_feasible)].final_road_late.median()}")
    fe = d[d.final_euclid_feasible]
    print(f"road / Euclid travel time of final routes (Euclid-feasible runs): "
          f"median {np.median(fe.final_road_cost / fe.final_euclid_cost):.3f}")

    from euro_data import euro_names, load_euro
    from src.datasets import load_solomon, train_test_split_instances
    from review_runner import gh_names, load_gh
    rows = [("EURO", n, *tightness(load_euro(n))) for n in euro_names()]
    rows += [("Solomon", n, *tightness(load_solomon(n))) for n in train_test_split_instances()[1]]
    rows += [("GH", n, *tightness(load_gh(n))) for n in gh_names()]
    tt = pd.DataFrame(rows, columns=["set", "instance", "dist_over_width", "width_over_horizon"])
    print("\nTime-window tightness, first 100 customers (median)")
    print(tt.groupby("set")[["dist_over_width", "width_over_horizon"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
