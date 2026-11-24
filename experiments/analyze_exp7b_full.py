"""Large-neighbourhood search with route selection against OR-Tools at equal time."""
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sm = pd.read_csv(HERE / "results" / "exp7b_full_summary.csv")
it = pd.read_csv(HERE / "results" / "exp7b_full_iterations.csv")

sm["lns_%"] = 100 * (sm.lns_exact - sm.start) / sm.start
sm["ort_%"] = 100 * (sm.ort_long - sm.start) / sm.start
sm["lns_minus_ort_%"] = 100 * (sm.lns_exact - sm.ort_long) / sm.ort_long
sm["family"] = sm.instance.str.extract(r"^([A-Z]+)")[0]

print(sm.sort_values(["instance", "seed"])[
    ["instance", "seed", "lns_%", "ort_%", "lns_minus_ort_%", "lns_wall"]].round(2)
    .to_string(index=False))

w = (sm.lns_exact < sm.ort_long - 1e-6).sum()
l = (sm.lns_exact > sm.ort_long + 1e-6).sum()
print(f"\nruns: {len(sm)}   LNS better: {w}   OR-Tools better: {l}   tied: {len(sm)-w-l}")
print(f"mean improvement over start: LNS {sm['lns_%'].mean():.2f}%   "
      f"OR-Tools same time {sm['ort_%'].mean():.2f}%")
print(f"LNS vs OR-Tools (same time): mean {sm['lns_minus_ort_%'].mean():+.2f}%")
d = sm.ort_long - sm.lns_exact
if (d != 0).any():
    print(f"paired Wilcoxon, LNS lower cost: p={wilcoxon(d, alternative='greater').pvalue:.3f}"
          f"   (two-sided p={wilcoxon(d).pvalue:.3f})")

print("\nby family (mean % vs start):")
print(sm.groupby("family")[["lns_%", "ort_%"]].mean().round(2).to_string())

it["sel_gain"] = 100 * (it.best_single - it.exact) / it.best_single
print(f"\nper iteration ({len(it)}): exact selection beats best single OR-Tools re-solve in "
      f"{(it.exact < it.best_single - 1e-9).mean()*100:.0f}%, mean gain {it.sel_gain.mean():.2f}%; "
      f"improves current routes in {(it.exact < it.current - 1e-9).mean()*100:.0f}%")
