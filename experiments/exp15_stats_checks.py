"""Statistics recomputed from the stored result files."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
d = pd.concat([pd.read_csv(R / f) for f in
               ["scorer_ablation.csv", "scaling_large_solve.csv", "scaling_100_solve.csv"]])
d["base"] = d.instance.str.split("[").str[0]
print("rows", len(d), "sizes", sorted(d.N.unique()))
print("\n== feasibility % by arm (published grid) ==")
print(d.groupby("condition").is_feasible.mean().mul(100).round(1).to_dict())
print(d.groupby(["N", "condition"]).is_feasible.mean().mul(100).round(1).unstack().to_string())

cell = d.groupby(["base", "N", "condition"]).is_feasible.mean().unstack()


def counts(a, b):
    x = cell[a] - cell[b]
    return int((x > 0).sum()), int((x == 0).sum()), int((x < 0).sum())


print("\n== GNN vs heuristic_fixed ==")
w, t, l = counts("gnn", "heuristic_fixed")
print(f"pairs: {w} wins {t} ties {l} losses; sign test over pairs p={binomtest(w, w + l, alternative='greater').pvalue:.3f}")
inst = cell.groupby("base")[["gnn", "heuristic_fixed", "heuristic"]].mean()
x = inst.gnn - inst.heuristic_fixed
print(f"one value per base instance (13): better {(x > 0).sum()} tied {(x == 0).sum()} worse {(x < 0).sum()}; "
      f"Wilcoxon one-sided p={wilcoxon(inst.gnn, inst.heuristic_fixed, alternative='greater').pvalue:.3f}, "
      f"two-sided p={wilcoxon(inst.gnn, inst.heuristic_fixed).pvalue:.3f}")
print("per size, base instances, two-sided Wilcoxon (median over seeds):")
med = d.groupby(["base", "N", "condition"]).is_feasible.median().unstack()
for N in sorted(d.N.unique()):
    s = med.xs(N, level="N")
    for b in ["heuristic_fixed", "heuristic"]:
        dd = s.gnn - s[b]
        nz = dd[dd != 0]
        p = wilcoxon(s.gnn, s[b]).pvalue if len(nz) else float("nan")
        print(f"  N={N:3d} vs {b:15s}: better {(dd > 0).sum()} worse {(dd < 0).sum()} p={p:.3f}")

print("\n== GNN vs tuned heuristic, one value per base instance ==")
x = inst.gnn - inst.heuristic
print(f"better {(x > 0).sum()} tied {(x == 0).sum()} worse {(x < 0).sum()}; one-sided p="
      f"{wilcoxon(inst.gnn, inst.heuristic, alternative='greater').pvalue:.3f}")

print("\n== N=10: instances where arms differ ==")
n10 = d[d.N == 10].groupby(["base", "condition"]).is_feasible.sum().unstack()
print(n10[(n10.nunique(axis=1) > 1)].to_string())

print("\n== hash seed: set inside the process ==")
code = ("import sys; sys.path.insert(0, r'%s'); from src.utils import set_global_seed;"
        "a=hash('SN_12_7'); set_global_seed(0); print(a==hash('SN_12_7'), a)") % ROOT
outs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.split()[-2:]
        for _ in range(3)]
print("unchanged after seeding, hash in 3 fresh processes:", outs)
