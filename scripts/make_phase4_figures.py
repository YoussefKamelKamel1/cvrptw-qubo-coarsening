"""Scaling figures -> figures/.

fig_scaling_vars : QUBO variable-count explosion vs N (log y), to N=100.
fig_feasibility_scaling : coarsened feasibility vs N, GNN vs tuned/fixed heuristic
                          (the GNN's significant, tuning-free advantage that grows
                          at large N as fixed tuning collapses).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
C = {"none": "#7f7f7f", "heuristic": "#1f77b4", "heuristic_fixed": "#2ca02c",
     "gnn": "#d62728"}
LAB = {"none": "no coarsening", "heuristic": "heuristic (per-family tuned)",
       "heuristic_fixed": "heuristic (fixed, untuned)", "gnn": "GNN (learned, ours)"}


def fig_vars():
    d = pd.read_csv(ROOT / "results" / "scaling_large_vars.csv").sort_values("N")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for c in ["none", "heuristic", "gnn"]:
        ax.plot(d.N, d[c], "o-", color=C[c], label=LAB[c])
    ax.set_yscale("log")
    ax.set_xlabel("customers $N$")
    ax.set_ylabel("QUBO variables (log)")
    ax.set_title("QUBO size vs. instance size")
    ax.legend(fontsize=9)
    ax.annotate("15,450 vars\n(intractable)", (100, 15450), xytext=(72, 15450),
                fontsize=8, color=C["none"], va="center")
    fig.savefig(OUT / "fig_scaling_vars.pdf")
    fig.savefig(OUT / "fig_scaling_vars.png", dpi=140)
    plt.close(fig)
    print("wrote fig_scaling_vars")


def fig_feas():
    frames = []
    for f in ["scorer_ablation.csv", "scaling_large_solve.csv", "scaling_100_solve.csv"]:
        p = ROOT / "results" / f
        if p.exists():
            frames.append(pd.read_csv(p)[["instance", "N", "seed", "condition", "is_feasible"]])
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["instance", "N", "seed", "condition"])
    agg = (df.groupby(["N", "condition"])["is_feasible"].mean().mul(100)
           .unstack("condition"))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for c in ["gnn", "heuristic", "heuristic_fixed"]:
        if c in agg:
            ax.plot(agg.index, agg[c], "o-", color=C[c], label=LAB[c])
    ax.set_xlabel("customers $N$")
    ax.set_ylabel("coarsened feasibility (%)")
    ax.set_title("Feasibility-preserving coarsening")
    ax.legend(fontsize=9)
    fig.savefig(OUT / "fig_feasibility_scaling.pdf")
    fig.savefig(OUT / "fig_feasibility_scaling.png", dpi=140)
    plt.close(fig)
    print("wrote fig_feasibility_scaling")
    print(agg.round(1).to_string())


if __name__ == "__main__":
    fig_vars()
    fig_feas()
