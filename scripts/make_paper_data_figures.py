"""Baseline, penalty and N=10 feasibility figures -> figures/.

The penalty figures read the same CSV as the tables, so figures and tables agree.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datasets import load_solomon
from src.coarsening import graph_from_instance
from src.qubo import vrp_problem_from_graph, PenaltyWeights, adaptive_penalties
from src.solvers import _k_max_fqs

FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
CC = {"none": "#7f7f7f", "heuristic": "#1f77b4", "heuristic_fixed": "#2ca02c", "gnn": "#d62728"}
CL = {"none": "no coarsening", "heuristic": "heuristic (tuned)",
      "heuristic_fixed": "heuristic (fixed)", "gnn": "GNN (ours)"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=140)
    plt.close(fig)
    print("wrote", name)


def baseline():
    d = pd.read_csv(ROOT / "results" / "phase1_results.csv")
    d = d[d.N == 10]
    feas = d.groupby(["family_group", "condition"]).is_feasible.mean().mul(100).unstack("condition")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    feas[["none", "heuristic"]].plot.bar(ax=ax, color=["#7f7f7f", "#1f77b4"])
    ax.set_ylabel("feasibility (%)"); ax.set_ylim(0, 105); ax.set_xlabel("Solomon family")
    ax.legend(["no coarsening", "heuristic coarsening"], fontsize=8)
    save(fig, "fig_phase1_feasibility")

    v = d.groupby(["family_group", "condition"]).num_qubo_vars.mean().unstack("condition")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    v[["none", "heuristic"]].plot.bar(ax=ax, color=["#7f7f7f", "#1f77b4"])
    ax.set_ylabel("mean QUBO variables"); ax.set_xlabel("Solomon family")
    ax.legend(["no coarsening", "heuristic coarsening"], fontsize=8)
    save(fig, "fig_phase1_qubovars")


def penalties():
    d = pd.read_csv(ROOT / "results" / "phase2_full.csv")
    d = d[d.N == 10]
    pv = d.groupby(["family_group", "penalty_mode"]).pre_total_viol.mean().unstack("penalty_mode")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    pv[["static", "adaptive"]].plot.bar(ax=ax, color=["#ff7f0e", "#1f77b4"])
    ax.set_ylabel("mean pre-repair violations"); ax.set_xlabel("Solomon family"); ax.legend(fontsize=9)
    # adaptive lands near zero on a 0-35 axis; label values so the bars are not
    # mistaken for missing data
    ax.set_ylim(0, pv[["static", "adaptive"]].to_numpy().max() * 1.42)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=7.5, padding=1)
    save(fig, "fig_phase2_prerepair")

    fe = d.groupby(["family_group", "penalty_mode"]).is_feasible.mean().mul(100).unstack("penalty_mode")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    fe[["static", "adaptive"]].plot.bar(ax=ax, color=["#ff7f0e", "#1f77b4"])
    ax.set_ylabel("post-repair feasibility (%)"); ax.set_ylim(0, 105)
    ax.set_xlabel("Solomon family"); ax.legend(fontsize=9)
    save(fig, "fig_phase2_feasibility")
    print("  phase2 check: pre-viol static=%.1f adaptive=%.1f" %
          (d[d.penalty_mode == "static"].pre_total_viol.mean(),
           d[d.penalty_mode == "adaptive"].pre_total_viol.mean()))

    insts = ["C101", "R101", "RC101", "R201"]
    stat, adap = [], []
    for name in insts:
        inst = load_solomon(name).truncate(10)
        g, dpt = graph_from_instance(inst)
        prob, _ = vrp_problem_from_graph(g, dpt, inst.capacity)
        nv = len(prob.capacities); kl = [_k_max_fqs(len(prob.dests), nv)] * nv

        def dr(pw, sc):
            q = prob.get_qubo(kl, pw, smart_capacity=sc).dict
            a = np.array([abs(x) for x in q.values() if x != 0]); return a.max() / a.min()
        stat.append(dr(PenaltyWeights(), False)); adap.append(dr(adaptive_penalties(prob), True))
    x = np.arange(len(insts)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(x - w/2, stat, w, label="static", color="#ff7f0e")
    ax.bar(x + w/2, adap, w, label="adaptive", color="#1f77b4")
    ax.set_yscale("log"); ax.set_ylabel("QUBO dynamic range (max/min)")
    ax.set_xticks(x); ax.set_xticklabels(insts); ax.legend(fontsize=9)
    save(fig, "fig_phase2_dynrange")


def n10_feasibility():
    d = pd.read_csv(ROOT / "results" / "phase3_eval_multiN.csv")
    order = ["none", "heuristic", "heuristic_fixed", "gnn"]
    n10 = d[d.N == 10]
    ft = (n10.groupby(["family_group", "condition"]).is_feasible.mean().mul(100)
          .unstack("condition")[order])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ft.plot.bar(ax=ax, color=[CC[c] for c in order])
    ax.set_ylabel("feasibility (%)"); ax.set_ylim(0, 108); ax.set_xlabel("Solomon family")
    ax.legend([CL[c] for c in order], fontsize=7)
    save(fig, "fig_phase3_feasibility")
    print("  N=10 R-type feasibility:", ft.loc["R"].round(0).to_dict() if "R" in ft.index else "n/a")


if __name__ == "__main__":
    baseline()
    penalties()
    n10_feasibility()
