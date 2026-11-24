"""Per-instance statistics (Wilcoxon, bootstrap) of the traced runs in
experiments/results/<tag>/runs.csv.

  python experiments/analyze_review.py
"""
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TAGS = ["solomon_stored_rank_fixed", "solomon_sym_rank_fixed", "gh_stored_rank_asloaded",
        "gh_sym_rank_asloaded"]
STAGES = [("coarse_feasible", "before expansion"), ("final_feasible", "final (paper's metric)"),
          ("final_rep_feasible", "final + repair after expansion")]


def test(d, col, other):
    t = d.groupby(["instance", "arm"])[col].mean().unstack()
    x = t.gnn - t[other]
    try:
        p = wilcoxon(t.gnn, t[other], alternative="greater").pvalue
    except ValueError:
        p = float("nan")
    return (f"{100 * t.gnn.mean():.1f} vs {100 * t[other].mean():.1f} | "
            f"better {(x > 0).sum()} / tied {(x == 0).sum()} / worse {(x < 0).sum()} | p = {p:.4f}")


lines = ["# Review re-runs: summary", ""]
for tag in TAGS:
    f = RES / tag / "runs.csv"
    if not f.exists():
        lines += [f"## {tag}", "", "_not run yet_", ""]
        continue
    d = pd.read_csv(f)
    lines += [f"## {tag}", "", f"{len(d)} runs, {d.instance.nunique()} instances, "
              f"sizes {sorted(d.N.unique())}", ""]
    lines += ["| stage | GNN | heuristic-fixed | tuned |", "|---|---|---|---|"]
    for col, label in STAGES:
        r = d.groupby("arm")[col].mean().mul(100).round(1)
        lines.append(f"| {label} | {r.gnn} | {r.heuristic_fixed} | {r.heuristic} |")
    lost = (d.coarse_feasible & ~d.final_feasible).groupby(d.arm).sum()
    lines += ["", f"Lost at expansion (runs): {lost.to_dict()}", ""]
    for col, label in STAGES:
        lines.append(f"- {label}, GNN vs heuristic-fixed: {test(d, col, 'heuristic_fixed')}")
        lines.append(f"- {label}, GNN vs tuned: {test(d, col, 'heuristic')}")
    lines += ["", "Final feasibility (%) by size:", "",
              d.groupby(["N", "arm"]).final_feasible.mean().mul(100).round(1).unstack().to_markdown(), ""]
    ok = d.pivot_table(index=["instance", "N", "seed"], columns="arm",
                       values=["final_feasible", "final_distance"])
    both = ok["final_feasible"].gnn.astype(bool) & ok["final_feasible"].heuristic_fixed.astype(bool)
    rel = (ok["final_distance"].gnn[both] / ok["final_distance"].heuristic_fixed[both] - 1) * 100
    lines += [f"Distance where GNN and heuristic-fixed are both feasible: {both.sum()} runs, "
              f"mean GNN - fixed {rel.mean():+.1f}%", ""]
    if d.family.notna().any():
        lines += ["Final feasibility (%) by family:", "",
                  d.groupby(["family", "arm"]).final_feasible.mean().mul(100).round(1).unstack().to_markdown(), ""]

out = RES / "review_summary.md"
out.write_text("\n".join(lines))
print("\n".join(lines))
