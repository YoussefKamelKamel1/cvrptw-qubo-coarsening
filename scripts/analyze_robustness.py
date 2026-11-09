"""Summarise the robustness runs: penalties at N=20,40, SA temperature range, and
leave-one-family-out.

  python scripts/analyze_robustness.py [penalty] [schedule] [lofo]
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

R = Path(__file__).resolve().parent.parent / "results"


def inst_test(df, key, a, b, metric="pre_total_viol", alt="greater"):
    """One value per base instance (median across seeds and conditions), paired a vs b."""
    d = df.copy()
    d["base"] = d.instance.str.split("[").str[0]
    piv = d.groupby(["base", key])[metric].median().unstack(key).dropna(subset=[a, b])
    if (piv[a] == piv[b]).all():
        return len(piv), float("nan")
    return len(piv), wilcoxon(piv[a], piv[b], alternative=alt).pvalue


def exp9():
    f = R / "penalty_larger_n.csv"
    if not f.exists():
        print("penalty_larger_n.csv missing"); return
    d = pd.read_csv(f)
    print("=== penalties at larger N (56 instances x 5 seeds x {none, heuristic}) ===")
    for N in sorted(d.N.unique()):
        s = d[d.N == N]
        g = s.groupby("penalty_mode").agg(pre_viol=("pre_total_viol", "mean"),
                                         pre_feas=("pre_feasible", "mean"),
                                         post_feas=("is_feasible", "mean"),
                                         vars=("num_qubo_vars", "mean"), runs=("seed", "size"))
        g[["pre_feas", "post_feas"]] *= 100
        print(f"-- N={N}\n{g.round(2).to_string()}")
        for m in ("adaptive", "capcrush"):
            n, p = inst_test(s, "penalty_mode", "static", m)
            print(f"   static > {m} (pre-repair violations, per instance): n={n} p={p:.2e}")
        print(s.groupby(["penalty_mode", "condition"]).num_qubo_vars.mean().round(1).unstack().to_string())


def exp10():
    f = R / "sa_schedule.csv"
    if not f.exists():
        print("sa_schedule.csv missing"); return
    d = pd.read_csv(f)
    print("\n=== SA temperature range (N=10, 56 instances x 3 seeds x {none, heuristic}) ===")
    g = d.groupby(["penalty_mode", "setting"]).agg(pre_viol=("pre_total_viol", "mean"),
                                                  pre_feas=("pre_feasible", "mean"),
                                                  post_feas=("is_feasible", "mean"),
                                                  runs=("seed", "size"))
    g[["pre_feas", "post_feas"]] *= 100
    print(g.round(2).to_string())
    st = d[d.penalty_mode == "static"]
    best = g.loc["static"].pre_viol.idxmin()
    ad = d[(d.penalty_mode == "adaptive") & (d.setting == "default")].assign(setting="adaptive_default")
    both = pd.concat([st[st.setting == best], ad])
    n, p = inst_test(both, "setting", best, "adaptive_default")
    print(f"best static schedule = {best}; still worse than adaptive default? n={n} p={p:.2e}")


def exp11():
    f = R / "lofo.csv"
    if not f.exists():
        print("lofo.csv missing"); return
    d = pd.read_csv(f)
    print("\n=== leave-one-family-out (13 test instances x N in {10,20,40} x 3 seeds) ===")
    g = d.groupby(["model_tag", "N"]).agg(feas=("is_feasible", "mean"),
                                          pre_feas=("pre_feasible", "mean"),
                                          gap=("opt_gap", "mean")).mul(100).round(1)
    print(g.unstack("N").to_string())
    print(d.groupby(["family", "model_tag"]).is_feasible.mean().mul(100).round(1).unstack().to_string())
    cell = d.groupby(["instance", "model_tag"]).is_feasible.mean().unstack()
    for other in ("control_all", "deployed"):
        x = cell["lofo"] - cell[other]
        print(f"lofo vs {other}: better {int((x > 0).sum())} tied {int((x == 0).sum())} "
              f"worse {int((x < 0).sum())} of {len(x)} instance-size pairs; "
              f"mean {cell['lofo'].mean()*100:.1f}% vs {cell[other].mean()*100:.1f}%")
    # compare with the heuristic runs on the same instances and sizes
    sa = pd.read_csv(R / "scorer_ablation.csv")
    h = sa[sa.condition.isin(["heuristic", "heuristic_fixed"])].groupby(
        ["instance", "condition"]).is_feasible.mean().unstack()
    j = cell.join(h, how="inner")
    print(f"same cells, mean feasibility: " +
          ", ".join(f"{c} {j[c].mean()*100:.1f}%" for c in j.columns))
    for c in ("heuristic", "heuristic_fixed"):
        x = j["lofo"] - j[c]
        print(f"lofo vs {c}: better {int((x > 0).sum())} tied {int((x == 0).sum())} worse {int((x < 0).sum())}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["penalty", "schedule", "lofo"]
    for w in which:
        {"penalty": exp9, "schedule": exp10, "lofo": exp11}[w]()
