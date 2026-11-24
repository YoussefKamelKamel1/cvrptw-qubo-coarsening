"""LaTeX rows for Supplementary Tables S1 to S3."""
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parent / "results"
ARMS = ["gnn", "heuristic", "heuristic_fixed"]


def pct(x):
    return f"${x:.1f}$"


def rows(df, key, cols):
    out = []
    for k, sub in df.groupby(key, sort=False):
        vals = []
        for c in cols:
            vals += [pct(sub.loc[sub.arm == a, c].mean() * 100) if c != "num_qubo_vars"
                     else f"${sub.loc[sub.arm == a, c].mean():,.0f}$".replace(",", "{,}")
                     for a in ARMS]
        out.append(f"{k} & " + " & ".join(vals) + r" \\")
    return out


def merged(runs, full):
    f = full[["instance", "N", "seed", "arm", "full_feasible"]]
    return runs.merge(f, on=["instance", "N", "seed", "arm"])


gh = merged(pd.read_csv(R / "gh_stored_rank_asloaded/runs.csv"),
            pd.read_csv(R / "gh_stored_rank_asloaded/fullcheck_repair.csv"))
so = merged(pd.read_csv(R / "solomon_stored_rank_fixed/runs.csv"),
            pd.read_csv(R / "solomon_stored_rank_fixed/fullcheck_repair.csv"))
cols = ["final_feasible", "full_feasible", "num_qubo_vars"]

print("% S1a GH by N")
gh = gh.sort_values("N")
print("\n".join(rows(gh, "N", cols)))
all_row = gh.assign(k="all")
print("\n".join(rows(all_row, "k", cols)))
print("% S1b GH by family")
print("\n".join(rows(gh.sort_values("family"), "family", cols)))
print("% S2 Solomon per base instance")
print("\n".join(rows(so.sort_values("instance"), "instance", ["final_feasible", "full_feasible"])))
print("% S3 Solomon by N")
print("\n".join(rows(so.sort_values("N"), "N", ["final_feasible", "full_feasible"])))
print("\n".join(rows(so.assign(k="all"), "k", ["final_feasible", "full_feasible"])))
print("% families", so.groupby("instance").family.first().to_dict())
