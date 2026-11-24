"""Energy of the reference solution against the sampled solution, and the terms
responsible for the difference.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "results" / "exp3_energy_diagnosis.csv")

cases = d.drop_duplicates(["instance", "N"])
print("=== can the published encoding express the reference at all? ===")
print(cases.groupby("N").agg(cases=("instance", "size"),
                             representable=("repr_default", "mean"),
                             ref_routes=("ref_routes", "mean"),
                             longest=("longest", "mean"),
                             k_default=("k_default", "mean")).round(2).to_string())

e = d.dropna(subset=["E_ref"]).copy()
e["formulation"] = e.E_ref > e.E_sa
# a case is a formulation failure only if the reference beats NONE of the 5 SA seeds
per_case = e.groupby(["instance", "N"]).agg(
    E_ref=("E_ref", "first"), best_sa=("E_sa", "min"))
per_case["verdict"] = per_case.apply(
    lambda r: "formulation" if r.E_ref > r.best_sa else "search", axis=1)

print("\n=== verdict per instance-size (reference vs best of 5 SA seeds) ===")
v = per_case.reset_index().groupby("N").verdict.value_counts().unstack(fill_value=0)
print(v.to_string())
print(f"\noverall: {(per_case.verdict == 'formulation').sum()} formulation, "
      f"{(per_case.verdict == 'search').sum()} search, of {len(per_case)} cases")

terms = [c for c in e.columns if c.startswith("d_")]
f = e[e.formulation]
print("\n=== in formulation failures: mean energy difference by term (ref - SA) ===")
print("    positive = the term penalises the cheaper reference solution more")
summ = pd.DataFrame({
    "mean_delta": f[terms].mean(),
    "share_positive": (f[terms] > 1e-9).mean(),
}).sort_values("mean_delta", ascending=False)
print(summ.round(3).to_string())

print("\n=== counterfactual: drop one term, would the reference then win? ===")
for t in terms:
    flips = ((e.E_ref - e[t]) < e.E_sa).groupby([e.instance, e.N]).any()
    print(f"  without {t[2:]:9s}: reference wins in {flips.mean()*100:5.1f}% of cases")
