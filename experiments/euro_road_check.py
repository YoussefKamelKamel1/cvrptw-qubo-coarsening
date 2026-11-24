"""Re-check stored EURO solutions against the road travel-time matrix.

  python experiments/euro_road_check.py [tag]   (default euro_stored_rank_asloaded)
Writes experiments/results/<tag>/road_check.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from euro_data import _euclid, _parse, scale_factor  # noqa: E402


def check(routes, N, T, p):
    """Feasibility and travel time of int routes [[0, i, j, 0], ...] under times T."""
    e, l = p["tw"][:, 0], p["tw"][:, 1]
    seen, late, over, cost = [], 0, 0, 0.0
    for r in routes:
        r = [int(c) for c in r]
        if r[0] != 0:
            r = [0] + r
        if r[-1] != 0:
            r = r + [0]
        t, load = e[0], 0.0
        for a, b in zip(r[:-1], r[1:]):
            t += (p["svc"][a] if a != 0 else 0.0) + T[a, b]
            cost += T[a, b]
            if t > l[b] + 1e-6:
                late += 1
            t = max(t, e[b])
            if b != 0:
                seen.append(b)
                load += p["dem"][b]
        over += load > p["cap"] + 1e-9
    cover = sorted(seen) == list(range(1, N + 1))
    return dict(feasible=bool(cover and late == 0 and over == 0), late=late,
                overloaded=over, covered=cover, cost=cost)


def main(tag="euro_stored_rank_asloaded"):
    out = HERE / "results" / tag
    rows = []
    for ln in (out / "traces.jsonl").read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        p, N = _parse(r["instance"]), r["N"]
        E = _euclid(p["xy"]) * scale_factor(r["instance"])
        row = {k: r[k] for k in ["instance", "N", "seed", "arm"]}
        row["pipe_final_feasible"] = r["final"]["is_feasible"]
        row["pipe_rep_feasible"] = r["final_rep"]["is_feasible"]
        for stage, key in [("final", "final_routes"), ("rep", "final_rep_routes")]:
            for metric, T in [("euclid", E), ("road", p["T"])]:
                c = check(r[key], N, T, p)
                for k, v in c.items():
                    row[f"{stage}_{metric}_{k}"] = v
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(out / "road_check.csv", index=False)
    agree_f = (d.final_euclid_feasible == d.pipe_final_feasible).mean()
    agree_r = (d.rep_euclid_feasible == d.pipe_rep_feasible).mean()
    print(f"{len(d)} runs; checker agrees with pipeline: final {agree_f:.4f}, after repair {agree_r:.4f}")
    print(d.groupby("arm")[["final_euclid_feasible", "final_road_feasible",
                            "rep_euclid_feasible", "rep_road_feasible"]].mean().round(3))


if __name__ == "__main__":
    main(*sys.argv[1:])
