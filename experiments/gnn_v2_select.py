"""Choose each model's checkpoint by final feasibility on six held-out training
instances.

  python experiments/gnn_v2_select.py --tag r1
"""
import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from gnn_v2_labels import VAL                                     # noqa: E402

SIZES = (20, 40, 60)
SEEDS = (0, 1)


def main():
    import pandas as pd
    from rev_runner import run, job
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root = HERE / "results" / "v2"
    ckpts = sorted(root.glob(f"{a.tag}_seed*/ep*.pt"))
    jobs = []
    for ck in ckpts:
        for n in VAL:
            for N in SIZES:
                for s in SEEDS:
                    j = job("v2sel", n, N, s, "gnn", f"{ck.parent.name}/{ck.stem}",
                            dict(P=0.5, radiusCoeff=0.5))
                    j["model"] = str(ck)
                    jobs.append(j)
    jobs.sort(key=lambda j: j["model"])   # keeps one model per worker for longer
    print(f"{len(ckpts)} checkpoints, {len(jobs)} runs", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(a.workers, maxtasksperchild=40) as pool:
        recs = list(pool.imap(run, jobs, chunksize=4))
    d = pd.DataFrame(recs)
    d.to_csv(root / f"{a.tag}_select_runs.csv", index=False)
    d["run"] = d.cfg.str.split("/").str[0]
    d["ep"] = d.cfg.str.split("/").str[1]
    g = d.groupby(["run", "ep"]).agg(feas=("final_feasible", "mean"),
                                     vars=("num_qubo_vars", "mean")).reset_index()
    g["epn"] = g.ep.str[2:].astype(int)
    sel = {}
    for run_, sub in g.groupby("run"):
        best = sub.sort_values(["feas", "vars", "epn"], ascending=[False, True, False]).iloc[0]
        sel[run_] = dict(ckpt=f"v2/{run_}/{best.ep}.pt", val_feas=float(best.feas),
                         val_vars=float(best.vars))
    print(g.pivot(index="ep", columns="run", values="feas").round(3).to_string(), flush=True)
    (root / f"{a.tag}_selected.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1), f"\n{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
