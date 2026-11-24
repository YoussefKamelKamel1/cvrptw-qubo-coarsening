"""Pipeline runs for the additional comparisons of Section 5.4 and the robustness
section: re-tuned heuristic, five retrained models, label-free heuristic, other
trunks, ensemble and improved training.

  python experiments/rev_runner.py tune|test|sens|check|nolabel|retrain|arch|ens5|v2final [--workers 4]
Writes experiments/results/rev_<mode>/runs.jsonl and runs.csv.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

SIZES = [10, 20, 40, 50, 60, 80, 100]
REF = HERE / "results" / "solomon_stored_rank_fixed" / "runs.csv"
TUNE_GRID = [dict(alpha=1.0, beta=b, P=p, radiusCoeff=rc)
             for b, p, rc in itertools.product([0.3, 0.6, 1.0, 1.5], [0.4, 0.5, 0.7], [0.5, 1.0, 2.0])]
SENS_GRID = [dict(P=p, radiusCoeff=rc)
             for p, rc in itertools.product([0.4, 0.5, 0.6, 0.7], [0.5, 1.0, 2.0])]

_MODEL = None


def run(spec: dict) -> dict:
    global _MODEL
    import torch
    torch.set_num_threads(1)
    from review_runner import load, make_class, _repair_on_original, MODEL
    from src.coarsening import graph_from_instance, SpatioTemporalGraphCoarsener
    from src.experiment import solve_on_graph
    from src.metrics import calculate_route_metrics
    from src.solvers import SolverConfig
    from src.utils import set_global_seed

    t0 = time.perf_counter()
    name, N, seed, arm, prm = spec["instance"], spec["N"], spec["seed"], spec["arm"], spec["params"]
    set_global_seed(seed)
    inst = load(name, "solomon", "fixed").truncate(N)
    graph, depot = graph_from_instance(inst)
    if arm == "gnn":
        from src.gnn_coarsening import GNNCoarsener, load_model
        mpath = spec.get("model", str(MODEL))
        if _MODEL is None or _MODEL[0] != mpath:
            from gnn_ensemble import load_spec   # single checkpoint or 'ens:a,b,...'
            _MODEL = (mpath, load_spec(mpath, device="cpu"))
        co = make_class(GNNCoarsener, "stored", "rank")(
            graph, P=prm["P"], radiusCoeff=prm["radiusCoeff"], depot_id=depot,
            model=_MODEL[1], device="cpu")
    else:
        co = make_class(SpatioTemporalGraphCoarsener, "stored", "rank")(
            graph=graph, depot_id=depot, **prm)
    coarse, layers = co.coarsen()
    t_coarsen = time.perf_counter() - t0

    captured = []
    orig = co.inflate_route
    co.inflate_route = lambda r: (captured.append([list(x) for x in r]), orig(r))[1]
    routes, m, _ = solve_on_graph(coarse, depot, inst.capacity, "FQS",
                                  SolverConfig("sa", 200, 200, seed=seed),
                                  penalty_mode="adaptive", coarsener=co)
    mc = calculate_route_metrics(coarse, captured[0], depot, inst.capacity)
    mr = calculate_route_metrics(graph, _repair_on_original(graph, depot, inst.capacity, routes),
                                 depot, inst.capacity)
    return dict(key=spec["key"], set=spec["set"], instance=name, family=inst.family, N=N,
                seed=seed, arm=arm, cfg=spec["cfg"], **{f"prm_{k}": v for k, v in prm.items()},
                coarse_n=len(coarse.nodes), n_merges=len(layers),
                num_qubo_vars=int(m["num_qubo_vars"]),
                pre_feasible=bool(m["pre_feasible"]), pre_total_viol=int(m["pre_total_viol"]),
                pre_tw_viol=int(m["pre_tw_viol"]), pre_cap_viol=int(m["pre_cap_viol"]),
                coarse_feasible=bool(mc["is_feasible"]),
                final_feasible=bool(m["is_feasible"]), final_distance=float(m["total_distance"]),
                final_tw_viol=int(m["time_window_violations"]),
                final_cap_viol=int(m["capacity_violations"]),
                final_rep_feasible=bool(mr["is_feasible"]),
                time_coarsen_s=round(t_coarsen, 4), time_total_s=round(time.perf_counter() - t0, 3))


def job(set_, inst, N, seed, arm, cfg, params):
    return dict(key=f"{set_}|{inst}|{N}|{seed}|{arm}|{cfg}", set=set_, instance=inst, N=N,
                seed=seed, arm=arm, cfg=cfg, params=params)


def cfg_name(p: dict) -> str:
    return "_".join(f"{k[0] if k != 'radiusCoeff' else 'rc'}{v:g}" for k, v in p.items())


def select(tune_csv: Path) -> dict:
    """Best tune-set configuration per (N, family), with P free and with P = 0.5."""
    import pandas as pd
    d = pd.read_csv(tune_csv)
    g = (d.groupby(["N", "family", "cfg", "prm_beta", "prm_P", "prm_radiusCoeff"])
         .agg(feas=("final_feasible", "mean"), vars=("num_qubo_vars", "mean"),
              viol=("pre_total_viol", "mean"), n=("instance", "size")).reset_index())
    out = {}
    for sel, sub in [("free", g), ("P0.5", g[g.prm_P == 0.5])]:
        best = (sub.sort_values(["feas", "vars", "viol"], ascending=[False, True, True])
                .groupby(["N", "family"]).head(1))
        for r in best.itertuples():
            out[(sel, int(r.N), r.family)] = dict(alpha=1.0, beta=float(r.prm_beta), P=float(r.prm_P),
                                                  radiusCoeff=float(r.prm_radiusCoeff))
    return out


def build(mode: str) -> list:
    from review_runner import load
    from src.datasets import train_test_split_instances, family_from_name
    from src.runner_parallel import HEUR_FIXED
    train, test = train_test_split_instances()
    avail = lambda n: load(n, "solomon", "fixed").coords.shape[0] - 1
    jobs = []
    if mode == "tune":
        for N in sorted(SIZES, reverse=True):
            for n in train:
                if avail(n) >= N:
                    jobs += [job("tune", n, N, 0, "heuristic", cfg_name(p), p) for p in TUNE_GRID]
    elif mode == "test":
        best = select(HERE / "results" / "rev_tune" / "runs.csv")
        (HERE / "results" / "rev_tune" / "selected.json").write_text(json.dumps(
            {f"{s}|{N}|{f}": p for (s, N, f), p in best.items()}, indent=1))
        for N in sorted(SIZES, reverse=True):
            for n in test:
                if avail(n) < N:
                    continue
                fam = family_from_name(n)
                for sel in ["free", "P0.5"]:
                    p = best[(sel, N, fam)]
                    jobs += [job("test", n, N, s, "heuristic", f"retuned_{sel}", p) for s in (0, 1, 2)]
    elif mode == "sens":
        for N in sorted(SIZES, reverse=True):
            for n in test:
                if avail(n) < N:
                    continue
                for p in SENS_GRID:
                    for arm in ["gnn", "heuristic"]:
                        q = p if arm == "gnn" else {**HEUR_FIXED, **p}
                        jobs += [job("sens", n, N, s, arm, cfg_name(p), q) for s in (0, 1, 2)]
    elif mode == "nolabel":
        # one heuristic setting chosen on the training instances at N <= 20, all
        # families pooled (no family labels), P free
        p = dict(alpha=1.0, beta=0.3, P=0.7, radiusCoeff=1.0)
        for N in sorted(SIZES, reverse=True):
            for n in test:
                if avail(n) >= N:
                    jobs += [job("nolabel", n, N, s, "heuristic", "nolabel_free", p) for s in (0, 1, 2)]
    elif mode == "retrain":
        # the GNN with the retrained checkpoints (experiments/retrain_sage.py)
        for ts in (0, 1, 2, 3, 4):
            mp = str(HERE / "results" / "retrain" / f"sage_seed{ts}.pt")
            for N in sorted(SIZES, reverse=True):
                for n in test:
                    if avail(n) >= N:
                        for s in (0, 1, 2):
                            j = job("retrain", n, N, s, "gnn", f"train{ts}",
                                    dict(P=0.5, radiusCoeff=0.5))
                            j["model"] = mp
                            jobs.append(j)
    elif mode == "arch":
        # GAT and GCN trunks, five trainings each, at the sizes of Table 4a
        for arch in ("gat", "gcn"):
            for ts in (0, 1, 2, 3, 4):
                mp = str(HERE / "results" / "retrain" / f"{arch}_seed{ts}.pt")
                for N in (10, 20, 40):
                    for n in test:
                        for s in (0, 1, 2):
                            j = job("arch", n, N, s, "gnn", f"{arch}{ts}",
                                    dict(P=0.5, radiusCoeff=0.5))
                            j["model"] = mp
                            jobs.append(j)
    elif mode == "ens5":
        # ensemble of the five retrained GraphSAGE models (mean merge probability)
        mp = "ens:" + ",".join(str(HERE / "results" / "retrain" / f"sage_seed{t}.pt")
                               for t in range(5))
        for N in sorted(SIZES, reverse=True):
            for n in test:
                if avail(n) >= N:
                    for s in (0, 1, 2):
                        j = job("ens5", n, N, s, "gnn", "ens5", dict(P=0.5, radiusCoeff=0.5))
                        j["model"] = mp
                        jobs.append(j)
    elif mode == "v2final":
        # GNN v2: ensemble and each of the five round-2 models chosen by
        # gnn_v2_select.py, on the held-out test instances (protocol fixed in advance)
        sel = json.loads((HERE / "results" / "v2" / "r2_selected.json").read_text())
        specs = {"v2ens": "ens:" + ",".join(str(HERE / "results" / v["ckpt"]) for v in sel.values())}
        specs.update({f"v2_{k}": str(HERE / "results" / v["ckpt"]) for k, v in sel.items()})
        for cfg, mp in specs.items():
            for N in sorted(SIZES, reverse=True):
                for n in test:
                    if avail(n) >= N:
                        for s in (0, 1, 2):
                            j = job("v2final", n, N, s, "gnn", cfg, dict(P=0.5, radiusCoeff=0.5))
                            j["model"] = mp
                            jobs.append(j)
    elif mode == "check":
        for N in [10, 20, 40]:
            for n in test:
                jobs += [job("check", n, N, s, "heuristic", "fixed", dict(HEUR_FIXED)) for s in (0, 1, 2)]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "test", "sens", "check", "nolabel", "retrain", "arch", "ens5", "v2final"])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    out = HERE / "results" / f"rev_{a.mode}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = build(a.mode)
    f = out / "runs.jsonl"
    done = set()
    if f.exists():
        for ln in f.read_text().splitlines():
            try:
                done.add(json.loads(ln)["key"])
            except Exception:  # noqa: BLE001
                pass
    todo = [j for j in jobs if j["key"] not in done]
    print(f"rev_{a.mode}: {len(jobs)} jobs, {len(todo)} to run", flush=True)
    os.environ["PYTHONHASHSEED"] = "0"
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool, open(f, "a") as fh:
        for k, r in enumerate(pool.imap_unordered(run, todo, chunksize=1), 1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if k % 200 == 0:
                print(f"  [{k}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    import pandas as pd
    recs = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
    pd.DataFrame(recs).to_csv(out / "runs.csv", index=False)
    print(f"wrote {out} ({len(recs)} runs) in {time.time() - t0:.0f}s", flush=True)

    if a.mode == "check":
        ref = pd.read_csv(REF)
        ref = ref[ref.arm == "heuristic_fixed"].set_index(["instance", "N", "seed"])
        d = pd.DataFrame(recs).set_index(["instance", "N", "seed"])
        j = d.join(ref, rsuffix="_ref", how="inner")
        same = ((j.final_feasible == j.final_feasible_ref) & (j.num_qubo_vars == j.num_qubo_vars_ref)
                & ((j.final_distance - j.final_distance_ref).abs() < 1e-6))
        print(f"check: {int(same.sum())}/{len(j)} runs identical to the traced run")


if __name__ == "__main__":
    main()
