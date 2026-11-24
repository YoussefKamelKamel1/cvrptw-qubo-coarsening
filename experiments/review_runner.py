"""Traced runs of the three coarseners (family-tuned heuristic, heuristic-fixed, GNN)
on Solomon, Gehring-Homberger and EURO instances.

  python experiments/review_runner.py --dataset solomon|gh|euro --scoring stored --tiebreak rank --c104 fixed

Per run it stores four measurements of the same solution: pre-repair, after repair
on the coarse graph, after expansion (final), and after one more repair on the
original instance. Outputs in experiments/results/<tag>/: manifest.json,
traces.jsonl and runs.csv. Interrupted runs resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

MODEL = ROOT / "results" / "gnn_model_rl.pt"
GH_DIR = HERE / "data_gh" / "extracted"
GH_ZIP = HERE / "data_gh" / "raw" / "homberger_200_customer_instances.zip"
ARMS = ["heuristic", "heuristic_fixed", "gnn"]
SIZES = [10, 20, 40, 50, 60, 80, 100]


# --------------------------------------------------------------------------- data
def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gh_names() -> list:
    return sorted(p.stem.upper() for p in GH_DIR.glob("*.TXT"))


def load_gh(name: str):
    import numpy as np
    from src.datasets import VRPInstance
    lines = (GH_DIR / f"{name}.TXT").read_text().splitlines()
    nv, cap = [float(x) for x in lines[4].split()]
    rows = [[float(x) for x in ln.split()] for ln in lines[9:] if len(ln.split()) == 7]
    a = np.array(rows)
    assert a[0, 0] == 0 and (np.diff(a[:, 0]) == 1).all(), name
    fam = name.split("_")[0]          # C1_2_1 -> C1
    return VRPInstance(name=name, coords=a[:, 1:3], demand=a[:, 3], capacity=cap,
                       n_vehicles=int(nv), ready_time=a[:, 4], due_time=a[:, 5],
                       service_time=a[:, 6], family=fam, meta={"source": "gehring-homberger 200"})


def load(name: str, dataset: str, c104: str):
    if dataset == "gh":
        return load_gh(name)
    if dataset == "euro":
        from euro_data import load_euro
        return load_euro(name)
    from src.datasets import load_solomon
    inst = load_solomon(name)
    if name == "C104" and c104 == "fixed":
        import numpy as np
        from dataclasses import replace
        ins = lambda arr, v: np.insert(arr, 36, v, axis=0)  # file numbers the depot 1
        inst = replace(inst, coords=ins(inst.coords, [5.0, 45.0]), demand=ins(inst.demand, 10.0),
                       ready_time=ins(inst.ready_time, 0.0), due_time=ins(inst.due_time, 1110.0),
                       service_time=ins(inst.service_time, 90.0))
    return inst


# ---------------------------------------------------------------- coarsener variants
def make_class(base, scoring: str, tiebreak: str):
    from src.coarsening import euclidean

    class Variant(base):
        def _reconnect(self, g, super_node, ni, nj):
            if tiebreak != "rank":
                return super()._reconnect(g, super_node, ni, nj)
            nbrs = (g.get_neighbors(ni.id) | g.get_neighbors(nj.id)) - {
                ni.id, nj.id, super_node.id, self.depot_id}
            for nb in sorted(nbrs):
                if nb in g.nodes:
                    g.add_edge(super_node.id, nb, euclidean(super_node, g.nodes[nb]))

    if scoring != "sym":
        return Variant

    if base.__name__ == "GNNCoarsener":
        class SymGNN(Variant):
            def score_edges(self, graph):
                super().score_edges(graph)
                first = {id(e): e.D_ij for e in graph.edges}
                for e in graph.edges:
                    e.u_id, e.v_id = e.v_id, e.u_id
                super().score_edges(graph)
                for e in graph.edges:
                    e.u_id, e.v_id = e.v_id, e.u_id
                    e.D_ij = 0.5 * (first[id(e)] + e.D_ij)
        return SymGNN

    class SymHeur(Variant):
        def merge_score(self, graph, edge):
            ni, nj = graph.nodes[edge.u_id], graph.nodes[edge.v_id]
            tau = euclidean(ni, nj)
            d_ij = self.alpha * tau + self.beta * max(0.0, nj.e - (ni.t + ni.s + tau))
            d_ji = self.alpha * tau + self.beta * max(0.0, ni.e - (nj.t + nj.s + tau))
            return min(d_ij, d_ji)
    return SymHeur


# ------------------------------------------------------------------------- one run
_MODEL = None


def _repair_on_original(graph, depot, capacity, routes):
    """One more repair of the expanded routes, on the original instance."""
    from src.qubo import vrp_problem_from_graph
    from src.solvers import VRPSolution
    nc = len(graph.nodes) - 1
    nv = max(2, nc // 2, len(routes))
    prob, int_to_id = vrp_problem_from_graph(graph, depot, capacity, num_vehicles=nv)
    id_to_int = {nid: i for i, nid in enumerate(int_to_id)}
    int_routes = [[id_to_int[c] for c in r if c != depot] for r in routes]
    sol = VRPSolution(prob, {}, None, solution=[])
    rep = sol._repair_time_windows(sol._repair_solution(int_routes))
    return [[depot] + [int_to_id[c] for c in r] + [depot] for r in rep if r]


def run(spec: dict) -> dict:
    global _MODEL
    import torch
    torch.set_num_threads(1)
    from src.coarsening import graph_from_instance, SpatioTemporalGraphCoarsener
    from src.datasets import SOLOMON_FAMILY_HYPERPARAMS
    from src.experiment import solve_on_graph
    from src.metrics import calculate_route_metrics
    from src.runner_parallel import HEUR_FIXED, GNN_FIXED_P, GNN_FIXED_RC
    from src.solvers import SolverConfig
    from src.utils import set_global_seed

    t0 = time.perf_counter()
    name, N, seed, arm = spec["instance"], spec["N"], spec["seed"], spec["arm"]
    set_global_seed(seed)
    inst = load(name, spec["dataset"], spec["c104"]).truncate(N)
    cfg = SolverConfig("sa", 200, 200, seed=seed)
    graph, depot = graph_from_instance(inst)
    if arm == "gnn":
        from src.gnn_coarsening import GNNCoarsener, load_model
        if _MODEL is None:
            _MODEL = load_model(str(MODEL), device="cpu")
        cls = make_class(GNNCoarsener, spec["scoring"], spec["tiebreak"])
        co = cls(graph, P=GNN_FIXED_P, radiusCoeff=GNN_FIXED_RC, depot_id=depot,
                 model=_MODEL, device="cpu")
        params = {"P": GNN_FIXED_P, "radiusCoeff": GNN_FIXED_RC}
    else:
        params = SOLOMON_FAMILY_HYPERPARAMS[inst.family] if arm == "heuristic" else HEUR_FIXED
        cls = make_class(SpatioTemporalGraphCoarsener, spec["scoring"], spec["tiebreak"])
        co = cls(graph=graph, depot_id=depot, **params)
    coarse, layers = co.coarsen()
    t_coarsen = time.perf_counter() - t0

    captured = []
    orig_inflate = co.inflate_route
    co.inflate_route = lambda r: (captured.append([list(x) for x in r]), orig_inflate(r))[1]
    routes_final, m, sol = solve_on_graph(coarse, depot, inst.capacity, "FQS", cfg,
                                          penalty_mode="adaptive", coarsener=co)
    coarse_routes = captured[0]
    mc = calculate_route_metrics(coarse, coarse_routes, depot, inst.capacity)
    routes_rep = _repair_on_original(graph, depot, inst.capacity, routes_final)
    mr = calculate_route_metrics(graph, routes_rep, depot, inst.capacity)

    keep = ["is_feasible", "is_valid", "total_distance", "num_vehicles",
            "time_window_violations", "capacity_violations", "num_customers_visited"]
    pick = lambda d: {k: (float(d[k]) if isinstance(d[k], float) else d[k]) for k in keep}
    return dict(
        key=spec["key"], instance=name, family=inst.family, N=N, seed=seed, arm=arm,
        dataset=spec["dataset"], scoring=spec["scoring"], tiebreak=spec["tiebreak"],
        c104=spec["c104"], coarsen_params=params, n_customers_loaded=int(inst.coords.shape[0] - 1),
        coarse_n=len(coarse.nodes), num_qubo_vars=int(m["num_qubo_vars"]),
        merges=[list(x) for x in layers],
        pre={"feasible": bool(m["pre_feasible"]), "total_viol": int(m["pre_total_viol"]),
             "tw_viol": int(m["pre_tw_viol"]), "cap_viol": int(m["pre_cap_viol"])},
        raw_routes=[[int(c) for c in r] for r in sol.raw_solution],
        coarse_routes=coarse_routes, coarse=pick(mc),
        final_routes=routes_final, final=pick(m),
        final_rep_routes=routes_rep, final_rep=pick(mr),
        time_coarsen_s=round(t_coarsen, 3), time_total_s=round(time.perf_counter() - t0, 3),
    )


def flat(r: dict) -> dict:
    out = {k: r[k] for k in ["instance", "family", "N", "seed", "arm", "dataset", "scoring",
                             "tiebreak", "c104", "n_customers_loaded", "coarse_n",
                             "num_qubo_vars", "time_total_s"]}
    out["n_merges"] = len(r["merges"])
    out["pre_feasible"], out["pre_total_viol"] = r["pre"]["feasible"], r["pre"]["total_viol"]
    for st in ["coarse", "final", "final_rep"]:
        out[f"{st}_feasible"] = r[st]["is_feasible"]
        out[f"{st}_distance"] = r[st]["total_distance"]
        out[f"{st}_vehicles"] = r[st]["num_vehicles"]
    return out


# ---------------------------------------------------------------------------- main
def manifest(args, tag, n_jobs):
    files = [Path(__file__), ROOT / "src" / "coarsening.py", ROOT / "src" / "gnn_coarsening.py",
             ROOT / "src" / "experiment.py", ROOT / "src" / "solvers.py", ROOT / "src" / "qubo.py",
             ROOT / "src" / "metrics.py", ROOT / "src" / "datasets.py",
             ROOT / "src" / "runner_parallel.py", MODEL]
    if args.dataset == "gh":
        files.append(GH_ZIP)
    if args.dataset == "euro":
        from euro_data import EURO_DIR, SAMPLE
        files += [HERE / "euro_data.py", SAMPLE] + sorted(EURO_DIR.glob("ORTEC*.txt"))
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    vers = {}
    for mod in ["numpy", "torch", "torch_geometric", "dwave.samplers", "ortools", "pandas"]:
        try:
            m = __import__(mod, fromlist=["x"])
            vers[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            vers[mod] = f"unavailable ({type(e).__name__})"
    return dict(tag=tag, command=" ".join([Path(sys.executable).name] + sys.argv), args=vars(args),
                n_jobs=n_jobs, git_commit=git("rev-parse", "HEAD"),
                git_tracked_changes=git("status", "--short", "--untracked-files=no"),
                file_sha256={str(f.relative_to(ROOT)): _sha(f) for f in files},
                python=sys.version, platform=platform.platform(), packages=vers,
                PYTHONHASHSEED_workers="0", started=time.strftime("%Y-%m-%dT%H:%M:%S"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="solomon", choices=["solomon", "gh", "euro"])
    ap.add_argument("--scoring", default="stored", choices=["stored", "sym"])
    ap.add_argument("--tiebreak", default="hash", choices=["hash", "rank"])
    ap.add_argument("--c104", default="asloaded", choices=["asloaded", "fixed"])
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    args = ap.parse_args()

    if args.dataset == "gh":
        names = args.instances or gh_names()
    elif args.dataset == "euro":
        from euro_data import euro_names
        names = args.instances or euro_names()
    else:
        from src.datasets import train_test_split_instances
        names = args.instances or train_test_split_instances()[1]
    tag = args.tag or f"{args.dataset}_{args.scoring}_{args.tiebreak}_{args.c104}"
    out = HERE / "results" / tag
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for N in sorted(args.sizes, reverse=True):
        for n in names:
            avail = load(n, args.dataset, args.c104).coords.shape[0] - 1
            if avail < N:
                continue
            for s in args.seeds:
                for arm in args.arms:
                    jobs.append(dict(key=f"{n}|{N}|{s}|{arm}", instance=n, N=N, seed=s, arm=arm,
                                     dataset=args.dataset, scoring=args.scoring,
                                     tiebreak=args.tiebreak, c104=args.c104))
    tr = out / "traces.jsonl"
    done = set()
    if tr.exists():
        for ln in tr.read_text().splitlines():
            try:
                done.add(json.loads(ln)["key"])
            except Exception:  # noqa: BLE001  (a line cut by an interruption)
                pass
    todo = [j for j in jobs if j["key"] not in done]
    man = manifest(args, tag, len(jobs))
    man["resumed_with_done"] = len(done)
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"{tag}: {len(jobs)} jobs, {len(done)} already done, {len(todo)} to run", flush=True)

    os.environ["PYTHONHASHSEED"] = "0"   # inherited by every spawned worker
    t0 = time.time()
    with get_context("spawn").Pool(args.workers) as pool, open(tr, "a") as f:
        for k, r in enumerate(pool.imap_unordered(run, todo), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            if k % 100 == 0:
                print(f"  [{k}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)

    import pandas as pd
    recs = [json.loads(ln) for ln in tr.read_text().splitlines() if ln.strip()]
    keys = {j["key"] for j in jobs}
    recs = [r for r in recs if r["key"] in keys]
    pd.DataFrame([flat(r) for r in recs]).to_csv(out / "runs.csv", index=False)
    man.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), runtime_s=round(time.time() - t0),
               n_records=len(recs), complete=len(recs) == len(jobs))
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"wrote {out} ({len(recs)}/{len(jobs)} runs) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
