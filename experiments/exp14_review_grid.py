"""GNN grid (N = 10, 20, 40, 50, 60) with per-run traces.

  python experiments/exp14_review_grid.py --orient stored --workers 3
  python experiments/exp14_review_grid.py --orient stored --c104 fixed --only C104 --workers 3
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

MODEL = str(ROOT / "results" / "gnn_model_rl.pt")
SIZES = [10, 20, 40, 50, 60, 80, 100]
ARMS = ["heuristic", "heuristic_fixed", "gnn"]


def _load(name, c104_fixed):
    from src.datasets import load_solomon
    inst = load_solomon(name)
    if name != "C104" or not c104_fixed:
        return inst
    import numpy as np
    from dataclasses import replace
    # restore file row "37" (5,45, demand 10, ready 0, due 1110, service 90); the file
    # numbers the depot 1, so this is array index 36
    ins = lambda a, v: np.insert(a, 36, v, axis=0)
    return replace(inst, coords=ins(inst.coords, [5.0, 45.0]), demand=ins(inst.demand, 10.0),
                   ready_time=ins(inst.ready_time, 0.0), due_time=ins(inst.due_time, 1110.0),
                   service_time=ins(inst.service_time, 90.0))


def _flip_graph(g):
    for e in g.edges:
        e.u_id, e.v_id = e.v_id, e.u_id


def _flipped(cls):
    from src.coarsening import euclidean

    class Flipped(cls):
        def _reconnect(self, g, super_node, ni, nj):
            nbrs = (g.get_neighbors(ni.id) | g.get_neighbors(nj.id)) - {
                ni.id, nj.id, super_node.id, self.depot_id}
            for nb in nbrs:
                if nb in g.nodes:
                    g.add_edge(nb, super_node.id, euclidean(super_node, g.nodes[nb]))
    return Flipped


_MODEL = None


def run(spec):
    global _MODEL
    import torch
    torch.set_num_threads(1)
    from src.datasets import SOLOMON_FAMILY_HYPERPARAMS
    from src.coarsening import graph_from_instance, SpatioTemporalGraphCoarsener
    from src.experiment import solve_on_graph
    from src.metrics import calculate_route_metrics
    from src.runner_parallel import HEUR_FIXED, GNN_FIXED_P, GNN_FIXED_RC
    from src.solvers import SolverConfig
    from src.utils import set_global_seed

    name, N, seed, arm = spec["name"], spec["N"], spec["seed"], spec["arm"]
    flipped = spec["orient"] == "flipped"
    set_global_seed(seed)
    inst = _load(name, spec["c104"] == "fixed")
    if inst.coords.shape[0] - 1 < N:
        return None
    inst = inst.truncate(N)
    cfg = SolverConfig("sa", 200, 200, seed=seed)
    graph, depot = graph_from_instance(inst)
    if flipped:
        _flip_graph(graph)
    if arm == "gnn":
        from src.gnn_coarsening import GNNCoarsener, load_model
        if _MODEL is None:
            _MODEL = load_model(MODEL, device="cpu")
        cls = _flipped(GNNCoarsener) if flipped else GNNCoarsener
        co = cls(graph, P=GNN_FIXED_P, radiusCoeff=GNN_FIXED_RC, depot_id=depot,
                 model=_MODEL, device="cpu")
    else:
        params = SOLOMON_FAMILY_HYPERPARAMS[inst.family] if arm == "heuristic" else HEUR_FIXED
        cls = _flipped(SpatioTemporalGraphCoarsener) if flipped else SpatioTemporalGraphCoarsener
        co = cls(graph=graph, depot_id=depot, **params)
    coarse, _ = co.coarsen()

    captured = []
    orig = co.inflate_route
    co.inflate_route = lambda r: (captured.append([list(x) for x in r]), orig(r))[1]
    _, m, _ = solve_on_graph(coarse, depot, inst.capacity, "FQS", cfg,
                             penalty_mode="adaptive", coarsener=co)
    cm = calculate_route_metrics(coarse, captured[0], depot, inst.capacity)
    return dict(instance=name, N=N, seed=seed, condition=arm, orient=spec["orient"],
                c104=spec["c104"], is_feasible=bool(m["is_feasible"]),
                coarse_feasible=bool(cm["is_feasible"]), pre_feasible=bool(m["pre_feasible"]),
                pre_total_viol=int(m["pre_total_viol"]), coarse_n=len(coarse.nodes),
                num_qubo_vars=int(m["num_qubo_vars"]), total_distance=float(m["total_distance"]),
                coarse_distance=float(cm["total_distance"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orient", default="stored", choices=["stored", "flipped"])
    ap.add_argument("--c104", default="asloaded", choices=["asloaded", "fixed"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    from src.datasets import train_test_split_instances
    _, test = train_test_split_instances()
    names = a.only or test
    jobs = [dict(name=n, N=N, seed=s, arm=arm, orient=a.orient, c104=a.c104)
            for N in sorted(SIZES, reverse=True) for n in names for s in (0, 1, 2) for arm in ARMS]
    out = HERE / "results" / f"exp14_{a.orient}_{a.c104}{'_' + '_'.join(a.only) if a.only else ''}.csv"
    t0, rows = time.time(), []
    with Pool(a.workers) as pool, open(out, "w", newline="") as f:
        w = None
        for k, r in enumerate(pool.imap_unordered(run, jobs), 1):
            if r is None:
                continue
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r)); w.writeheader()
            w.writerow(r); f.flush(); rows.append(r)
            if k % 50 == 0:
                print(f"  [{k}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {out} ({len(rows)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
