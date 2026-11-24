"""Large-neighbourhood search started from the OR-Tools solution, with exact route
selection on each subproblem.

  python experiments/exp7_lns.py --instances R104 RC108 --N 100 --iters 10
"""
import argparse
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SA_READS, SA_SWEEPS, PENALTY = 200, 1000, 1.1
SUB_ORT_TIME = 1.0


# ---------------------------------------------------------------- instance helpers
def subset(inst, customers):
    """VRPInstance over the depot plus the given (original-index) customers."""
    from src.datasets import VRPInstance
    idx = [0] + list(customers)
    take = lambda a: None if a is None else a[idx].copy()          # noqa: E731
    return VRPInstance(name=f"{inst.name}-sub", coords=inst.coords[idx].copy(),
                       demand=inst.demand[idx].copy(), capacity=inst.capacity,
                       n_vehicles=inst.n_vehicles, ready_time=take(inst.ready_time),
                       due_time=take(inst.due_time), service_time=take(inst.service_time),
                       family=inst.family)


def route_cost(D, r):
    seq = [0] + list(r) + [0]
    return float(sum(D[seq[i], seq[i + 1]] for i in range(len(seq) - 1)))


def route_ok(inst, D, r):
    """Capacity and time-window feasibility of one route on the full instance."""
    if inst.demand[list(r)].sum() > inst.capacity + 1e-9:
        return False
    t, prev = inst.ready_time[0], 0
    for j in r:
        t += D[prev, j]
        if t > inst.due_time[j] + 1e-9:
            return False
        t = max(t, inst.ready_time[j]) + inst.service_time[j]
        prev = j
    return t + D[prev, 0] <= inst.due_time[0] + 1e-9


def total(D, routes):
    return sum(route_cost(D, r) for r in routes)


def pick_destroy(routes, coords, rng):
    """A random route and its nearest route by centroid."""
    if len(routes) < 2:
        return list(range(len(routes)))
    cen = [coords[r].mean(axis=0) for r in routes]
    a = rng.randrange(len(routes))
    b = min((i for i in range(len(routes)) if i != a),
            key=lambda i: float(np.linalg.norm(cen[i] - cen[a])))
    return [a, b]


# ---------------------------------------------------------------- subproblem solvers
def _sp_pool(sub, current_sub_routes, seed):
    """Set-partitioning column pool on the subproblem, always containing the current routes."""
    from exp5_setpartition_seeded import neighbourhood, NBHD_CAP
    from src.coarsening import graph_from_instance
    from src.qubo import vrp_problem_from_graph
    from src.set_partition import generate_pool, route_cost as sp_cost
    g, dep = graph_from_instance(sub)
    problem, int_to_id = vrp_problem_from_graph(g, dep, sub.capacity)
    id_to_int = {v: i for i, v in enumerate(int_to_id)}
    to_int = lambda r: [id_to_int[str(c)] for c in r]               # noqa: E731
    cur = [tuple(to_int(r)) for r in current_sub_routes]
    rand = [tuple(r) for r in generate_pool(problem, n_perms=60, seed=seed)]
    base = list(dict.fromkeys(rand + cur))
    nb = sorted(neighbourhood([list(r) for r in cur], problem) - set(base),
                key=lambda r: sp_cost(problem, list(r)) / len(r))[:NBHD_CAP]
    pool = [list(r) for r in base + nb]
    back = lambda r: [int(int_to_id[i]) for i in r]                 # noqa: E731
    return problem, pool, back


def solve_sub(kind, sub, current_sub_routes, seed):
    """Return a list of subproblem routes (sub indices) or None."""
    if kind == "ort":
        from src.baselines import solve_cvrptw_ortools
        s = solve_cvrptw_ortools(sub, time_limit_s=SUB_ORT_TIME)
        return [list(r) for r in s.routes] if s.feasible else None

    from src.set_partition import cover_status, decode_setpartition, route_cost as sp_cost
    problem, pool, back = _sp_pool(sub, current_sub_routes, seed)
    if kind == "exact":
        from exp5_setpartition_seeded import exact_optimum
        from ortools.linear_solver import pywraplp
        s = pywraplp.Solver.CreateSolver("SCIP")
        s.SetTimeLimit(60_000)
        y = [s.BoolVar(f"y{i}") for i in range(len(pool))]
        for c in problem.dests:
            s.Add(sum(y[i] for i, r in enumerate(pool) if c in r) == 1)
        s.Minimize(sum(sp_cost(problem, pool[i]) * y[i] for i in range(len(pool))))
        if s.Solve() not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return None
        return [back(pool[i]) for i in range(len(pool)) if y[i].solution_value() > 0.5]

    from dwave.samplers import SimulatedAnnealingSampler
    from src.set_partition import build_setpartition_qubo
    q = build_setpartition_qubo(problem, pool, penalty_scale=PENALTY)
    ss = SimulatedAnnealingSampler().sample_qubo(q.dict, num_reads=SA_READS,
                                                 num_sweeps=SA_SWEEPS, seed=seed)
    best, best_c = None, math.inf
    for d in ss.aggregate().data(["sample"]):
        sel = decode_setpartition(dict(d.sample), pool)
        if cover_status(sel, problem)[0]:
            c = sum(sp_cost(problem, r) for r in sel)
            if c < best_c:
                best, best_c = sel, c
    return [back(r) for r in best] if best else None


# ---------------------------------------------------------------- one LNS run
def lns(inst, D, start_routes, kind, iters, seed):
    routes = [list(r) for r in start_routes]
    cost0 = total(D, routes)
    rng = random.Random(seed)
    accepted, t0 = 0, time.perf_counter()
    trace = [cost0]
    for it in range(iters):
        idx = pick_destroy(routes, inst.coords, rng)
        freed = sorted(c for i in idx for c in routes[i])
        sub = subset(inst, freed)
        to_sub = {c: k + 1 for k, c in enumerate(freed)}
        cur_sub = [[to_sub[c] for c in routes[i]] for i in idx]
        new_sub = solve_sub(kind, sub, cur_sub, seed * 1000 + it)
        if new_sub:
            new = [[freed[k - 1] for k in r] for r in new_sub]
            served = sorted(c for r in new for c in r)
            ok = served == freed and all(route_ok(inst, D, r) for r in new)
            old_c = sum(route_cost(D, routes[i]) for i in idx)
            new_c = sum(route_cost(D, r) for r in new)
            if ok and new_c < old_c - 1e-9:
                routes = [r for i, r in enumerate(routes) if i not in idx] + new
                accepted += 1
        trace.append(total(D, routes))
    assert all(route_ok(inst, D, r) for r in routes)
    assert sorted(c for r in routes for c in r) == list(range(1, inst.n_customers + 1))
    return dict(cost=total(D, routes), routes=len(routes), accepted=accepted,
                wall=time.perf_counter() - t0, trace=trace)


def _instance(args):
    name, N, iters, seed = args
    import harness  # noqa: F401
    from src.baselines import solve_cvrptw_ortools
    from src.datasets import load_solomon
    full = load_solomon(name)
    inst = full.truncate(min(N, full.n_customers))
    D = inst.distance_matrix()
    ort = solve_cvrptw_ortools(inst, time_limit_s=3.0)
    start = [list(r) for r in ort.routes]
    assert all(route_ok(inst, D, r) for r in start), "checker disagrees with OR-Tools"
    base = total(D, start)
    rows = [dict(instance=name, N=inst.n_customers, seed=seed, arm="ORT-3s",
                 cost=base, routes=len(start), accepted=0, wall=3.0)]
    sa_wall = None
    for kind, arm in (("sa", "LNS-SA"), ("exact", "LNS-exact"), ("ort", "LNS-ORT")):
        r = lns(inst, D, start, kind, iters, seed)
        if kind == "sa":
            sa_wall = r["wall"]
        rows.append(dict(instance=name, N=inst.n_customers, seed=seed, arm=arm,
                         cost=r["cost"], routes=r["routes"], accepted=r["accepted"],
                         wall=r["wall"], trace=";".join(f"{c:.2f}" for c in r["trace"])))
    budget = 3.0 + sa_wall
    long = solve_cvrptw_ortools(inst, time_limit_s=budget)
    rows.append(dict(instance=name, N=inst.n_customers, seed=seed, arm="ORT-long",
                     cost=long.cost if long.feasible else float("nan"),
                     routes=len(long.routes), accepted=0, wall=budget))
    for r in rows:
        r["start_cost"] = base
    return rows


def main():
    import pandas as pd
    from harness import OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--N", type=int, nargs="+", default=[100])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="exp7_lns_pilot.csv")
    a = ap.parse_args()
    jobs = [(n, N, a.iters, s) for N in a.N for n in a.instances for s in a.seeds]
    workers = min(8, os.cpu_count(), len(jobs))
    print(f"{len(jobs)} LNS runs on {workers} workers, {a.iters} iterations each", flush=True)
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        for fu in as_completed([ex.submit(_instance, j) for j in jobs]):
            rows.extend(fu.result())
            print(f"  finished one instance at {time.perf_counter()-t0:.0f}s", flush=True)
    out = OUT / a.out
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows) in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
