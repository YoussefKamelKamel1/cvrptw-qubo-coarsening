"""Large-neighbourhood search with larger route pools per subproblem."""
import argparse
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

K_ORT, ORT_EACH = 8, 0.5          # randomised OR-Tools re-solves per subproblem
INSERT_PERMS, INSERT_MAX = 300, 400
SA_READS, SA_SWEEPS, PENALTY = 200, 1000, 1.1


def _rich_pool(inst, D, freed, current, rng):
    """Columns (original customer indices) for one subproblem, and the K solutions."""
    from exp5_setpartition_seeded import neighbourhood, NBHD_CAP
    from exp7_lns import subset, route_ok, route_cost
    from src.baselines import solve_cvrptw_ortools
    from src.coarsening import graph_from_instance
    from src.qubo import vrp_problem_from_graph
    from src.set_partition import generate_pool

    sols = []
    for _ in range(K_ORT):
        perm = freed[:]
        rng.shuffle(perm)
        s = solve_cvrptw_ortools(subset(inst, perm), time_limit_s=ORT_EACH)
        if s.feasible:
            sols.append([[perm[k - 1] for k in r] for r in s.routes])

    sub = subset(inst, freed)
    g, dep = graph_from_instance(sub)
    problem, int_to_id = vrp_problem_from_graph(g, dep, sub.capacity)
    ins = [[freed[int(int_to_id[i]) - 1] for i in r]
           for r in generate_pool(problem, n_perms=INSERT_PERMS, seed=rng.randrange(1 << 30),
                                  max_routes=INSERT_MAX)]

    seed_routes = [r for s in sols for r in s] + [list(r) for r in current]
    cols = {tuple(r) for r in seed_routes + ins}
    # local variants of the good routes, evaluated on the full instance
    full_problem_like = _FullView(inst, D)
    nb = sorted(neighbourhood(seed_routes, full_problem_like) - cols,
                key=lambda r: route_cost(D, r) / len(r))[:NBHD_CAP]
    cols |= set(nb)
    # Variants built across DIFFERENT solutions can repeat a customer inside one
    # route (relocating c into a route that already holds it); the cover constraint
    # would count such a route once, so reject repeats and foreign customers here.
    allowed = set(freed)
    pool = [list(r) for r in cols
            if len(set(r)) == len(r) and set(r) <= allowed and route_ok(inst, D, r)]
    return pool, sols


class _FullView:
    """Minimal VRPProblem-like view so set_partition.route_feasible runs on the full
    instance (used only to filter local variants; route_ok re-checks everything)."""
    def __init__(self, inst, D):
        n = inst.n_customers
        self.source_depot = 0
        self.dests = list(range(1, n + 1))
        self.capacities = [inst.capacity]
        self.weights = {j: float(inst.demand[j]) for j in range(n + 1)}
        self.time_costs = D
        self.costs = D
        self.time_windows = {j: (float(inst.ready_time[j]), float(inst.due_time[j]))
                             for j in range(n + 1)}
        self.service_times = {j: float(inst.service_time[j]) for j in range(n + 1)}


def _exact(pool, freed, D):
    from ortools.linear_solver import pywraplp
    from exp7_lns import route_cost
    s = pywraplp.Solver.CreateSolver("SCIP")
    s.SetTimeLimit(60_000)
    y = [s.BoolVar(f"y{i}") for i in range(len(pool))]
    for c in freed:
        s.Add(sum(y[i] for i, r in enumerate(pool) if c in r) == 1)
    s.Minimize(sum(route_cost(D, pool[i]) * y[i] for i in range(len(pool))))
    if s.Solve() not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None
    return [pool[i] for i in range(len(pool)) if y[i].solution_value() > 0.5]


def _sa(pool, freed, D, seed):
    """SA on the set-partitioning QUBO, built over local column indices."""
    from collections import defaultdict
    from dwave.samplers import SimulatedAnnealingSampler
    from exp7_lns import route_cost
    costs = [route_cost(D, r) for r in pool]
    A = PENALTY * max(costs)
    Q = defaultdict(float)
    for i, c in enumerate(costs):
        Q[(i, i)] += c
    cover = defaultdict(list)
    for i, r in enumerate(pool):
        for c in r:
            cover[c].append(i)
    for idx in cover.values():
        for i in idx:
            Q[(i, i)] -= A
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                Q[(idx[a], idx[b])] += 2 * A
    ss = SimulatedAnnealingSampler().sample_qubo(dict(Q), num_reads=SA_READS,
                                                 num_sweeps=SA_SWEEPS, seed=seed)
    target = sorted(freed)
    best, best_c = None, math.inf
    for d in ss.aggregate().data(["sample"]):
        sel = [pool[i] for i, v in d.sample.items() if v == 1]
        if sorted(c for r in sel for c in r) == target:
            c = sum(costs[pool.index(r)] for r in sel)
            if c < best_c:
                best, best_c = sel, c
    return best


def _instance(args):
    name, N, iters, seed, run_sa = args
    import harness  # noqa: F401
    from exp7_lns import pick_destroy, route_cost, route_ok, total
    from src.baselines import solve_cvrptw_ortools
    from src.datasets import load_solomon
    full = load_solomon(name)
    inst = full.truncate(min(N, full.n_customers))
    D = inst.distance_matrix()
    ort = solve_cvrptw_ortools(inst, time_limit_s=3.0)
    routes = [list(r) for r in ort.routes]
    start = total(D, routes)
    rng = random.Random(seed)
    its, t_lns, t_sa = [], 0.0, 0.0
    for it in range(iters):
        t0 = time.perf_counter()
        idx = pick_destroy(routes, inst.coords, rng)
        freed = sorted(c for i in idx for c in routes[i])
        cur = [routes[i] for i in idx]
        cur_c = sum(route_cost(D, r) for r in cur)
        pool, sols = _rich_pool(inst, D, freed, cur, rng)
        best_single = min((sum(route_cost(D, r) for r in s) for s in sols), default=math.inf)
        ex = _exact(pool, freed, D)
        ex_c = sum(route_cost(D, r) for r in ex) if ex else math.inf
        t_lns += time.perf_counter() - t0
        t1 = time.perf_counter()
        sa = _sa(pool, freed, D, seed * 1000 + it) if run_sa else None
        sa_c = sum(route_cost(D, r) for r in sa) if sa else math.inf
        t_sa += time.perf_counter() - t1
        its.append(dict(instance=name, N=inst.n_customers, seed=seed, it=it, pool=len(pool),
                        current=cur_c, best_single=best_single, exact=ex_c, sa=sa_c))
        exact_cover = ex is not None and sorted(c for r in ex for c in r) == freed
        if exact_cover and ex_c < cur_c - 1e-9 and all(route_ok(inst, D, r) for r in ex):
            routes = [r for i, r in enumerate(routes) if i not in idx] + ex
    assert all(route_ok(inst, D, r) for r in routes)
    assert sorted(c for r in routes for c in r) == list(range(1, inst.n_customers + 1))
    long = solve_cvrptw_ortools(inst, time_limit_s=3.0 + t_lns)
    summary = dict(instance=name, N=inst.n_customers, seed=seed, start=start,
                   lns_exact=total(D, routes), ort_long=long.cost if long.feasible else math.nan,
                   lns_wall=t_lns, sa_wall=t_sa)
    return summary, its


def main():
    import pandas as pd
    from harness import OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--N", type=int, nargs="+", default=[100])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--no-sa", action="store_true")
    a = ap.parse_args()
    jobs = [(n, N, a.iters, s, not a.no_sa) for N in a.N for n in a.instances
            for s in a.seeds]
    workers = min(8, os.cpu_count(), len(jobs))
    print(f"{len(jobs)} runs on {workers} workers, {a.iters} iterations", flush=True)
    sums, its, t0 = [], [], time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        for fu in as_completed([ex.submit(_instance, j) for j in jobs]):
            s, i = fu.result()
            sums.append(s); its.extend(i)
            print(f"  finished {s['instance']} at {time.perf_counter()-t0:.0f}s", flush=True)
    pd.DataFrame(sums).to_csv(OUT / f"exp7b_{a.tag}_summary.csv", index=False)
    pd.DataFrame(its).to_csv(OUT / f"exp7b_{a.tag}_iterations.csv", index=False)
    print(f"done in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
