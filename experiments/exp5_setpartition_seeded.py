"""Set partitioning with the OR-Tools routes and local variants in the pool, at several
penalty weights.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NS = (10, 15, 20)
SEEDS = range(3)
READS, SWEEPS = 500, 1000
NBHD_CAP = 300


def neighbourhood(routes, problem):
    from src.set_partition import route_feasible
    out = set()

    def add(r):
        r = tuple(r)
        if r and route_feasible(problem, list(r)):
            out.add(r)

    for r in routes:
        for i in range(len(r)):
            for j in range(i + 2, len(r) + 1):
                add(r[:i] + r[i:j][::-1] + r[j:])           # 2-opt
        for i in range(1, len(r)):
            add(r[:i]); add(r[i:])                          # split
    for ai, a in enumerate(routes):
        for bi, b in enumerate(routes):
            if ai == bi:
                continue
            add(a + b)                                      # merge
            for i, c in enumerate(a):                       # relocate c from a into b
                rest = a[:i] + a[i + 1:]
                add(rest)
                for p in range(len(b) + 1):
                    add(b[:p] + [c] + b[p:])
            if ai < bi:
                for i in range(len(a)):                     # swap
                    for j in range(len(b)):
                        na, nb = a[:], b[:]
                        na[i], nb[j] = b[j], a[i]
                        add(na); add(nb)
    return out


def exact_optimum(pool, costs, custs):
    from ortools.linear_solver import pywraplp
    s = pywraplp.Solver.CreateSolver("SCIP")
    s.SetTimeLimit(60_000)
    y = [s.BoolVar(f"y{i}") for i in range(len(pool))]
    for c in custs:
        s.Add(sum(y[i] for i, r in enumerate(pool) if c in r) == 1)
    s.Minimize(sum(costs[i] * y[i] for i in range(len(pool))))
    st = s.Solve()
    if st not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None, False
    return s.Objective().Value(), st == pywraplp.Solver.OPTIMAL


def _case(args):
    name, N, seed = args
    import harness  # noqa: F401
    from dwave.samplers import SimulatedAnnealingSampler
    from src.baselines import solve_cvrptw_ortools
    from src.coarsening import graph_from_instance
    from src.datasets import load_solomon
    from src.qubo import vrp_problem_from_graph
    from src.set_partition import (generate_pool, build_setpartition_qubo,
                                   decode_setpartition, cover_status, route_cost,
                                   route_feasible)

    inst = load_solomon(name).truncate(N)
    ort = solve_cvrptw_ortools(inst, time_limit_s=3.0)
    if not ort.feasible:
        return []
    graph, depot = graph_from_instance(inst)
    problem, int_to_id = vrp_problem_from_graph(graph, depot, inst.capacity)
    id_to_int = {v: i for i, v in enumerate(int_to_id)}
    ort_routes = [[id_to_int[str(c)] for c in r] for r in ort.routes]
    ort_cost = sum(route_cost(problem, r) for r in ort_routes)
    assert all(route_feasible(problem, r) for r in ort_routes), "OR-Tools route infeasible"
    assert abs(ort_cost - ort.cost) < 1e-3 * max(1, ort.cost), "cost units differ"
    custs = list(problem.dests)

    rand = [tuple(r) for r in generate_pool(problem, n_perms=60, seed=seed)]
    seeded = list(dict.fromkeys(rand + [tuple(r) for r in ort_routes]))
    nb = sorted(neighbourhood(ort_routes, problem) - set(seeded),
                key=lambda r: route_cost(problem, list(r)) / len(r))[:NBHD_CAP]
    pools = {"A_random": rand, "B_+ortools": seeded, "C_+neighbourhood": seeded + nb}

    # (pool, penalty_scale): 10 is the paper's setting; any scale > 1 keeps the
    # minimum-energy state an exact cover, so lower scales test conditioning
    runs = [(a, 10.0) for a in pools] + [("C_+neighbourhood", 2.0),
                                         ("C_+neighbourhood", 1.1)]
    rows, exact_cache = [], {}
    for arm, scale in runs:
        pool = [list(r) for r in pools[arm]]
        costs = [route_cost(problem, r) for r in pool]
        if arm not in exact_cache:
            exact_cache[arm] = exact_optimum(pool, costs, custs)
        opt, proven = exact_cache[arm]
        qubo = build_setpartition_qubo(problem, pool, penalty_scale=scale)
        t0 = time.perf_counter()
        ss = SimulatedAnnealingSampler().sample_qubo(qubo.dict, num_reads=READS,
                                                     num_sweeps=SWEEPS, seed=seed)
        t_sa = time.perf_counter() - t0
        best, covered, total = None, 0, 0
        for d in ss.aggregate().data(["sample", "num_occurrences"]):
            sel = decode_setpartition(dict(d.sample), pool)
            occ = int(d.num_occurrences)
            total += occ
            if cover_status(sel, problem)[0]:
                covered += occ
                c = sum(route_cost(problem, r) for r in sel)
                best = c if best is None else min(best, c)
        rows.append(dict(instance=name, N=N, seed=seed, arm=arm, penalty_scale=scale,
                         pool=len(pool),
                         ort_cost=ort_cost, ort_routes=len(ort_routes),
                         exact_cost=opt, exact_proven=proven,
                         sa_cost=best, sa_cover_frac=covered / total if total else 0,
                         sa_time=t_sa))
    return rows


def main():
    import pandas as pd
    from harness import OUT
    from src.datasets import train_test_split_instances
    _, test = train_test_split_instances()
    jobs = [(n, N, s) for N in NS for n in test for s in SEEDS]
    workers = min(8, os.cpu_count())
    print(f"{len(jobs)} cases on {workers} workers", flush=True)
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_case, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            rows.extend(fu.result())
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)}  {time.perf_counter()-t0:.0f}s", flush=True)
    out = OUT / "exp5_setpartition_seeded.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows) in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
