"""Simulated annealing, tabu search and steepest descent on the set-partitioning QUBOs."""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NS = (10, 15, 20)
SEEDS = range(3)
PENALTY = 1.1
SAMPLERS = [("SA 500x1000", "sa", dict(num_reads=500, num_sweeps=1000)),
            ("SA 100x10000", "sa", dict(num_reads=100, num_sweeps=10000)),
            ("tabu 100x100ms", "tabu", dict(num_reads=100, timeout=100))]


def _evaluate(ss, pool, problem):
    from src.set_partition import decode_setpartition, cover_status, route_cost
    best, covered, total = None, 0, 0
    for d in ss.aggregate().data(["sample", "num_occurrences"]):
        sel = decode_setpartition(dict(d.sample), pool)
        occ = int(d.num_occurrences)
        total += occ
        if cover_status(sel, problem)[0]:
            covered += occ
            c = sum(route_cost(problem, r) for r in sel)
            best = c if best is None else min(best, c)
    return best, covered / total if total else 0.0


def _case(args):
    name, N, seed = args
    import harness  # noqa: F401
    from dwave.samplers import SimulatedAnnealingSampler, TabuSampler
    from exp5_setpartition_seeded import neighbourhood, exact_optimum, NBHD_CAP
    from src.baselines import solve_cvrptw_ortools
    from src.coarsening import graph_from_instance
    from src.datasets import load_solomon
    from src.qubo import vrp_problem_from_graph
    from src.set_partition import generate_pool, build_setpartition_qubo, route_cost

    inst = load_solomon(name).truncate(N)
    ort = solve_cvrptw_ortools(inst, time_limit_s=3.0)
    if not ort.feasible:
        return []
    graph, depot = graph_from_instance(inst)
    problem, int_to_id = vrp_problem_from_graph(graph, depot, inst.capacity)
    id_to_int = {v: i for i, v in enumerate(int_to_id)}
    ort_routes = [[id_to_int[str(c)] for c in r] for r in ort.routes]
    ort_cost = sum(route_cost(problem, r) for r in ort_routes)

    rand = [tuple(r) for r in generate_pool(problem, n_perms=60, seed=seed)]
    seeded = list(dict.fromkeys(rand + [tuple(r) for r in ort_routes]))
    nb = sorted(neighbourhood(ort_routes, problem) - set(seeded),
                key=lambda r: route_cost(problem, list(r)) / len(r))[:NBHD_CAP]
    pool = [list(r) for r in seeded + nb]
    costs = [route_cost(problem, r) for r in pool]
    opt, proven = exact_optimum(pool, costs, list(problem.dests))
    qubo = build_setpartition_qubo(problem, pool, penalty_scale=PENALTY)

    rows = []
    for label, kind, kw in SAMPLERS:
        t0 = time.perf_counter()
        if kind == "sa":
            ss = SimulatedAnnealingSampler().sample_qubo(qubo.dict, seed=seed, **kw)
        else:
            ss = TabuSampler().sample_qubo(qubo.dict, seed=seed, **kw)
        wall = time.perf_counter() - t0
        best, cover = _evaluate(ss, pool, problem)
        rows.append(dict(instance=name, N=N, seed=seed, sampler=label, pool=len(pool),
                         ort_cost=ort_cost, exact_cost=opt, exact_proven=proven,
                         cost=best, cover_frac=cover, wall=wall))
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
    out = OUT / "exp6_samplers.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows) in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
