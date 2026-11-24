"""Energy of the OR-Tools solution written into the position QUBO, against the best of
five simulated-annealing runs on the same QUBO.
"""
import inspect
import math
import os
import sys
import textwrap
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NS = (10, 15, 20, 40)
SEEDS = range(5)
FLEETS = (2,)

MARKERS = [
    ("# 1. UNIQUE VISITS", "uniq"),
    ("# 2. CONTINUITY + SELF-LOOPS", "cont"),
    ("# 3. CAPACITY (log-slack equality)", "cap"),
    ("# A. depot initial check", "tw_depot"),
    ("# B. pairwise k -> k+1", "tw_pair"),
    ("# C. triangle lookahead k -> k+2", "tw_tri"),
    ("# 5. OBJECTIVE (Clarke-Wright savings)", "dist"),
]
INLINE = [
    ("qubo.add(((i, j, k), (i, j, k + 1)), selfloop)",
     "_SEC[0] = 'self'; qubo.add(((i, j, k), (i, j, k + 1)), selfloop); _SEC[0] = 'cont'"),
    ("qubo.add(((i, j1, k), (i, j2, k + 1)), tight)",
     "_SEC[0] = 'tw_tight'; qubo.add(((i, j1, k), (i, j2, k + 1)), tight); "
     "_SEC[0] = 'tw_pair'"),
]


def tagged_builder():
    """The published get_qubo, instrumented to record which term adds each coefficient."""
    import src.qubo as q
    src = textwrap.dedent(inspect.getsource(q.VRPProblem.get_qubo))
    for marker, tag in MARKERS:
        assert src.count(marker) == 1, marker
        src = src.replace(marker, f"_SEC[0] = '{tag}'  {marker}")
    for old, new in INLINE:
        assert src.count(old) == 1, old
        src = src.replace(old, new)

    sec = ["?"]

    class TaggedQubo(q.Qubo):
        def __init__(self):
            super().__init__()
            self.terms = defaultdict(lambda: defaultdict(float))

        def add(self, key, value):
            super().add(key, value)
            self.terms[sec[0]][self._norm(key)] += value

        @staticmethod
        def _norm(key):
            if isinstance(key, tuple) and len(key) == 2 and not isinstance(key[0], int):
                try:
                    return tuple(sorted(key))
                except TypeError:
                    return tuple(sorted(key, key=str))
            return key

    ns = dict(vars(q))
    ns.update(Qubo=TaggedQubo, _SEC=sec)
    exec(compile(src, "<tagged-get_qubo>", "exec"), ns)
    return ns["get_qubo"]


def energy(coeffs, x):
    return sum(c * x.get(a, 0) * x.get(b, 0) for (a, b), c in coeffs.items())


def _case(args):
    name, N, div = args
    import harness
    from oracle_coarsener import OracleCoarsener
    from src.baselines import solve_cvrptw_ortools
    from src.coarsening import graph_from_instance
    from src.datasets import load_solomon
    from src.qubo import adaptive_penalties, vrp_problem_from_graph
    from src.solvers import SolverConfig, solve_qubo, _k_max_fqs

    inst = load_solomon(name).truncate(N)
    ref = solve_cvrptw_ortools(inst, time_limit_s=3.0)
    if not ref.feasible:
        return []
    g, dep = graph_from_instance(inst)
    c = OracleCoarsener(graph=g, depot_id=dep, alpha=1.0, beta=1.0,
                        P=harness.GNN_FIXED_P, radiusCoeff=harness.GNN_FIXED_RC,
                        routes=ref.routes)
    cg, _ = c.coarsen()
    n_c = len(cg.nodes) - 1
    problem, int_to_id = vrp_problem_from_graph(cg, dep, inst.capacity,
                                                num_vehicles=max(2, n_c // div))
    pw = adaptive_penalties(problem, None)
    n_v = len(problem.capacities)
    k_default = _k_max_fqs(len(problem.dests), n_v)

    # reference assignment on the coarsened graph
    id_to_int = {v: k for k, v in int_to_id.items()} if isinstance(int_to_id, dict) \
        else {v: i for i, v in enumerate(int_to_id)}
    owner = {}
    for sn in cg.nodes:
        if sn != dep:
            for o in c.inflate_route([[sn]])[0]:
                owner[o] = sn
    coarse_routes = []
    for r in ref.routes:
        seq = []
        for cust in map(str, r):
            if not seq or seq[-1] != owner[cust]:
                seq.append(owner[cust])
        coarse_routes.append(seq)
    longest = max(len(s) for s in coarse_routes)
    # does the published encoding (default k_max) represent the reference at all?
    repr_default = len(coarse_routes) <= n_v and longest <= k_default
    # widen k_max just enough to express it; fleet and penalties stay as published
    k_max = max(k_default, longest)
    limits = [k_max] * n_v
    base = dict(instance=name, N=N, fleet_divisor=div, n_v=n_v, k_default=k_default,
                k_max=k_max, ref_routes=len(coarse_routes), longest=longest,
                repr_default=int(repr_default), ref_cost=ref.cost)
    if len(coarse_routes) > n_v:
        return [dict(base, skipped="more reference routes than vehicles")]

    tagged = tagged_builder()(problem, limits, pw, smart_capacity=True)
    plain = problem.get_qubo(limits, pw, smart_capacity=True)

    # fidelity: tagged terms must sum to the original QUBO exactly
    merged = defaultdict(float)
    for t in tagged.terms.values():
        for k, v in t.items():
            merged[k] += v
    assert set(merged) == set(plain.dict) and all(
        math.isclose(merged[k], plain.dict[k], rel_tol=1e-9, abs_tol=1e-6)
        for k in plain.dict), "tagged builder does not reproduce get_qubo"

    x_ref = {}
    for i, seq in enumerate(coarse_routes):
        for k, node in enumerate(seq):
            x_ref[(i, id_to_int[node], k)] = 1
    Q = problem.capacities[0]
    total = sum(problem.weights.get(j, 0) for j in problem.dests)
    if total > Q:                                   # capacity kept only if binding
        M = math.floor(math.log2(max(1.0, min(Q, total)))) + 1
        for i in range(n_v):
            load = sum(problem.weights.get(j, 0) * x_ref.get((i, j, k), 0)
                       for j in problem.dests for k in range(k_max))
            slack = int(round(Q - load))
            for m in range(M):
                if (slack >> m) & 1:
                    x_ref[("s", i, m)] = 1

    rows = []
    e_ref = {t: energy(cf, x_ref) for t, cf in tagged.terms.items()}
    for seed in SEEDS:
        cfg = SolverConfig("sa", 200, 200, seed=seed)
        x_sa = dict(solve_qubo(plain, cfg, limit=1)[0])
        e_sa = {t: energy(cf, x_sa) for t, cf in tagged.terms.items()}
        row = dict(base, seed=seed, E_ref=sum(e_ref.values()), E_sa=sum(e_sa.values()))
        for t in tagged.terms:
            row[f"d_{t}"] = e_ref[t] - e_sa[t]          # > 0: term penalises the reference more
        rows.append(row)
    return rows


def main():
    import pandas as pd
    from harness import OUT
    from src.datasets import train_test_split_instances
    _, test = train_test_split_instances()
    jobs = [(n, N, div) for div in FLEETS for N in NS for n in test]
    with ProcessPoolExecutor(min(8, os.cpu_count())) as ex:
        rows = [r for rs in ex.map(_case, jobs) for r in rs]
    df = pd.DataFrame(rows)
    out = OUT / "exp3_energy_diagnosis.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
