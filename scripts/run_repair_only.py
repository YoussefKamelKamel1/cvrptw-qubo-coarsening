"""repair-only baseline.

The classical repair (greedy insertion + time-window repair + inter-route moves)
is strong and dominates post-repair feasibility. To show how much of the pipeline
result is the repair alone, we run the SAME repair on an EMPTY sample: no SA, no
QUBO, no coarsening. The repair then inserts every customer greedily. Comparing
this to the full pipeline (none/heuristic/gnn) isolates the pipeline's added value
over repair alone.

Writes results/repair_only.csv and prints gap/feasibility vs the phase-4 pipeline.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import train_test_split_instances, load_solomon   # noqa: E402
from src.coarsening import graph_from_instance                      # noqa: E402
from src.qubo import vrp_problem_from_graph                         # noqa: E402
from src.solvers import VRPSolution, _k_max_fqs                     # noqa: E402
from src.refcache import build_cache, load_cache                    # noqa: E402


def repair_only(name, N):
    inst = load_solomon(name).truncate(N)
    graph, depot = graph_from_instance(inst)
    problem, _ = vrp_problem_from_graph(graph, depot, inst.capacity)
    nc, nv = len(problem.dests), len(problem.capacities)
    k = _k_max_fqs(nc, nv)
    # empty sample -> raw_solution empty -> repair inserts all customers greedily
    sol = VRPSolution(problem, {}, [k] * nv)
    return sol.check(), sol.total_cost()


def main():
    import numpy as np
    import pandas as pd

    _, test = train_test_split_instances()
    Ns = [10, 15, 20]
    build_cache([(n, N) for N in Ns for n in test])
    cache = load_cache()

    rows = []
    for N in Ns:
        for name in test:
            feas, cost = repair_only(name, N)
            ref = cache.get((name, N))
            gap = ((cost - ref) / ref) if (ref and feas) else np.nan
            rows.append(dict(instance=name, N=N, feasible=feas, cost=cost,
                             ortools=ref, opt_gap=gap))
    rr = pd.DataFrame(rows)
    rr.to_csv(ROOT / "results" / "repair_only.csv", index=False)

    print("=== Repair-only baseline (no SA/QUBO/coarsening) ===")
    for N in Ns:
        s = rr[rr.N == N]
        fe = s[s.feasible]
        print(f"  N={N}: feasible {100*s.feasible.mean():4.0f}%   "
              f"mean gap {fe.opt_gap.mean()*100:6.1f}%  (n={len(fe)})")

    # side-by-side vs the phase-4 pipeline
    print("\n=== vs full pipeline (phase4_full.csv, feasible-run mean gap) ===")
    p4 = pd.read_csv(ROOT / "results" / "phase4_full.csv")
    fe = p4[p4.is_feasible]
    print(f"  {'N':>3} {'repair-only':>12} {'none':>8} {'heuristic':>10} {'gnn':>8}")
    for N in Ns:
        ro = rr[(rr.N == N) & rr.feasible].opt_gap.mean() * 100
        row = f"  {N:>3} {ro:>11.1f}%"
        for c in ["none", "heuristic", "gnn"]:
            v = fe[(fe.N == N) & (fe.condition == c)].opt_gap.mean() * 100
            row += f" {v:>7.1f}%"
        print(row)


if __name__ == "__main__":
    main()
