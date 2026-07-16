"""Does a coarsened starting point beat cold greedy once both are polished?

Apply the SAME local search (2-opt + relocate) to:
  * the GNN-coarsened pipeline output (SA -> decode -> repair -> inflate), and
  * the repair-only solution (cold greedy, no SA/QUBO/coarsening),
at equal budget, then compare gap to OR-Tools. If GNN+LS < repair-only+LS the
coarsening provides useful structure; if they converge, it does not.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import train_test_split_instances, load_solomon        # noqa: E402
from src.coarsening import graph_from_instance                           # noqa: E402
from src.qubo import vrp_problem_from_graph                              # noqa: E402
from src.solvers import VRPSolution, SolverConfig, _k_max_fqs            # noqa: E402
from src.experiment import solve_on_graph                                # noqa: E402
from src.gnn_coarsening import GNNCoarsener, load_model                  # noqa: E402
from src.refcache import build_cache, load_cache                         # noqa: E402
from src.local_search import (local_search, solution_distance,           # noqa: E402
                              route_tw_violations, route_demand, _cap_of)

MODEL = str(ROOT / "results" / "gnn_model_rl.pt")


def full_feasible(p, routes) -> bool:
    seen = []
    for i, r in enumerate(routes):
        seen += r
        if route_demand(p, r) > _cap_of(p, i) or route_tw_violations(p, r) != 0:
            return False
    return sorted(seen) == sorted(p.dests)


def to_int_routes(routes_str, id_to_int, depot_id):
    out = []
    for r in routes_str:
        cr = [id_to_int[c] for c in r if c != depot_id]
        if cr:
            out.append(cr)
    return out


def main():
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    _, test = train_test_split_instances()
    Ns = [10, 15, 20]
    seeds = [0, 1, 2]
    build_cache([(n, N) for N in Ns for n in test])
    cache = load_cache()
    model = load_model(MODEL, device="cpu")

    rows = []
    for N in Ns:
        for name in test:
            inst = load_solomon(name).truncate(N)
            graph, depot = graph_from_instance(inst)
            problem, int_to_id = vrp_problem_from_graph(graph, depot, inst.capacity)
            id_to_int = {nid: i for i, nid in enumerate(int_to_id)}
            nv = len(problem.capacities)
            ref = cache.get((name, N))

            # ---- repair-only (deterministic) ----
            ro = VRPSolution(problem, {}, [_k_max_fqs(len(problem.dests), nv)] * nv).solution
            ro_ls = local_search(problem, ro)
            rows.append(dict(instance=name, N=N, seed=-1, method="repair_only",
                             feas0=full_feasible(problem, ro), cost0=solution_distance(problem, ro),
                             feas1=full_feasible(problem, ro_ls), cost1=solution_distance(problem, ro_ls),
                             ref=ref))

            # ---- GNN pipeline (per seed) ----
            for seed in seeds:
                cfg = SolverConfig("sa", 200, 200, seed=seed)
                g2, _ = graph_from_instance(inst)
                coarsener = GNNCoarsener(g2, P=0.5, radiusCoeff=0.5, depot_id=depot,
                                         model=model, device="cpu")
                coarsened, _ = coarsener.coarsen()
                routes_str, m, _ = solve_on_graph(coarsened, depot, inst.capacity, "FQS", cfg,
                                                  penalty_mode="adaptive", coarsener=coarsener)
                ri = to_int_routes(routes_str, id_to_int, depot)
                ri_ls = local_search(problem, ri)
                rows.append(dict(instance=name, N=N, seed=seed, method="gnn",
                                 feas0=bool(m["is_feasible"]), cost0=solution_distance(problem, ri),
                                 feas1=full_feasible(problem, ri_ls), cost1=solution_distance(problem, ri_ls),
                                 ref=ref))

    df = pd.DataFrame(rows)
    df["gap0"] = np.where(df.feas0 & (df.ref > 0), (df.cost0 - df.ref) / df.ref, np.nan)
    df["gap1"] = np.where(df.feas1 & (df.ref > 0), (df.cost1 - df.ref) / df.ref, np.nan)
    df.to_csv(ROOT / "results" / "local_search.csv", index=False)

    print("=== mean gap to OR-Tools (%), before -> after local search ===")
    print(f"  {'N':>3}  {'repair-only':>22}  {'GNN pipeline':>22}")
    for N in Ns:
        ro = df[(df.N == N) & (df.method == "repair_only")]
        gn = df[(df.N == N) & (df.method == "gnn")]
        print(f"  {N:>3}  {ro.gap0.mean()*100:7.1f} -> {ro.gap1.mean()*100:6.1f}"
              f"          {gn.gap0.mean()*100:7.1f} -> {gn.gap1.mean()*100:6.1f}")

    print("\n=== head-to-head after LS (per instance,N: GNN mean vs repair-only) ===")
    g = (df[df.method == "gnn"].groupby(["instance", "N"]).gap1.mean().rename("gnn"))
    r = (df[df.method == "repair_only"].set_index(["instance", "N"]).gap1.rename("ro"))
    hh = pd.concat([g, r], axis=1).dropna()
    print(f"  n={len(hh)}  GNN+LS lower on {(hh.gnn < hh.ro).sum()}/{len(hh)}"
          f"  mean gnn={hh.gnn.mean()*100:.1f}% ro={hh.ro.mean()*100:.1f}%")
    if len(hh) > 3 and not (hh.gnn.values == hh.ro.values).all():
        _, p = wilcoxon(hh.ro, hh.gnn, alternative="greater")
        print(f"  Wilcoxon (repair-only gap > GNN gap): p={p:.3f}")


if __name__ == "__main__":
    main()
