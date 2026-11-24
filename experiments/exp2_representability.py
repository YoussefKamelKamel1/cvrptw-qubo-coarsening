"""Whether the coarsened problem can still represent the OR-Tools solution."""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NS = (10, 15, 20, 40, 60, 100)


def _segments(routes):
    """All consecutive pairs (ordered) served by OR-Tools."""
    nxt = set()
    for r in routes:
        for a, b in zip(r, r[1:]):
            nxt.add((a, b))
    return nxt


def _contiguous(seq, routes):
    """'exact', 'reversed' or 'broken' for one merged group."""
    for r in routes:
        n = len(seq)
        for i in range(len(r) - n + 1):
            win = r[i:i + n]
            if win == seq:
                return "exact"
            if win == seq[::-1]:
                return "reversed"
    return "broken"


def _one(args):
    name, N, method = args
    import harness
    from src.baselines import solve_cvrptw_ortools
    from src.coarsening import graph_from_instance, SpatioTemporalGraphCoarsener
    from src.datasets import load_solomon, SOLOMON_FAMILY_HYPERPARAMS
    from src.gnn_coarsening import GNNCoarsener

    full = load_solomon(name)
    inst = full.truncate(min(N, full.n_customers))
    sol = solve_cvrptw_ortools(inst, time_limit_s=3.0)
    if not sol.feasible:
        return None
    routes = [[str(c) for c in r] for r in sol.routes]

    g, dep = graph_from_instance(inst)
    assert {c for r in routes for c in r} == {n for n in g.nodes if n != dep}
    if method == "gnn":
        c = GNNCoarsener(g, P=harness.GNN_FIXED_P, radiusCoeff=harness.GNN_FIXED_RC,
                         depot_id=dep, model=harness._model(), device="cpu",
                         stop_threshold=None)
    else:
        c = SpatioTemporalGraphCoarsener(graph=g, depot_id=dep,
                                         **SOLOMON_FAMILY_HYPERPARAMS[inst.family])
    cg, _ = c.coarsen()

    groups = []
    for sn in cg.nodes:
        if sn == dep:
            continue
        seq = c.inflate_route([[sn]])[0]
        if len(seq) >= 2:
            groups.append(seq)

    nxt = _segments(routes)
    adj_any = nxt | {(b, a) for a, b in nxt}
    pairs = [(a, b) for s in groups for a, b in zip(s, s[1:])]
    kinds = [_contiguous(s, routes) for s in groups]
    n_c = len({c for r in routes for c in r})
    n_adj = sum(len(r) - 1 for r in routes)
    return dict(instance=name, N=N, method=method, groups=len(groups),
                merged_customers=sum(len(s) for s in groups),
                exact=kinds.count("exact"), reversed=kinds.count("reversed"),
                broken=kinds.count("broken"),
                representable=int(all(k == "exact" for k in kinds)),
                pairs=len(pairs),
                pair_exact=sum(p in nxt for p in pairs),
                pair_any=sum(p in adj_any for p in pairs),
                chance=n_adj / (n_c * (n_c - 1) / 2))


def main():
    import pandas as pd
    from harness import OUT
    from src.datasets import train_test_split_instances
    _, test = train_test_split_instances()
    jobs = [(n, N, m) for m in ("gnn", "heuristic") for N in NS for n in test]
    with ProcessPoolExecutor(min(8, os.cpu_count())) as ex:
        rows = [r for r in ex.map(_one, jobs) if r]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp2_representability.csv", index=False)

    g = df.groupby(["method", "N"]).agg(
        groups=("groups", "sum"), exact=("exact", "sum"), reversed=("reversed", "sum"),
        broken=("broken", "sum"), pairs=("pairs", "sum"), pair_exact=("pair_exact", "sum"),
        pair_any=("pair_any", "sum"), representable=("representable", "mean"),
        chance=("chance", "mean"))
    g["exact%"] = 100 * g.exact / g.groups
    g["broken%"] = 100 * g.broken / g.groups
    g["pair_prec%"] = 100 * g.pair_any / g.pairs
    g["chance%"] = 100 * g.chance
    g["repr%"] = 100 * g.representable
    print(g[["groups", "exact%", "broken%", "pair_prec%", "chance%", "repr%"]]
          .round(1).to_string())


if __name__ == "__main__":
    main()
