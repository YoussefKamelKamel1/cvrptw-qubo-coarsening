"""Selection among candidate coarsenings (Supplementary Note 1): candidates, features
and pipeline run.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

GH400_DIR = HERE / "data_gh400" / "extracted"
VAL = {"solomon": ["C101", "C201", "R101", "R201", "RC101", "RC201"],
       "gh400": ["C1_4_1", "C2_4_1", "R1_4_1", "R2_4_1", "RC1_4_1", "RC2_4_1"],
       "euro": None}   # filled from train40 (first 6)
HEUR = [dict(alpha=1.0, beta=1.0, P=p, radiusCoeff=rc) for p in (0.4, 0.5, 0.6, 0.7) for rc in (0.5, 1.0)]
GNN_P = (0.4, 0.5, 0.6, 0.7)
N_NOISY = 2
FIXED_KEY = "heur_P0.5_rc0.5"       # heuristic-fixed


def gh400_names():
    return sorted(p.stem.upper() for p in GH400_DIR.glob("*.TXT"))


def euro_train_names():
    return [l.strip()[:-4] for l in (HERE / "data_euro" / "train40.txt").read_text().splitlines() if l.strip()]


def load_instance(dataset, name):
    if dataset == "gh400":
        import numpy as np
        from src.datasets import VRPInstance
        lines = (GH400_DIR / f"{name}.TXT").read_text().splitlines()
        nv, cap = [float(x) for x in lines[4].split()]
        a = np.array([[float(x) for x in ln.split()] for ln in lines[9:] if len(ln.split()) == 7])
        assert a[0, 0] == 0 and (np.diff(a[:, 0]) == 1).all(), name
        return VRPInstance(name=name, coords=a[:, 1:3], demand=a[:, 3], capacity=cap,
                           n_vehicles=int(nv), ready_time=a[:, 4], due_time=a[:, 5],
                           service_time=a[:, 6], family=name.split("_")[0],
                           meta={"source": "gehring-homberger 400"})
    from review_runner import load
    return load(name, dataset, "fixed")


def make_candidates(inst, policy, seed):
    """All candidate coarsenings of one truncated instance: list of (key, coarsener, coarse)."""
    import numpy as np
    from review_runner import make_class
    from src.coarsening import graph_from_instance, SpatioTemporalGraphCoarsener
    from src.gnn_coarsening import GNNCoarsener
    out = []
    for p in HEUR:
        g, d = graph_from_instance(inst)
        co = make_class(SpatioTemporalGraphCoarsener, "stored", "rank")(graph=g, depot_id=d, **p)
        coarse, _ = co.coarsen()
        out.append((f"heur_P{p['P']}_rc{p['radiusCoeff']}", co, coarse))
    rng = np.random.default_rng(seed)
    variants = [(f"gnn_P{P}", P, 0.0) for P in GNN_P] + [(f"gnnnoisy{k}", 0.5, 0.35) for k in range(N_NOISY)]
    for key, P, ex in variants:
        g, d = graph_from_instance(inst)
        co = make_class(GNNCoarsener, "stored", "rank")(
            g, P=P, radiusCoeff=0.5, depot_id=d, model=policy, device="cpu", explore=ex,
            rng=np.random.default_rng(rng.integers(1 << 30)))
        coarse, _ = co.coarsen()
        out.append((key, co, coarse))
    return out


def qubo_vars(coarse, depot, capacity):
    from src.experiment import _count_vars
    from src.qubo import vrp_problem_from_graph
    prob, _ = vrp_problem_from_graph(coarse, depot, capacity)
    return _count_vars(prob, "FQS", smart_capacity=True)


def run_candidate(inst, co, coarse, seed):
    """The paper's pipeline on a given coarsening (as in experiments/rev_runner.py)."""
    from review_runner import _repair_on_original
    from src.coarsening import graph_from_instance
    from src.experiment import solve_on_graph
    from src.metrics import calculate_route_metrics
    from src.solvers import SolverConfig
    from src.utils import set_global_seed
    set_global_seed(seed)
    routes, m, _ = solve_on_graph(coarse, co.depot_id, inst.capacity, "FQS",
                                  SolverConfig("sa", 200, 200, seed=seed),
                                  penalty_mode="adaptive", coarsener=co)
    graph, depot = graph_from_instance(inst)
    mr = calculate_route_metrics(graph, _repair_on_original(graph, depot, inst.capacity, routes),
                                 depot, inst.capacity)
    return dict(final_feasible=bool(m["is_feasible"]), pre_feasible=bool(m["pre_feasible"]),
                num_qubo_vars=int(m["num_qubo_vars"]), final_distance=float(m["total_distance"]),
                final_rep_feasible=bool(mr["is_feasible"]))


def global_feats(inst, coarse, depot, key, nvars):
    """Candidate-level features: size and depth of the coarsening, and how tight the
    merged windows are relative to the original ones."""
    import numpy as np
    nodes = [n for nid, n in coarse.nodes.items() if nid != depot]
    n0 = len(inst.coords) - 1
    sup = [n for n in nodes if n.is_super_node]
    horizon = float(inst.due_time[0])
    w_orig = float(np.mean(inst.due_time[1:] - inst.ready_time[1:])) / horizon
    w_coarse = float(np.mean([max(0.0, n.l - n.e) for n in nodes])) / horizon if nodes else 0.0
    s_sup = float(np.mean([n.s for n in sup])) / horizon if sup else 0.0
    P = float(key.split("_P")[1].split("_")[0]) if "_P" in key else 0.5
    return [len(nodes) / max(1, n0), len(sup) / max(1, len(nodes)), np.log(max(1, nvars)) / 10.0,
            P, w_orig, w_coarse, w_coarse / max(1e-6, w_orig), s_sup, np.log(n0) / 5.0,
            float(inst.capacity > 0 and inst.demand[1:].sum() > inst.capacity)]

