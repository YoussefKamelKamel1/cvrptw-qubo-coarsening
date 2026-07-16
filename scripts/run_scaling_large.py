"""Large-N tractability curve + coarsened solve quality at N=50/60.

Part A (fast, no solving): QUBO variable count vs N up to 100 for none/heuristic/
GNN -- the explosion curve that motivates coarsening (and QPU embeddability).
Part B: actually solve the coarsened pipeline (GNN, heuristic) at N=50/60, 3 seeds,
to confirm the feasibility-preserving GNN edge persists at larger N.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import train_test_split_instances, load_solomon    # noqa: E402
from src.coarsening import graph_from_instance, SpatioTemporalGraphCoarsener  # noqa: E402
from src.datasets import SOLOMON_FAMILY_HYPERPARAMS                   # noqa: E402
from src.qubo import vrp_problem_from_graph                          # noqa: E402
from src.experiment import _count_vars                               # noqa: E402
from src.gnn_coarsening import GNNCoarsener, load_model              # noqa: E402
from src.runner_parallel import build_jobs, run_grid, HEUR_FIXED     # noqa: E402

MODEL = str(ROOT / "results" / "gnn_model_rl.pt")


def count_vars(inst, depot_graph, depot, mode, model):
    graph, _ = graph_from_instance(inst)
    coarsener = None
    if mode == "heuristic":
        coarsener = SpatioTemporalGraphCoarsener(
            graph=graph, depot_id=depot, **SOLOMON_FAMILY_HYPERPARAMS[inst.family])
        graph, _ = coarsener.coarsen()
    elif mode == "gnn":
        coarsener = GNNCoarsener(graph, P=0.5, radiusCoeff=0.5, depot_id=depot,
                                 model=model, device="cpu")
        graph, _ = coarsener.coarsen()
    problem, _ = vrp_problem_from_graph(graph, depot, inst.capacity)
    return _count_vars(problem, "FQS", smart_capacity=True)


def main():
    import numpy as np
    import pandas as pd

    _, test = train_test_split_instances()
    model = load_model(MODEL, device="cpu")

    # ---- Part A: variable-count explosion curve (no solving) ----
    print("=== QUBO variables vs N (mean over 13 test instances) ===")
    print(f"  {'N':>4} {'none':>8} {'heuristic':>10} {'gnn':>8}  {'none/gnn':>8}")
    curveN = [10, 20, 30, 40, 50, 60, 80, 100]
    crows = []
    for N in curveN:
        vals = {"none": [], "heuristic": [], "gnn": []}
        for name in test:
            try:
                inst = load_solomon(name).truncate(N)
            except ValueError:
                continue  # instance has fewer than N customers
            g, depot = graph_from_instance(inst)
            for mode in vals:
                vals[mode].append(count_vars(inst, g, depot, mode, model))
        m = {k: float(np.mean(v)) for k, v in vals.items()}
        crows.append(dict(N=N, **m))
        print(f"  {N:>4} {m['none']:>8.0f} {m['heuristic']:>10.0f} {m['gnn']:>8.0f}  "
              f"{m['none']/max(1,m['gnn']):>7.1f}x")
    pd.DataFrame(crows).to_csv(ROOT / "results" / "scaling_large_vars.csv", index=False)

    # ---- Part B: solve coarsened pipeline at N=50/60 ----
    Ns = [50, 60]
    seeds = [0, 1, 2]
    conditions = ["heuristic", "heuristic_fixed", "gnn"]
    nw = min(8, os.cpu_count())
    jobs = build_jobs(test, Ns, seeds, conditions, penalty_mode="adaptive",
                      num_reads=200, num_sweeps=200, model_path=MODEL)
    print(f"\nSolving coarsened at N=50/60: {len(jobs)} jobs", flush=True)
    t = time.perf_counter()
    rows = run_grid(jobs, n_workers=nw, progress_every=100)
    print(f"done in {time.perf_counter()-t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "scaling_large_solve.csv", index=False)

    print("\n=== coarsened feasibility %, pre-viol, vars at N=50/60 ===")
    for N in Ns:
        for c in conditions:
            s = df[(df.N == N) & (df.condition == c)]
            print(f"  N={N} {c:16s} feas={100*s.is_feasible.mean():5.1f}%  "
                  f"pre_viol={s.pre_total_viol.mean():6.1f}  vars={s.num_qubo_vars.mean():5.0f}")

    from scipy.stats import wilcoxon
    print("\n=== GNN vs tuned heuristic feasibility (paired, N=50/60 pooled) ===")
    piv = (df.groupby(["instance", "N", "condition"])["is_feasible"].mean()
           .unstack("condition").dropna(subset=["gnn", "heuristic"]))
    x, y = piv["gnn"].to_numpy(), piv["heuristic"].to_numpy()
    if not (x == y).all():
        _, p = wilcoxon(x, y, alternative="greater")
        print(f"  gnn={x.mean()*100:.1f}% heur={y.mean()*100:.1f}%  "
              f"GNN>=heur on {int((x>=y).sum())}/{len(x)} (strictly {int((x>y).sum())})  p={p:.3f}")


if __name__ == "__main__":
    main()
