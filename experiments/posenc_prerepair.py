"""Pre-repair feasibility of the position encoding with GNN coarsening at N = 10, 15, 20.

  python experiments/posenc_prerepair.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from multiprocessing import get_context  # noqa: E402


def task(t):
    import torch
    torch.set_num_threads(1)
    from gnn_ensemble import load_any
    from review_runner import load, make_class
    from src.coarsening import graph_from_instance
    from src.gnn_coarsening import GNNCoarsener
    from v3_common import run_candidate
    name, N = t
    model = load_any(str(HERE.parent / "results" / "gnn_model_rl.pt"), device="cpu")
    inst = load(name, "solomon", "fixed").truncate(N)
    out = []
    for seed in (0, 1, 2):
        g, d = graph_from_instance(inst)
        co = make_class(GNNCoarsener, "stored", "rank")(g, P=0.5, radiusCoeff=0.5, depot_id=d,
                                                       model=model, device="cpu")
        coarse, _ = co.coarsen()
        r = run_candidate(inst, co, coarse, seed)
        out.append(dict(instance=name, N=N, seed=seed, **r))
    return out


if __name__ == "__main__":
    from src.datasets import train_test_split_instances
    ts = [(n, N) for N in (10, 15, 20) for n in train_test_split_instances()[1]]
    with get_context("spawn").Pool(4) as pool:
        rows = [r for rs in pool.map(task, ts) for r in rs]
    (HERE / "results" / "posenc_prerepair.json").write_text(json.dumps(rows))
    for N in (10, 15, 20):
        rr = [r for r in rows if r["N"] == N]
        print(N, len(rr), "pre", sum(r["pre_feasible"] for r in rr) / len(rr),
              "final", sum(r["final_feasible"] for r in rr) / len(rr),
              "vars", sum(r["num_qubo_vars"] for r in rr) / len(rr))
