"""Expert-iteration labels for the improved training: three annealing runs per
candidate, six noisy rollouts, sizes 10, 15, 20 and 40.

  python experiments/gnn_v2_labels.py --policy ens:a.pt,b.pt --out experiments/results/v2/labels_r1.pt
"""
import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "scripts"))

VAL = ["C101", "C201", "R101", "R201", "RC101", "RC201"]
SIZES = [10, 15, 20, 40]
SEEDS = (0, 1, 2)
N_NOISY = 6
_POLICY = None


def task(args):
    global _POLICY
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from gen_labels_rl import SWEEP, reward, label_from_snapshots, GNN_P, GNN_RC
    from gnn_ensemble import load_spec
    from src.baselines import solve_cvrptw_ortools
    from src.coarsening import SpatioTemporalGraphCoarsener, graph_from_instance
    from src.datasets import load_solomon
    from src.experiment import solve_on_graph
    from src.gnn_coarsening import GNNCoarsener
    from src.solvers import SolverConfig
    from src.utils import set_global_seed

    name, N, policy, seed = args
    if _POLICY is None or _POLICY[0] != policy:
        _POLICY = (policy, load_spec(policy, device="cpu"))
    model = _POLICY[1]
    inst = load_solomon(name).truncate(N)
    ref = solve_cvrptw_ortools(inst, time_limit_s=2.0)
    ref_cost = ref.cost if ref.feasible else None
    rng = np.random.default_rng(seed * 100003 + N * 1009 + sum(map(ord, name)))

    def pool():
        for p in SWEEP:
            g, d = graph_from_instance(inst)
            yield "heur", SpatioTemporalGraphCoarsener(graph=g, depot_id=d, **p)
        g, d = graph_from_instance(inst)
        yield "gnn", GNNCoarsener(g, P=GNN_P, radiusCoeff=GNN_RC, depot_id=d,
                                  model=model, device="cpu", explore=0.0)
        for _ in range(N_NOISY):
            g, d = graph_from_instance(inst)
            yield "gnn", GNNCoarsener(g, P=GNN_P, radiusCoeff=GNN_RC, depot_id=d, model=model,
                                      device="cpu", explore=0.35,
                                      rng=np.random.default_rng(rng.integers(1 << 30)))

    best = None
    for tag, co in pool():
        coarse, _ = co.coarsen(record_levels=True)
        snaps = co.level_snapshots
        rs, feas = [], []
        for s in SEEDS:
            set_global_seed(s)
            _, m, _ = solve_on_graph(coarse, co.depot_id, inst.capacity, "FQS",
                                     SolverConfig("sa", 200, 200, seed=s),
                                     penalty_mode="adaptive", coarsener=co)
            rs.append(reward(m, ref_cost)); feas.append(bool(m["is_feasible"]))
        r = float(np.mean(rs))
        if best is None or r > best[0]:
            best = (r, tag, snaps, float(np.mean(feas)))
    return dict(name=name, N=N, tag=best[1], reward=best[0], feas=best[3],
                examples=label_from_snapshots(inst, best[2]))


def main():
    import torch
    from src.datasets import train_test_split_instances
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train, test = train_test_split_instances()
    names = [n for n in train if n not in VAL]
    assert not set(names) & set(test) and not set(VAL) & set(test)
    tasks = [(n, N, a.policy, a.seed) for N in sorted(SIZES, reverse=True) for n in names]
    t0 = time.time()
    res = []
    with get_context("spawn").Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(task, tasks), 1):
            res.append(r)
            if k % 20 == 0:
                print(f"  [{k}/{len(tasks)}] {time.time() - t0:.0f}s", flush=True)
    wins = sum(r["tag"] == "gnn" for r in res)
    examples = [e for r in res for e in r["examples"]]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"examples": examples, "Ns": SIZES, "train_instances": names, "val_instances": VAL,
                "test_instances": test, "gnn_win_rate": wins / len(res),
                "summary": [{k: v for k, v in r.items() if k != "examples"} for r in res]}, out)
    print(f"wrote {len(examples)} labelled graphs from {len(res)} instance-size pairs; "
          f"policy wins {wins}/{len(res)} ({100 * wins / len(res):.0f}%) in {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
