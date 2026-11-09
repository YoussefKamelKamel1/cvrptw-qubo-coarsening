"""Expert-iteration labelling that caches EVERY candidate, not just the winner.

The reference `gen_labels_rl.py` scores every rollout and keeps only the highest
reward, so changing the reward weights means repeating the whole expensive pass
(one full pipeline run per candidate, plus an OR-Tools reference per instance).

A candidate's measured outcome does not depend on the reward weights; the weights
only decide which candidate wins. So we run the expensive part once and store each
candidate's metrics alongside the labels it would produce. `sweep_reward.py` can
then re-score offline and rebuild any label set in seconds.

The reference implementation is untouched: this file duplicates its pool, rollout
and labelling logic so the reference pipeline stays reproducible as-is.

  python scripts/gen_labels_cached.py --Ns 10 15 20 --out results/label_cache.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

from src.baselines import solve_cvrptw_ortools                         # noqa: E402
from src.coarsening import (SpatioTemporalGraphCoarsener,              # noqa: E402
                            graph_from_instance)
from src.datasets import load_solomon, train_test_split_instances      # noqa: E402
from src.experiment import solve_on_graph                              # noqa: E402
from src.gnn_coarsening import (GNNCoarsener, build_features,          # noqa: E402
                                load_model)
from src.solvers import SolverConfig                                   # noqa: E402

# identical to the reference sweep, so the candidate pool is the same
SWEEP = [
    {"alpha": 1.0, "beta": 1.0, "P": 0.5, "radiusCoeff": 2.0},
    {"alpha": 1.0, "beta": 0.6, "P": 0.4, "radiusCoeff": 0.5},
    {"alpha": 0.7, "beta": 0.8, "P": 0.7, "radiusCoeff": 2.0},
    {"alpha": 1.0, "beta": 1.0, "P": 0.5, "radiusCoeff": 0.5},
    {"alpha": 0.8, "beta": 0.6, "P": 0.5, "radiusCoeff": 1.0},
]
GNN_P, GNN_RC = 0.5, 0.5


def eval_rollout(inst, coarsener, cfg):
    coarsened, _ = coarsener.coarsen(record_levels=True)
    _, m, _ = solve_on_graph(coarsened, coarsener.depot_id, inst.capacity, "FQS",
                             cfg, penalty_mode="adaptive", coarsener=coarsener)
    return m, coarsener.level_snapshots


def build_pool(inst, model, device, n_gnn, rng):
    for p in SWEEP:
        g, d = graph_from_instance(inst)
        yield "heur", SpatioTemporalGraphCoarsener(graph=g, depot_id=d, **p)
    g, d = graph_from_instance(inst)
    yield "gnn", GNNCoarsener(g, P=GNN_P, radiusCoeff=GNN_RC, depot_id=d,
                              model=model, device=device, explore=0.0)
    for _ in range(n_gnn):
        g, d = graph_from_instance(inst)
        sub = np.random.default_rng(rng.integers(1 << 30))
        yield "gnn", GNNCoarsener(g, P=GNN_P, radiusCoeff=GNN_RC, depot_id=d,
                                  model=model, device=device, explore=0.35, rng=sub)


def label_from_snapshots(inst, snapshots):
    g, depot = graph_from_instance(inst)
    examples = []
    for level, (gsnap, merges) in enumerate(snapshots):
        fb = build_features(gsnap, depot)
        if fb.cand_edge_index.numel() == 0:
            continue
        labels = torch.tensor(
            [1.0 if frozenset((e.u_id, e.v_id)) in merges else 0.0
             for e in fb.cand_edges], dtype=torch.float32)
        examples.append({
            "instance": inst.name, "level": level,
            "node_feat": fb.node_feat, "mp_edge_index": fb.mp_edge_index,
            "cand_edge_index": fb.cand_edge_index, "cand_edge_feat": fb.cand_edge_feat,
            "labels": labels,
        })
    return examples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="results/gnn_model_multiN.pt")
    ap.add_argument("--Ns", type=int, nargs="+", default=[10, 15, 20])
    ap.add_argument("--n_gnn", type=int, default=4)
    ap.add_argument("--num_reads", type=int, default=200)
    ap.add_argument("--num_sweeps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ortools_time", type=float, default=2.0)
    ap.add_argument("--max-instances", type=int, default=None,
                    help="cap the training instances (for a quick check)")
    ap.add_argument("--out", default="results/label_cache.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ROOT / args.model, device=device)
    train, test = train_test_split_instances()
    if args.max_instances:
        train = train[:args.max_instances]
    rng = np.random.default_rng(args.seed)
    cfg = SolverConfig("sa", args.num_reads, args.num_sweeps, seed=args.seed)

    print(f"Caching every candidate: {len(train)} train instances x {args.Ns} Ns, "
          f"pool = {len(SWEEP)} heuristic + {args.n_gnn+1} gnn each", flush=True)

    cache, t0 = [], time.perf_counter()
    for N in args.Ns:
        for k, name in enumerate(train):
            inst = load_solomon(name).truncate(N)
            ref = solve_cvrptw_ortools(inst, time_limit_s=args.ortools_time)
            ref_cost = ref.cost if ref.feasible else None
            cands = []
            for tag, co in build_pool(inst, model, device, args.n_gnn, rng):
                m, snaps = eval_rollout(inst, co, cfg)
                cands.append({
                    "tag": tag,
                    # everything the reward could possibly use
                    "metrics": {kk: m[kk] for kk in
                                ("is_feasible", "pre_feasible", "pre_total_viol",
                                 "total_distance")},
                    "examples": label_from_snapshots(inst, snaps),
                })
            cache.append({"instance": name, "N": N, "ref_cost": ref_cost,
                          "candidates": cands})
            if (k + 1) % 10 == 0:
                print(f"  N={N} [{k+1}/{len(train)}] "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cache": cache, "Ns": args.Ns, "train_instances": train,
                "test_instances": test, "n_gnn": args.n_gnn,
                "solver": {"num_reads": args.num_reads,
                           "num_sweeps": args.num_sweeps, "seed": args.seed}}, out)
    n_cand = sum(len(c["candidates"]) for c in cache)
    print(f"\nCached {n_cand} candidates over {len(cache)} instance-size pairs "
          f"in {time.perf_counter()-t0:.0f}s")
    print(f"-> {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
