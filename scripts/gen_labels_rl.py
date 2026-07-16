"""Phase-4 expert-iteration label generation (downstream-reward).

For each training instance/N, build a POOL of coarsening rollouts:
  * heuristic rollouts (a hyperparameter sweep), and
  * the current GNN's own STOCHASTIC self-rollouts (score noise).
Evaluate each rollout end-to-end (coarsen -> adaptive QUBO -> SA -> inflate ->
metrics) and score it with a downstream REWARD combining post/pre-repair
feasibility and the optimality gap vs an OR-Tools reference. The highest-reward
rollout becomes the oracle; its per-level merges are the labels.

Because the GNN's own rollouts are in the pool, when one beats every heuristic
setting the GNN is trained on its OWN best behaviour -> it can exceed imitation
(the fix for "GNN only ties the heuristic"). This is expert iteration / DAgger.

Usage:
  python scripts/gen_labels_rl.py --model results/gnn_model_multiN.pt \
      --Ns 10 15 20 --n_gnn 4 --out results/gnn_labels_rl.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch        # noqa: E402

from src.baselines import solve_cvrptw_ortools                                   # noqa: E402
from src.coarsening import SpatioTemporalGraphCoarsener, graph_from_instance     # noqa: E402
from src.datasets import load_solomon, train_test_split_instances               # noqa: E402
from src.experiment import solve_on_graph                                       # noqa: E402
from src.gnn_coarsening import GNNCoarsener, build_features, load_model          # noqa: E402
from src.solvers import SolverConfig                                            # noqa: E402

# heuristic sweep (as in gen_labels) — guarantees the oracle is >= best heuristic
SWEEP = [
    {"alpha": 1.0, "beta": 1.0, "P": 0.5, "radiusCoeff": 2.0},
    {"alpha": 1.0, "beta": 0.6, "P": 0.4, "radiusCoeff": 0.5},
    {"alpha": 0.7, "beta": 0.8, "P": 0.7, "radiusCoeff": 2.0},
    {"alpha": 1.0, "beta": 1.0, "P": 0.5, "radiusCoeff": 0.5},
    {"alpha": 0.8, "beta": 0.6, "P": 0.5, "radiusCoeff": 1.0},
]
GNN_P, GNN_RC = 0.5, 0.5


def reward(m, ref_cost) -> float:
    """Downstream reward: feasibility dominates; then optimality gap; then raw viol."""
    r = 2.0 * float(m["is_feasible"]) + 1.0 * float(m["pre_feasible"])
    if m["is_feasible"] and ref_cost and ref_cost > 0:
        gap = max(0.0, (m["total_distance"] - ref_cost) / ref_cost)
        r -= 0.5 * min(gap, 2.0)
    r -= 0.02 * min(m["pre_total_viol"], 50)
    return r


def eval_rollout(inst, coarsener, cfg):
    """Run one rollout end-to-end; return (metrics, level_snapshots)."""
    coarsened, _ = coarsener.coarsen(record_levels=True)
    _, m, _ = solve_on_graph(coarsened, coarsener.depot_id, inst.capacity, "FQS",
                             cfg, penalty_mode="adaptive", coarsener=coarsener)
    return m, coarsener.level_snapshots


def build_pool(inst, model, device, n_gnn, rng):
    """Yield (tag, coarsener) rollouts on fresh graphs."""
    for p in SWEEP:
        g, d = graph_from_instance(inst)
        yield "heur", SpatioTemporalGraphCoarsener(graph=g, depot_id=d, **p)
    # deterministic GNN
    g, d = graph_from_instance(inst)
    yield "gnn", GNNCoarsener(g, P=GNN_P, radiusCoeff=GNN_RC, depot_id=d,
                              model=model, device=device, explore=0.0)
    # stochastic GNN self-rollouts
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="results/gnn_model_multiN.pt")
    ap.add_argument("--Ns", type=int, nargs="+", default=[10, 15, 20])
    ap.add_argument("--n_gnn", type=int, default=4)
    ap.add_argument("--num_reads", type=int, default=200)
    ap.add_argument("--num_sweeps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ortools_time", type=float, default=2.0)
    ap.add_argument("--out", type=str, default="results/gnn_labels_rl.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ROOT / args.model, device=device)
    train, test = train_test_split_instances()
    rng = np.random.default_rng(args.seed)
    cfg = SolverConfig("sa", args.num_reads, args.num_sweeps, seed=args.seed)

    print(f"Expert-iteration labeling: {len(train)} train x {args.Ns} Ns, "
          f"pool = {len(SWEEP)} heuristic + {args.n_gnn+1} gnn per instance", flush=True)

    examples = []
    gnn_wins = 0
    total = 0
    for N in args.Ns:
        for k, name in enumerate(train):
            inst = load_solomon(name).truncate(N)
            ref = solve_cvrptw_ortools(inst, time_limit_s=args.ortools_time)
            ref_cost = ref.cost if ref.feasible else None
            best = None  # (reward, tag, snapshots, metrics)
            for tag, co in build_pool(inst, model, device, args.n_gnn, rng):
                m, snaps = eval_rollout(inst, co, cfg)
                r = reward(m, ref_cost)
                if best is None or r > best[0]:
                    best = (r, tag, snaps, m)
            total += 1
            gnn_wins += int(best[1] == "gnn")
            examples.extend(label_from_snapshots(inst, best[2]))
            if (k + 1) % 10 == 0:
                print(f"  N={N} [{k+1}/{len(train)}] oracle={best[1]} "
                      f"reward={best[0]:.2f} feas={best[3]['is_feasible']} "
                      f"gnn_wins={gnn_wins}/{total}", flush=True)

    out = ROOT / args.out
    torch.save({"examples": examples, "Ns": args.Ns,
                "train_instances": train, "test_instances": test,
                "gnn_win_rate": gnn_wins / max(1, total)}, out)
    print(f"\nWrote {len(examples)} labelled graphs -> {out}")
    print(f"GNN rollouts won the oracle slot on {gnn_wins}/{total} "
          f"({100*gnn_wins/max(1,total):.0f}%) instance-Ns "
          f"(self-improvement signal)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
