"""Does the GraphSAGE trunk matter? Compare against GAT and GCN.

Tests whether the chosen architecture has an advantage over the alternatives.
Everything is held fixed except the message-passing layer: same
features, same expert-iteration labels, same optimiser, epochs, seed and validation
split. Each trained model is then evaluated the way it is actually used, by running
the full coarsening pipeline on held-out instances, because edge-ranking accuracy
and downstream feasibility are not the same thing.

Does not modify any existing module: the reference training script and model stay
as they are, and the variants subclass the reference scorer.

  python scripts/compare_architectures.py --epochs 300 --Ns 10 20 40
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
import torch.nn as nn                                                # noqa: E402
from torch_geometric.nn import GATConv, GCNConv                      # noqa: E402

from src.gnn_coarsening import MergeScorerGNN, GNNCoarsener          # noqa: E402
from src.datasets import load_solomon, train_test_split_instances    # noqa: E402
from src.coarsening import graph_from_instance                       # noqa: E402
from src.qubo import vrp_problem_from_graph, adaptive_penalties      # noqa: E402
from src.solvers import (VRPSolution, SolverConfig, solve_qubo,      # noqa: E402
                         _k_max_fqs)
from src.experiment import _routes_to_strings                        # noqa: E402
from src.metrics import calculate_route_metrics                      # noqa: E402
from src.utils import set_global_seed                                # noqa: E402

ARCHS = ["sage", "gat", "gcn"]


class VariantScorer(MergeScorerGNN):
    """Reference scorer with the message-passing layer swapped out."""

    def __init__(self, arch: str, hidden: int = 64, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__(hidden=hidden, num_layers=num_layers, dropout=dropout)
        if arch == "gat":
            # heads averaged (concat=False) so the width matches the other trunks
            self.convs = nn.ModuleList(
                [GATConv(hidden, hidden, heads=4, concat=False)
                 for _ in range(num_layers)])
        elif arch == "gcn":
            self.convs = nn.ModuleList(
                [GCNConv(hidden, hidden) for _ in range(num_layers)])
        elif arch != "sage":
            raise ValueError(arch)
        self.arch = arch


# --------------------------------------------------------------------------- #
# training (mirrors scripts/train_gnn.py so the comparison is like for like)
# --------------------------------------------------------------------------- #
def _to_device(ex, device):
    return (ex["node_feat"].to(device), ex["mp_edge_index"].to(device),
            ex["cand_edge_index"].to(device), ex["cand_edge_feat"].to(device),
            ex["labels"].to(device))


def _auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _recall_at_k(logits, labels):
    k = int(labels.sum().item())
    if k == 0:
        return float("nan")
    topk = torch.topk(logits, min(k, logits.numel())).indices
    return float(labels[topk].sum().item() / k)


@torch.no_grad()
def rank_eval(model, examples, device):
    model.eval()
    s, l, recs = [], [], []
    for ex in examples:
        nf, mp, ci, cf, y = _to_device(ex, device)
        if ci.numel() == 0:
            continue
        logits = model(nf, mp, ci, cf)
        s.append(logits.cpu().numpy())
        l.append(y.cpu().numpy())
        recs.append(_recall_at_k(logits, y))
    if not s:
        return {"auc": float("nan"), "recall_at_k": float("nan")}
    return {"auc": _auc(np.concatenate(s), np.concatenate(l)),
            "recall_at_k": float(np.nanmean(recs))}


def train_one(arch, examples, args, device):
    set_global_seed(args.seed)
    rng = random.Random(args.seed)
    instances = sorted({e["instance"] for e in examples})
    rng.shuffle(instances)
    val_inst = set(instances[:max(1, int(len(instances) * args.val_frac))])
    train_ex = [e for e in examples if e["instance"] not in val_inst]
    val_ex = [e for e in examples if e["instance"] in val_inst]

    pos = sum(int(e["labels"].sum()) for e in train_ex)
    tot = sum(int(e["labels"].numel()) for e in train_ex)
    pos_weight = torch.tensor([(tot - pos) / max(1, pos)], device=device)

    model = VariantScorer(arch, hidden=args.hidden, num_layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best, best_state = -1.0, None
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_ex)
        for ex in train_ex:
            nf, mp, ci, cf, y = _to_device(ex, device)
            if ci.numel() == 0:
                continue
            opt.zero_grad()
            loss = loss_fn(model(nf, mp, ci, cf), y)
            loss.backward()
            opt.step()
        if epoch % 25 == 0 or epoch == 1:
            va = rank_eval(model, val_ex or train_ex, device)
            score = va["recall_at_k"] if va["recall_at_k"] == va["recall_at_k"] else -1
            if score > best:
                best, best_state = score, {k: v.cpu().clone()
                                           for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    va = rank_eval(model, val_ex or train_ex, device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  [{arch}] trained in {time.perf_counter()-t0:.0f}s  "
          f"params={n_par}  val AUC={va['auc']:.3f}  recall@k={va['recall_at_k']:.3f}",
          flush=True)
    return model, dict(val_auc=va["auc"], val_recall=va["recall_at_k"], params=n_par)


# --------------------------------------------------------------------------- #
# downstream: run the actual pipeline with each trained scorer
# --------------------------------------------------------------------------- #
def downstream(model, name, N, device, seed, reads, sweeps):
    inst = load_solomon(name).truncate(N)
    graph, depot = graph_from_instance(inst)
    coarsener = GNNCoarsener(graph, P=0.5, radiusCoeff=0.5, depot_id=depot,
                             model=model, device=device)
    cgraph, _ = coarsener.coarsen()
    problem, int_to_id = vrp_problem_from_graph(cgraph, depot, inst.capacity)
    nv = len(problem.capacities)
    k = [_k_max_fqs(len(problem.dests), nv)] * nv
    qubo = problem.get_qubo(k, adaptive_penalties(problem), smart_capacity=True)
    n_vars = len({v for key in qubo.dict for v in key})

    sample = (solve_qubo(qubo, SolverConfig("sa", reads, sweeps, seed)) or [{}])[0]
    sol = VRPSolution(problem, sample, k)
    pre = sol.report(sol.raw_solution)
    routes = coarsener.inflate_route(_routes_to_strings(sol.solution, int_to_id, depot))
    m = calculate_route_metrics(coarsener.graph, routes, depot, inst.capacity)
    return dict(n_vars=n_vars, pre_viol=float(pre["total_violations"]),
                pre_feasible=bool(pre["feasible"]),
                feasible=bool(m["is_feasible"]),
                distance=float(m["total_distance"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="results/gnn_labels_rl.pt")
    ap.add_argument("--archs", nargs="+", default=ARCHS, choices=ARCHS)
    ap.add_argument("--Ns", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=None,
                    help="train each architecture once per seed and report the "
                         "spread; a single training run cannot separate an "
                         "architecture effect from initialisation noise")
    ap.add_argument("--num-reads", type=int, default=200)
    ap.add_argument("--num-sweeps", type=int, default=200)
    ap.add_argument("--out", default="results/architecture_comparison.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(ROOT / args.labels, weights_only=False)
    examples = data["examples"]
    _, test = train_test_split_instances()
    print(f"device={device} | {len(examples)} labelled graphs | "
          f"{len(test)} held-out instances | archs={args.archs}\n", flush=True)

    train_seeds = args.train_seeds or [args.seed]
    rows, rank = [], {}
    for arch in args.archs:
        print(f"== {arch} ==", flush=True)
        for ts in train_seeds:
            targs = SimpleNamespace(epochs=args.epochs, lr=args.lr, hidden=args.hidden,
                                    layers=args.layers, val_frac=args.val_frac, seed=ts)
            model, stats = train_one(arch, examples, targs, device)
            rank[(arch, ts)] = stats
            model.eval()
            for N in args.Ns:
                for name in test:
                    for sd in args.seeds:
                        set_global_seed(sd)
                        r = downstream(model, name, N, device, sd,
                                       args.num_reads, args.num_sweeps)
                        rows.append(dict(arch=arch, train_seed=ts, N=N,
                                         instance=name, seed=sd, **r))
                sub = [r for r in rows if r["arch"] == arch
                       and r["train_seed"] == ts and r["N"] == N]
                print(f"    seed {ts} N={N:3d}  "
                      f"feasible={100*np.mean([r['feasible'] for r in sub]):5.1f}%  "
                      f"pre-feasible={100*np.mean([r['pre_feasible'] for r in sub]):5.1f}%  "
                      f"pre-viol={np.mean([r['pre_viol'] for r in sub]):5.2f}", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== downstream coarsened feasibility (%), mean over training seeds "
          f"({len(train_seeds)} seeds; +/- is the spread across them) ===")
    print(f"{'N':>4} " + " ".join(f"{a:>16}" for a in args.archs))
    for N in args.Ns:
        line = f"{N:>4} "
        for a in args.archs:
            per_seed = [100 * np.mean([r["feasible"] for r in rows
                                       if r["arch"] == a and r["train_seed"] == ts
                                       and r["N"] == N])
                        for ts in train_seeds]
            line += f"  {np.mean(per_seed):6.1f} +/-{np.std(per_seed):5.1f}"
        print(line)

    print("\n=== edge-ranking quality (validation) ===")
    for a in args.archs:
        aucs = [rank[(a, ts)]["val_auc"] for ts in train_seeds]
        recs = [rank[(a, ts)]["val_recall"] for ts in train_seeds]
        print(f"  {a:>6}: AUC={np.nanmean(aucs):.3f}+/-{np.nanstd(aucs):.3f}  "
              f"recall@k={np.nanmean(recs):.3f}+/-{np.nanstd(recs):.3f}  "
              f"params={rank[(a, train_seeds[0])]['params']}")

    if len(train_seeds) > 1:
        print("\n=== is the gap larger than the seed-to-seed spread? ===")
        for N in args.Ns:
            best = None
            for a in args.archs:
                ps = [100 * np.mean([r["feasible"] for r in rows
                                     if r["arch"] == a and r["train_seed"] == ts
                                     and r["N"] == N]) for ts in train_seeds]
                if best is None or np.mean(ps) > best[1]:
                    best = (a, np.mean(ps), np.std(ps))
            others = [(a, np.mean([100 * np.mean([r["feasible"] for r in rows
                                                  if r["arch"] == a
                                                  and r["train_seed"] == ts
                                                  and r["N"] == N])
                                   for ts in train_seeds]))
                      for a in args.archs if a != best[0]]
            margin = best[1] - max(o[1] for o in others)
            verdict = "clear" if margin > 2 * best[2] else "within noise"
            print(f"  N={N:3d}: best={best[0]} by {margin:.1f} pp, "
                  f"seed spread {best[2]:.1f} pp -> {verdict}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
