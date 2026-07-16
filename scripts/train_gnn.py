"""Phase-3 GNN training: fit the merge scorer on oracle edge labels.

Loads the labelled graphs from gen_labels.py, trains MergeScorerGNN with a
class-weighted BCE loss (positives are sparse), and reports edge ranking quality
(ROC-AUC + recall of the oracle's merges among the top-scored pairs). Saves the
trained model for the GNN coarsener.

Usage: python scripts/train_gnn.py --labels results/gnn_labels_N10.pt \
           --out results/gnn_model_N10.pt --epochs 300
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.gnn_coarsening import MergeScorerGNN, save_model  # noqa: E402
from src.utils import set_global_seed                       # noqa: E402


def _to_device(ex, device):
    return (ex["node_feat"].to(device), ex["mp_edge_index"].to(device),
            ex["cand_edge_index"].to(device), ex["cand_edge_feat"].to(device),
            ex["labels"].to(device))


def _auc(scores, labels) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if labels.sum() == 0 or labels.sum() == len(labels):
            return float("nan")
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def _recall_at_oracle_k(logits, labels) -> float:
    """Fraction of oracle positives recovered in the top-k scored edges,
    where k = number of oracle positives (mirrors how the coarsener picks the
    lowest-D edges). NaN if no positives."""
    k = int(labels.sum().item())
    if k == 0:
        return float("nan")
    topk = torch.topk(logits, k).indices
    return float(labels[topk].sum().item() / k)


def evaluate(model, examples, device) -> dict:
    model.eval()
    import numpy as np
    all_scores, all_labels, recalls = [], [], []
    with torch.no_grad():
        for ex in examples:
            nf, mp, ci, cf, y = _to_device(ex, device)
            logits = model(nf, mp, ci, cf)
            if logits.numel() == 0:
                continue
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y.cpu().numpy())
            recalls.append(_recall_at_oracle_k(logits.cpu(), y.cpu()))
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    rec = np.nanmean(recalls) if recalls else float("nan")
    return {"auc": _auc(scores, labels), "recall_at_k": rec,
            "n_edges": len(labels), "pos_rate": float(labels.mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=str, nargs="+",
                    default=["results/gnn_labels_N10.pt"])
    ap.add_argument("--out", type=str, default="results/gnn_model_N10.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    examples = []
    Ns = []
    for lf in args.labels:
        data = torch.load(ROOT / lf)
        examples.extend(data["examples"])
        Ns.append(data.get("N", data.get("Ns", "?")))
    print(f"Loaded {len(examples)} labelled graphs from N={Ns}; device={device}")

    # split by INSTANCE so val graphs are from unseen instances
    instances = sorted({e["instance"] for e in examples})
    rng = random.Random(args.seed)
    rng.shuffle(instances)
    n_val = max(1, int(len(instances) * args.val_frac))
    val_inst = set(instances[:n_val])
    train_ex = [e for e in examples if e["instance"] not in val_inst]
    val_ex = [e for e in examples if e["instance"] in val_inst]
    print(f"Train graphs: {len(train_ex)}  Val graphs: {len(val_ex)} "
          f"(val instances: {sorted(val_inst)})")

    # class imbalance -> pos_weight
    pos = sum(int(e["labels"].sum()) for e in train_ex)
    tot = sum(int(e["labels"].numel()) for e in train_ex)
    pos_weight = torch.tensor([(tot - pos) / max(1, pos)], device=device)
    print(f"Positives {pos}/{tot} (pos_rate={pos/tot:.3f}), pos_weight={pos_weight.item():.1f}")

    model = MergeScorerGNN(hidden=args.hidden, num_layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_ex)
        total = 0.0
        for ex in train_ex:
            nf, mp, ci, cf, y = _to_device(ex, device)
            if ci.numel() == 0:
                continue
            opt.zero_grad()
            logits = model(nf, mp, ci, cf)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += float(loss)
        if epoch % 25 == 0 or epoch == 1:
            tr = evaluate(model, train_ex, device)
            va = evaluate(model, val_ex, device) if val_ex else tr
            print(f"  epoch {epoch:3d} loss={total/max(1,len(train_ex)):.4f} "
                  f"train[auc={tr['auc']:.3f} rec@k={tr['recall_at_k']:.3f}] "
                  f"val[auc={va['auc']:.3f} rec@k={va['recall_at_k']:.3f}]", flush=True)
            score = va["recall_at_k"] if va["recall_at_k"] == va["recall_at_k"] else -1
            if score > best_val:
                best_val = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    out = ROOT / args.out
    save_model(model, out)
    final = evaluate(model, val_ex, device) if val_ex else evaluate(model, train_ex, device)
    print(f"\nSaved model -> {out}")
    print(f"Best val recall@k={best_val:.3f}  final val: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
