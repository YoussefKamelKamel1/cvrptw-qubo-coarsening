"""Retrain the merge scorer (GraphSAGE, GAT or GCN) on the expert-iteration labels.

  python experiments/retrain_sage.py --arch sage --seeds 0 1 2 3 4
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import torch                                                   # noqa: E402
from compare_architectures import train_one                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--arch", default="sage", choices=["sage", "gat", "gcn"])
a = ap.parse_args()
out = HERE / "results" / "retrain"
out.mkdir(parents=True, exist_ok=True)
data = torch.load(ROOT / "results" / "gnn_labels_rl.pt", weights_only=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
stats = {}
for s in a.seeds:
    targs = SimpleNamespace(epochs=300, lr=3e-3, hidden=64, layers=2, val_frac=0.15, seed=s)
    model, st = train_one(a.arch, data["examples"], targs, device)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "hidden": 64, "num_layers": 2, "arch": a.arch}, out / f"{a.arch}_seed{s}.pt")
    stats[s] = st
    print(s, st, flush=True)
(out / f"stats_{a.arch}_{'_'.join(map(str, a.seeds))}.json").write_text(json.dumps(stats, indent=1))
