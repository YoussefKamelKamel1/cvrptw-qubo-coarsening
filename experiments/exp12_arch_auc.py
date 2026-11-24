"""Validation AUC of the three message-passing trunks (training only)."""
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import torch                                                       # noqa: E402
from compare_architectures import ARCHS, train_one                 # noqa: E402

data = torch.load(ROOT / "results" / "gnn_labels_rl.pt", weights_only=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
rows = []
for arch in ARCHS:
    for ts in (0, 1, 2):
        targs = SimpleNamespace(epochs=300, lr=3e-3, hidden=64, layers=2, val_frac=0.15, seed=ts)
        _, st = train_one(arch, data["examples"], targs, device)
        rows.append(dict(arch=arch, train_seed=ts, **st))
        print(rows[-1], flush=True)
with open(HERE / "results" / "exp12_arch_auc.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
