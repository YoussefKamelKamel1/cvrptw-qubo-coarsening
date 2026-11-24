"""Train a GraphSAGE merge scorer on the improved labels, saving a checkpoint every
25 epochs.

  python experiments/gnn_v2_train.py --labels experiments/results/v2/labels_r1.pt --seed 0 --tag r1
"""
import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import torch                                                            # noqa: E402
from torch import nn                                                    # noqa: E402
from compare_architectures import VariantScorer, _to_device            # noqa: E402
from src.utils import set_global_seed                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--epochs", type=int, default=300)
a = ap.parse_args()

set_global_seed(a.seed)
rng = random.Random(a.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
ex = list(torch.load(a.labels, weights_only=False)["examples"])
pos = sum(int(e["labels"].sum()) for e in ex)
tot = sum(int(e["labels"].numel()) for e in ex)
model = VariantScorer("sage", hidden=64, num_layers=2).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(tot - pos) / max(1, pos)], device=device))
out = HERE / "results" / "v2" / f"{a.tag}_seed{a.seed}"
out.mkdir(parents=True, exist_ok=True)
for epoch in range(1, a.epochs + 1):
    model.train()
    rng.shuffle(ex)
    for e in ex:
        nf, mp, ci, cf, y = _to_device(e, device)
        if ci.numel() == 0:
            continue
        opt.zero_grad()
        loss_fn(model(nf, mp, ci, cf), y).backward()
        opt.step()
    if epoch % 25 == 0:
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "hidden": 64, "num_layers": 2, "arch": "sage", "epoch": epoch},
                   out / f"ep{epoch:03d}.pt")
        print(f"epoch {epoch} saved", flush=True)
