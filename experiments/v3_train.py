"""Train the outcome model of the selector and calibrate its size penalty.

  python experiments/v3_train.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import numpy as np                                                        # noqa: E402
import torch                                                              # noqa: E402
from torch import nn                                                      # noqa: E402
from torch_geometric.data import Data, Batch                              # noqa: E402
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool  # noqa: E402
from v3_common import FIXED_KEY                                           # noqa: E402

R = HERE / "results" / "v3"


class ValueNet(nn.Module):
    def __init__(self, nf=11, ng=10, h=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(nf, h), nn.ReLU())
        self.c1, self.c2 = SAGEConv(h, h), SAGEConv(h, h)
        self.head = nn.Sequential(nn.Linear(2 * h + ng, h), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, b):
        x = self.enc(b.x)
        x = torch.relu(self.c1(x, b.edge_index))
        x = torch.relu(self.c2(x, b.edge_index))
        g = torch.cat([global_mean_pool(x, b.batch), global_max_pool(x, b.batch), b.glob], 1)
        return self.head(g).squeeze(-1)


def to_data(c):
    return Data(x=c["node_feat"], edge_index=c["mp_edge_index"], glob=c["glob"].unsqueeze(0),
                y=torch.tensor([c["feas"]], dtype=torch.float32))


def load_all():
    tasks = [torch.load(f, weights_only=False) for f in sorted((R / "data").glob("*.pt"))]
    return tasks


@torch.no_grad()
def predict(models, cands, device="cpu"):
    b = Batch.from_data_list([to_data(c) for c in cands]).to(device)
    return np.mean([torch.sigmoid(m(b)).cpu().numpy() for m in models], axis=0)


def select(tasks, preds, lam):
    feas, nv = [], []
    for t, p in zip(tasks, preds):
        logv = np.log([c["nvars"] for c in t["cands"]])
        k = int(np.argmax(p - lam * logv))
        feas.append(t["cands"][k]["feas"]); nv.append(t["cands"][k]["nvars"])
    return float(np.mean(feas)), float(np.mean(nv))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = load_all()
    tr = [t for t in tasks if t["split"] == "train"]
    va = [t for t in tasks if t["split"] == "val"]
    print(f"{len(tr)} train tasks, {len(va)} val tasks, {sum(len(t['cands']) for t in tr)} train candidates")
    trd = [to_data(c) for t in tr for c in t["cands"]]
    vad = [to_data(c) for t in va for c in t["cands"]]
    vb = Batch.from_data_list(vad).to(device)
    models = []
    for seed in range(5):
        torch.manual_seed(seed); np.random.seed(seed)
        m = ValueNet().to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_state, bad = 1e9, None, 0
        rng = np.random.default_rng(seed)
        for ep in range(200):
            m.train()
            idx = rng.permutation(len(trd))
            for s in range(0, len(idx), 64):
                b = Batch.from_data_list([trd[i] for i in idx[s:s + 64]]).to(device)
                opt.zero_grad()
                nn.functional.binary_cross_entropy_with_logits(m(b), b.y).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                vl = float(nn.functional.binary_cross_entropy_with_logits(m(vb), vb.y))
            if vl < best - 1e-4:
                best, best_state, bad = vl, {k: v.cpu().clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 20:
                    break
        m.load_state_dict(best_state); m.eval().cpu()
        torch.save(best_state, R / f"value_seed{seed}.pt")
        models.append(m)
        print(f"seed {seed}: best val loss {best:.4f} at epoch {ep - bad + 1}", flush=True)

    vp = [predict(models, t["cands"]) for t in va]
    fx = [next(c for c in t["cands"] if c["key"] == FIXED_KEY) for t in va]
    fixed_feas, fixed_vars = float(np.mean([c["feas"] for c in fx])), float(np.mean([c["nvars"] for c in fx]))
    grid = np.linspace(0, 0.5, 201)
    res = [(lam, *select(va, vp, lam)) for lam in grid]
    lam = min(res, key=lambda r: (abs(r[2] - fixed_vars), -r[1]))[0]
    sel_feas, sel_vars = select(va, vp, lam)
    oracle = float(np.mean([max(c["feas"] for c in t["cands"]) for t in va]))
    out = dict(lam=float(lam), val_selected_feas=sel_feas, val_selected_vars=sel_vars,
               val_fixed_feas=fixed_feas, val_fixed_vars=fixed_vars, val_oracle_feas=oracle,
               curve=[dict(lam=float(a), feas=b, vars=c) for a, b, c in res[::10]])
    (R / "value_calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    main()
