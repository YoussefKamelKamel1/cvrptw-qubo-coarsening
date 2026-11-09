"""Method figures -> figures/.

fig_instances    : customer layouts of C101, R101 and RC101 (first 25 customers).
fig_coarsening   : GNN coarsening of R101 at N=15 with the released policy.
fig_gnn_training : loss and edge-ranking curves of the multi-size model, values
                   taken from the train_gnn.py log of that run.

Run: python scripts/make_method_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.datasets import load_solomon
from src.coarsening import graph_from_instance
from src.gnn_coarsening import GNNCoarsener, load_model

FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.bbox": "tight", "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=140)
    plt.close(fig)
    print("  wrote", name)


def fig_instances():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, name in zip(axes, ["C101", "R101", "RC101"]):
        inst = load_solomon(name).truncate(25)
        sc = ax.scatter(inst.coords[1:, 0], inst.coords[1:, 1],
                        c=inst.ready_time[1:], cmap="viridis", s=45)
        ax.scatter(inst.coords[0, 0], inst.coords[0, 1], c="red", marker="*",
                   s=260, edgecolor="k", label="depot", zorder=5)
        ax.set_title(f"{name} ({inst.family})")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(sc, ax=axes, label="window open time $e_i$", shrink=0.8)
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, "fig_instances")


def fig_coarsening():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ROOT / "results" / "gnn_model_rl.pt", device=device)
    inst = load_solomon("R101").truncate(15)
    g, depot = graph_from_instance(inst)
    gc = GNNCoarsener(g, P=0.5, radiusCoeff=0.5, depot_id=depot, model=model, device=device)
    cg, layers = gc.coarsen()
    pos = {str(i): (inst.coords[i, 0], inst.coords[i, 1]) for i in range(inst.coords.shape[0])}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    # left: original
    a1.scatter([p[0] for k, p in pos.items() if k != depot],
               [p[1] for k, p in pos.items() if k != depot], s=60, c="#1f77b4")
    a1.scatter(*pos[depot], c="red", marker="*", s=280, edgecolor="k", zorder=5)
    a1.set_title(f"Original graph (N={inst.n_customers})")
    # right: coarsened, super-nodes at midpoints connected to their originals
    cmap = plt.get_cmap("tab20")
    ci = 0
    for nid, node in cg.nodes.items():
        if nid == depot:
            a2.scatter(node.x, node.y, c="red", marker="*", s=280, edgecolor="k", zorder=5)
            continue
        if node.is_super_node:
            col = cmap(ci % 20); ci += 1
            for leaf in node.original_nodes:
                if leaf in pos:
                    a2.plot([node.x, pos[leaf][0]], [node.y, pos[leaf][1]],
                            c=col, lw=1, alpha=0.6)
                    a2.scatter(*pos[leaf], c=[col], s=25, alpha=0.6)
            a2.scatter(node.x, node.y, c=[col], marker="s", s=110, edgecolor="k", zorder=4)
        else:
            a2.scatter(node.x, node.y, c="#1f77b4", s=60)
    a2.set_title(f"GNN-coarsened ({len(cg.nodes)} nodes, {len(layers)} merges)")
    for ax in (a1, a2):
        ax.set_xlabel("x"); ax.set_ylabel("y")
    save(fig, "fig_coarsening")


def fig_gnn_training():
    ep = [1,25,50,75,100,125,150,175,200,225,250,275,300]
    loss = [.9536,.4471,.3515,.3408,.3057,.3179,.2741,.2850,.2640,.2738,.3047,.2804,.2330]
    tauc = [.927,.967,.977,.982,.982,.984,.985,.984,.987,.986,.984,.986,.986]
    vauc = [.924,.966,.973,.974,.975,.974,.975,.974,.978,.977,.976,.974,.977]
    trec = [.480,.672,.711,.737,.737,.752,.764,.763,.791,.761,.739,.770,.755]
    vrec = [.480,.636,.692,.691,.690,.686,.651,.681,.694,.657,.670,.677,.737]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.7))
    a1.plot(ep, loss, marker="o", color="#333"); a1.set_title("Training loss (BCE)")
    a1.set_xlabel("epoch"); a1.set_ylabel("loss")
    a2.plot(ep, tauc, marker="o", color="#1f77b4", label="train AUC")
    a2.plot(ep, vauc, marker="o", color="#1f77b4", ls="--", label="val AUC")
    a2.plot(ep, trec, marker="s", color="#d62728", label="train recall@k")
    a2.plot(ep, vrec, marker="s", color="#d62728", ls="--", label="val recall@k")
    a2.set_title("Edge-ranking quality (multi-N model)")
    a2.set_xlabel("epoch"); a2.set_ylabel("score"); a2.set_ylim(0.4, 1.02); a2.legend(fontsize=8)
    save(fig, "fig_gnn_training")


if __name__ == "__main__":
    fig_instances()
    fig_coarsening()
    fig_gnn_training()
