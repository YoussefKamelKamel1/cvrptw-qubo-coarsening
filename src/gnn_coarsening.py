"""GNN-guided coarsening (Phase 3, headline contribution).

Replaces the heuristic spatio-temporal merge score D_ij with a learned scorer.
The learned policy scores which node pairs to merge; everything else in the
multilevel loop (rho threshold, feasibility gating, window tightening, super-node
creation, inflation) is unchanged -- so infeasible merges are still blocked by the
same feasibility check. This is the swappable "merge-scorer interface" from the
project plan (SpatioTemporalGraphCoarsener.score_edges).

Motivation (post Phase 1/2): the predecessor's weakness is per-family hand-tuned
coarsening params (alpha, beta, P, radiusCoeff). A learned policy trained on the
per-instance *oracle* coarsening (best hyperparameters found by search, labelled
by downstream feasibility+cost) should generalise across C/R/RC **without**
per-family tuning -- imitating the union of what works, not any one family.

Model: node encoder -> GraphSAGE message passing -> symmetric edge head giving
P(merge) per candidate pair. Node features: coords, demand, service, [e,l,t],
window width, degree, super-node flag, distance-to-depot. Edge features:
tau, bidirectional temporal slack, window overlap, feasibility flags, combined
demand, |t_i - t_j| -- all instance-normalised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from .coarsening import Edge, Graph, SpatioTemporalGraphCoarsener, euclidean

NODE_FEATURE_DIM = 11
EDGE_FEATURE_DIM = 8
_EPS = 1e-6


@dataclass
class FeatureBundle:
    node_feat: torch.Tensor        # [n, NODE_FEATURE_DIM]
    mp_edge_index: torch.Tensor    # [2, 2E] message-passing edges (both directions)
    cand_edge_index: torch.Tensor  # [2, m] candidate (non-depot) pairs, one direction
    cand_edge_feat: torch.Tensor   # [m, EDGE_FEATURE_DIM]
    cand_edges: list               # the m Edge objects, aligned with cand_edge_index cols
    labels: Optional[torch.Tensor] = None  # [m] 0/1 (training only)


def _normalizers(G: Graph) -> dict:
    xs = [nd.x for nd in G.nodes.values()]
    ys = [nd.y for nd in G.nodes.values()]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    diag = max(_EPS, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    horizon = max(_EPS, max(nd.l for nd in G.nodes.values()))
    max_demand = max(_EPS, max(nd.demand for nd in G.nodes.values()))
    return {"x0": x0, "y0": y0, "diag": diag, "horizon": horizon,
            "max_demand": max_demand}


def build_features(G: Graph, depot_id: str) -> FeatureBundle:
    """Extract node/edge features + candidate edges from a graph state."""
    nz = _normalizers(G)
    diag, horizon, maxd = nz["diag"], nz["horizon"], nz["max_demand"]
    ids = sorted(G.nodes.keys())
    idx = {nid: i for i, nid in enumerate(ids)}
    n = len(ids)
    depot = G.nodes[depot_id]

    node_feat = torch.zeros((n, NODE_FEATURE_DIM), dtype=torch.float32)
    for nid in ids:
        nd = G.nodes[nid]
        deg = len(G.get_neighbors(nid))
        node_feat[idx[nid]] = torch.tensor([
            (nd.x - nz["x0"]) / diag,
            (nd.y - nz["y0"]) / diag,
            nd.demand / maxd,
            nd.s / horizon,
            nd.e / horizon,
            nd.l / horizon,
            nd.t / horizon,
            (nd.l - nd.e) / horizon,
            deg / max(1, n - 1),
            1.0 if nd.is_super_node else 0.0,
            euclidean(nd, depot) / diag,
        ], dtype=torch.float32)

    # message-passing edges: all graph edges, both directions
    mp = []
    for e in G.edges:
        u, v = idx[e.u_id], idx[e.v_id]
        mp.append((u, v))
        mp.append((v, u))
    mp_edge_index = (torch.tensor(mp, dtype=torch.long).t().contiguous()
                     if mp else torch.zeros((2, 0), dtype=torch.long))

    # candidate edges: non-depot pairs only (depot never merges)
    cand_cols, cand_feat, cand_edges = [], [], []
    for e in G.edges:
        if e.u_id == depot_id or e.v_id == depot_id:
            continue
        ni, nj = G.nodes[e.u_id], G.nodes[e.v_id]
        tau = euclidean(ni, nj)
        slack_ij = max(0.0, nj.e - (ni.t + ni.s + tau))
        slack_ji = max(0.0, ni.e - (nj.t + nj.s + tau))
        overlap = max(0.0, min(ni.l, nj.l) - max(ni.e, nj.e))
        feas_ij = 1.0 if (ni.e + ni.s + tau <= nj.l) else 0.0
        feas_ji = 1.0 if (nj.e + nj.s + tau <= ni.l) else 0.0
        cand_cols.append((idx[e.u_id], idx[e.v_id]))
        cand_feat.append([
            tau / diag,
            slack_ij / horizon,
            slack_ji / horizon,
            overlap / horizon,
            feas_ij,
            feas_ji,
            (ni.demand + nj.demand) / maxd,
            abs(ni.t - nj.t) / horizon,
        ])
        cand_edges.append(e)

    cand_edge_index = (torch.tensor(cand_cols, dtype=torch.long).t().contiguous()
                       if cand_cols else torch.zeros((2, 0), dtype=torch.long))
    cand_edge_feat = torch.tensor(cand_feat, dtype=torch.float32) if cand_feat \
        else torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    return FeatureBundle(node_feat, mp_edge_index, cand_edge_index,
                         cand_edge_feat, cand_edges)


class MergeScorerGNN(nn.Module):
    """GraphSAGE node encoder + symmetric edge head -> P(merge) per pair."""

    def __init__(self, hidden: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(NODE_FEATURE_DIM, hidden), nn.ReLU())
        self.convs = nn.ModuleList(
            [SAGEConv(hidden, hidden) for _ in range(num_layers)])
        self.dropout = dropout
        # symmetric edge head: [h_i+h_j, |h_i-h_j|, edge_feat]
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden + EDGE_FEATURE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def encode(self, node_feat, mp_edge_index):
        h = self.node_enc(node_feat)
        for conv in self.convs:
            h = F.relu(conv(h, mp_edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, node_feat, mp_edge_index, cand_edge_index, cand_edge_feat):
        h = self.encode(node_feat, mp_edge_index)
        if cand_edge_index.numel() == 0:
            return torch.zeros((0,), device=node_feat.device)
        hi = h[cand_edge_index[0]]
        hj = h[cand_edge_index[1]]
        edge_repr = torch.cat([hi + hj, (hi - hj).abs(), cand_edge_feat], dim=-1)
        return self.edge_head(edge_repr).squeeze(-1)  # logits [m]

    def score_bundle(self, fb: FeatureBundle, device) -> torch.Tensor:
        return self.forward(fb.node_feat.to(device), fb.mp_edge_index.to(device),
                            fb.cand_edge_index.to(device),
                            fb.cand_edge_feat.to(device))


class GNNCoarsener(SpatioTemporalGraphCoarsener):
    """Coarsener whose merge scoring comes from a trained MergeScorerGNN.

    Overrides only ``score_edges``: one batched forward per level sets
    ``edge.D_ij = -P(merge)`` for candidate edges (so high merge-probability pairs
    sort first) and a large value for depot edges (never merged). P, radiusCoeff
    still control the rho threshold and stopping; alpha/beta are unused.
    """

    def __init__(self, graph: Graph, P: float, radiusCoeff: float, depot_id: str,
                 model: MergeScorerGNN, device="cpu", alpha: float = 1.0,
                 beta: float = 1.0, explore: float = 0.0, rng=None,
                 stop_threshold: Optional[float] = None):
        super().__init__(graph, alpha=alpha, beta=beta, P=P,
                         radiusCoeff=radiusCoeff, depot_id=depot_id)
        self.model = model.to(device).eval()
        self.device = device
        # explore > 0 -> add Gaussian noise to scores for diverse self-rollouts
        # (used by Phase-4 expert iteration). explore = 0 -> deterministic.
        self.explore = explore
        self.rng = rng if rng is not None else np.random.default_rng()
        # Phase-5 learned stopping: if set, stop coarsening once the best merge
        # probability drops below this threshold, letting the model choose depth
        # per instance (use a low P so the threshold controls stopping).
        self.stop_threshold = stop_threshold

    def stop_hook(self, graph: Graph, sorted_edges: list) -> bool:
        if self.stop_threshold is None or not sorted_edges:
            return False
        # candidate edges carry D_ij = -P(merge); depot/non-candidate = +1e6.
        # sorted ascending, so the first is the most confident merge.
        best_prob = -sorted_edges[0].D_ij
        return best_prob < self.stop_threshold

    @torch.no_grad()
    def score_edges(self, graph: Graph) -> None:
        fb = build_features(graph, self.depot_id)
        big = 1e6
        for e in graph.edges:
            e.D_ij = big  # default (covers depot edges + isolated)
        if fb.cand_edge_index.numel() == 0:
            return
        logits = self.model.score_bundle(fb, self.device)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        if self.explore > 0:
            prob = prob + self.rng.normal(0.0, self.explore, size=prob.shape)
        for e, p in zip(fb.cand_edges, prob.tolist()):
            e.D_ij = -p  # higher merge prob -> lower D -> merged first


def save_model(model: MergeScorerGNN, path) -> None:
    torch.save({"state_dict": model.state_dict(),
                "hidden": model.node_enc[0].out_features,
                "num_layers": len(model.convs)}, path)


def load_model(path, device="cpu") -> MergeScorerGNN:
    ckpt = torch.load(path, map_location=device)
    model = MergeScorerGNN(hidden=ckpt.get("hidden", 64),
                           num_layers=ckpt.get("num_layers", 2))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)
