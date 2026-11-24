"""Coarsener that merges only consecutive customers of a reference solution, in its
order.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coarsening import SpatioTemporalGraphCoarsener   # noqa: E402


class OracleCoarsener(SpatioTemporalGraphCoarsener):
    def __init__(self, *args, routes, **kwargs):
        super().__init__(*args, **kwargs)
        self.routes = [list(map(str, r)) for r in routes]

    def _seq(self, nid: str) -> list:
        return self.inflate_route([[nid]])[0]

    def _contiguous(self, seq: list) -> bool:
        n = len(seq)
        return any(r[i:i + n] == seq for r in self.routes
                   for i in range(len(r) - n + 1))

    def _oracle_order(self, ni, nj):
        """'i' if seq(i)+seq(j) is a route segment, 'j' if seq(j)+seq(i), else None."""
        a, b = self._seq(ni.id), self._seq(nj.id)
        if self._contiguous(a + b):
            return "i"
        if self._contiguous(b + a):
            return "j"
        return None

    def score_edges(self, graph) -> None:
        """Route-adjacent pairs rank first, ordered among themselves by the heuristic
        score, so the threshold admits the same number of merges per level as any
        other scorer rather than every adjacent pair at once."""
        for e in graph.edges:
            if self.depot_id in (e.u_id, e.v_id):
                e.D_ij = 1e12
                continue
            ni, nj = graph.nodes[e.u_id], graph.nodes[e.v_id]
            base = self.merge_score(graph, e)
            e.D_ij = base - 1e9 if self._oracle_order(ni, nj) else base

    def _feasibility(self, ni, nj):
        """Non-adjacent pairs are rejected; adjacent ones keep the original time check."""
        order = self._oracle_order(ni, nj)
        if order is None:
            return False, False
        feas_ij, feas_ji = super()._feasibility(ni, nj)
        return (feas_ij, False) if order == "i" else (False, feas_ji)

    def _order_by_slack(self, ni, nj):
        """Keep the reference solution's order instead of choosing by slack."""
        if self._oracle_order(ni, nj) == "i":
            return f"{ni.id} -> {nj.id}", 0.0
        return f"{nj.id} -> {ni.id}", 0.0
