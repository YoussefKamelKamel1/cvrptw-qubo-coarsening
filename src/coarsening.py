"""Spatio-temporal graph coarsening for CVRPTW (Phase-1 baseline).

Faithful port of the predecessor project's heuristic coarsener
(arXiv:2510.22329) into our editable source tree,
wired to ``src.datasets.VRPInstance``. Phases 2-3 modify this module (adaptive
penalties feed off the coarsened graph; the GNN replaces the merge scorer), so
we keep our own copy rather than importing the reference.

Merge metric (arXiv:2510.22329):
    D_ij = alpha * tau_ij + beta * max(0, e_j - (t_i + s_i + tau_ij))
with a service-window feasibility check on at least one visit order, midpoint
super-nodes, summed demand, and reversible inflation.

Design hook for Phase 3
-----------------------
The greedy merge choice is driven by ``merge_score(graph, edge)``. The GNN
policy will subclass/replace this single method, keeping the rest of the
multilevel loop identical -- this is the swappable "merge-scorer interface".
"""
from __future__ import annotations

import copy
import math
from typing import Optional

from .datasets import VRPInstance


def euclidean(n1: "Node", n2: "Node") -> float:
    return math.hypot(n1.x - n2.x, n1.y - n2.y)


class Node:
    """A customer or depot. ``t`` is the central time used by the merge metric."""

    def __init__(self, id, x, y, s, e, l, demand,
                 is_super_node=False, original_nodes=None):
        self.id = id
        self.x = x
        self.y = y
        self.s = s          # service time
        self.e = e          # earliest service start (window open)
        self.l = l          # latest service start (window close)
        self.demand = demand
        # Central time of the *start* of service (arXiv:2510.22329).
        self.t = (e + (l - s)) / 2 if (l - s) >= 0 else e
        self.is_super_node = is_super_node
        self.original_nodes = original_nodes if original_nodes is not None else [id]

    def __repr__(self):
        return (f"Node({self.id}, xy=({self.x:.1f},{self.y:.1f}), s={self.s:.1f}, "
                f"tw=[{self.e:.1f},{self.l:.1f}], d={self.demand:.1f}"
                f"{', SUPER' if self.is_super_node else ''})")


class Edge:
    def __init__(self, u_id, v_id, tau):
        self.u_id = u_id
        self.v_id = v_id
        self.tau = tau
        self.D_ij = 0.0


class Graph:
    """Undirected complete graph over nodes keyed by string id."""

    def __init__(self) -> None:
        self.nodes: dict = {}
        self.edges: list = []
        self.adj: dict = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.adj.setdefault(node.id, set())

    def add_edge(self, u_id, v_id, tau) -> None:
        if u_id not in self.nodes or v_id not in self.nodes:
            raise ValueError(f"Nodes {u_id} or {v_id} not in graph")
        if v_id in self.adj[u_id]:
            return  # edge exists
        self.edges.append(Edge(u_id, v_id, tau))
        self.adj[u_id].add(v_id)
        self.adj[v_id].add(u_id)

    def remove_node(self, node_id) -> None:
        if node_id not in self.nodes:
            return
        self.edges = [e for e in self.edges if e.u_id != node_id and e.v_id != node_id]
        for nb in list(self.adj.get(node_id, [])):
            self.adj[nb].discard(node_id)
        self.adj.pop(node_id, None)
        self.nodes.pop(node_id, None)

    def get_edge_by_nodes(self, u_id, v_id) -> Optional[Edge]:
        for e in self.edges:
            if (e.u_id == u_id and e.v_id == v_id) or (e.u_id == v_id and e.v_id == u_id):
                return e
        return None

    def get_neighbors(self, node_id) -> set:
        return self.adj.get(node_id, set())

    def distance(self, u_id, v_id) -> float:
        return euclidean(self.nodes[u_id], self.nodes[v_id])

    def build_complete_edges(self) -> None:
        ids = list(self.nodes.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if self.get_edge_by_nodes(ids[i], ids[j]) is None:
                    self.add_edge(ids[i], ids[j],
                                  euclidean(self.nodes[ids[i]], self.nodes[ids[j]]))


def graph_from_instance(inst: VRPInstance) -> tuple[Graph, str]:
    """Build a complete Graph from a VRPInstance. Depot is node id '0'.

    Node ids are the string indices '0'..'N' (index 0 = depot). Time-window
    fields default to a wide-open window when the instance has none (pure CVRP).
    """
    g = Graph()
    n = inst.coords.shape[0]
    tw = inst.has_time_windows
    big = float(inst.due_time[0]) if tw else 1e9
    for idx in range(n):
        e = float(inst.ready_time[idx]) if tw else 0.0
        l = float(inst.due_time[idx]) if tw else big
        s = float(inst.service_time[idx]) if tw else 0.0
        g.add_node(Node(str(idx), float(inst.coords[idx, 0]), float(inst.coords[idx, 1]),
                        s, e, l, float(inst.demand[idx])))
    g.build_complete_edges()
    return g, "0"


class SpatioTemporalGraphCoarsener:
    """Multilevel spatio-temporal coarsening (arXiv:2510.22329).

    The merge choice goes through ``merge_score`` so Phase 3's GNN can override
    just that method. Everything else (rho thresholding, feasibility, window
    tightening, super-node creation, inflation) is the published heuristic.
    """

    def __init__(self, graph: Graph, alpha: float, beta: float, P: float,
                 radiusCoeff: float, depot_id: str):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.P = P
        self.radiusCoeff = radiusCoeff
        self.depot_id = depot_id
        self.merge_layers: list = []  # (super_id, i_id, j_id, pi_order)

    # -- merge scorer (Phase-3 override point) ----------------------------- #
    def merge_score(self, graph: Graph, edge: Edge) -> float:
        """Heuristic spatio-temporal distance D_ij (lower = merge first)."""
        ni = graph.nodes[edge.u_id]
        nj = graph.nodes[edge.v_id]
        tau = euclidean(ni, nj)
        temporal_slack = max(0.0, nj.e - (ni.t + ni.s + tau))
        return self.alpha * tau + self.beta * temporal_slack

    def score_edges(self, graph: Graph) -> None:
        """Assign ``edge.D_ij`` for every edge in the current graph.

        This is the seam the GNN overrides: the learned coarsener scores all
        edges in a single batched forward pass per level, while the base class
        scores edge-by-edge. Everything downstream (rho threshold, feasibility
        gating, window tightening) is identical regardless of scorer.
        """
        for edge in graph.edges:
            edge.D_ij = self.merge_score(graph, edge)

    def stop_hook(self, graph: Graph, sorted_edges: list) -> bool:
        """Learned-stopping seam (Phase-5). Base class never stops early;
        ``GNNCoarsener`` overrides this to stop when no candidate merge is
        confident enough, so the model chooses the coarsening depth per instance
        instead of a fixed target ratio ``P``."""
        return False

    # -- feasibility / ordering / window tightening ------------------------ #
    @staticmethod
    def _feasibility(ni: Node, nj: Node) -> tuple[bool, bool]:
        tau = euclidean(ni, nj)
        feas_ij = (ni.e + ni.s + tau <= nj.l)
        feas_ji = (nj.e + nj.s + tau <= ni.l)
        return feas_ij, feas_ji

    @staticmethod
    def _order_by_slack(ni: Node, nj: Node) -> tuple[str, float]:
        tau = euclidean(ni, nj)
        slack_ij = nj.l - (ni.e + ni.s + tau)
        slack_ji = ni.l - (nj.e + nj.s + tau)
        if slack_ij >= slack_ji:
            return f"{ni.id} -> {nj.id}", slack_ij
        return f"{nj.id} -> {ni.id}", slack_ji

    @staticmethod
    def _new_window(ni: Node, nj: Node, pi_order: str) -> tuple[float, float]:
        tau = euclidean(ni, nj)
        first_id, _, _ = pi_order.split(" ")
        if first_id == ni.id:      # i -> j
            e_prime = max(ni.e, nj.e - (ni.s + tau))
            l_prime = min(ni.l, nj.l - ni.s - tau)
        else:                       # j -> i
            e_prime = max(nj.e, ni.e - (nj.s + tau))
            l_prime = min(nj.l, ni.l - nj.s - tau)
        return e_prime, l_prime

    def _reconnect(self, g: Graph, super_node: Node, ni: Node, nj: Node) -> None:
        nbrs = (g.get_neighbors(ni.id) | g.get_neighbors(nj.id)) - {
            ni.id, nj.id, super_node.id, self.depot_id}
        for nb in nbrs:
            if nb in g.nodes:
                g.add_edge(super_node.id, nb, euclidean(super_node, g.nodes[nb]))

    # -- main loop --------------------------------------------------------- #
    def coarsen(self, record_levels: bool = False) -> tuple[Graph, list]:
        # record_levels: for Phase-3 label generation, capture (graph state,
        # set of frozenset merges) before each level's merges are applied.
        self.level_snapshots: list = []
        G = copy.deepcopy(self.graph)
        n0 = len(G.nodes)
        while len(G.nodes) > self.P * n0:
            self.score_edges(G)
            sorted_edges = sorted(G.edges, key=lambda e: e.D_ij)
            if not sorted_edges:
                break
            if self.stop_hook(G, sorted_edges):
                break
            k = math.floor(0.1 * len(sorted_edges) * self.radiusCoeff)
            rho = sorted_edges[min(k, len(sorted_edges) - 1)].D_ij

            M = []
            U = {self.depot_id}
            for edge in sorted_edges:
                i_id, j_id = edge.u_id, edge.v_id
                if i_id in U or j_id in U:
                    continue
                if edge.D_ij > rho:
                    continue
                ni, nj = G.nodes[i_id], G.nodes[j_id]
                feas_ij, feas_ji = self._feasibility(ni, nj)
                if not (feas_ij or feas_ji):
                    continue
                pi_order, _ = self._order_by_slack(ni, nj)
                e_prime, l_prime = self._new_window(ni, nj, pi_order)
                if l_prime < e_prime:  # empty intersection -> skip
                    continue
                M.append((i_id, j_id, pi_order, e_prime, l_prime, ni.demand + nj.demand))
                U.add(i_id)
                U.add(j_id)

            if not M:
                break

            if record_levels:
                merges_here = {frozenset((m[0], m[1])) for m in M}
                self.level_snapshots.append((copy.deepcopy(G), merges_here))

            to_remove = set()
            for i_id, j_id, pi_order, e_prime, l_prime, demand_ij in M:
                ni, nj = G.nodes[i_id], G.nodes[j_id]
                sid = f"SN_{ni.id}_{nj.id}"
                tau_internal = euclidean(ni, nj)
                s_ij = ni.s + tau_internal + nj.s
                super_node = Node(
                    sid, (ni.x + nj.x) / 2, (ni.y + nj.y) / 2, s_ij, e_prime, l_prime,
                    demand_ij, is_super_node=True,
                    original_nodes=list(set(ni.original_nodes + nj.original_nodes)),
                )
                G.add_node(super_node)
                self.merge_layers.append((sid, i_id, j_id, pi_order))
                self._reconnect(G, super_node, ni, nj)
                to_remove.add(i_id)
                to_remove.add(j_id)
            for nid in to_remove:
                G.remove_node(nid)
        return G, self.merge_layers

    def inflate_route(self, coarsened_routes: list) -> list:
        """Expand super-nodes back to their ordered original pairs (reverse layers)."""
        out = []
        for route in coarsened_routes:
            inflated = list(route)
            if not inflated:
                continue
            for sid, i_id, j_id, pi_order in reversed(self.merge_layers):
                first_id, _, _ = pi_order.split(" ")
                pair = [i_id, j_id] if first_id == i_id else [j_id, i_id]
                expanded = []
                for node in inflated:
                    if node == sid:
                        expanded.extend(pair)
                    else:
                        expanded.append(node)
                inflated = expanded
            out.append(inflated)
        return out
