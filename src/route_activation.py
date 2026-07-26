"""Route-activation (edge-based) QUBO for CVRPTW.

An alternative to the FQS/APS position encoding used by the main pipeline: the
variables are arcs ``y_{ij}=1`` iff a vehicle travels directly from node i to node
j. This module builds that QUBO from a ``VRPProblem`` and decodes an arc sample
back to routes; the shared classical repair (``src.solvers.VRPSolution``) then
completes and validates them.

Purpose: test whether graph coarsening, a pre-QUBO graph-level step, transfers to a
fundamentally different encoding. Nothing in the reference pipeline is modified.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.qubo import Qubo, VRPProblem


@dataclass
class RASWeights:
    degree: float = 1_000_000.0        # each customer: exactly one in- and one out-arc
    depot_balance: float = 1_000_000.0  # depot out-arcs == in-arcs (route balance)
    two_cycle: float = 500_000.0        # discourage i->j->i
    tw_forbid: float = 500_000.0        # forbid arcs that cannot meet j's window
    order: float = 1.0                  # arc-cost objective weight


def _arc_tw_infeasible(problem: VRPProblem, i: int, j: int) -> bool:
    """True if arc i->j can never be served within j's time window."""
    depot = problem.source_depot
    if j == depot:
        return False
    if i == depot:
        arrival = problem.time_windows[depot][0] + problem.time_costs[depot][j]
    else:
        arrival = problem.true_earliest[i] + problem.service_times[i] + problem.time_costs[i][j]
    return arrival > problem.time_windows[j][1]


def build_ras_qubo(problem: VRPProblem, w: RASWeights | None = None) -> Qubo:
    """Build the route-activation QUBO for a (possibly coarsened) VRPProblem."""
    w = w or RASWeights()
    depot = problem.source_depot
    customers = list(problem.dests)
    nodes = [depot] + customers
    q = Qubo()

    def y(i, j):
        return ("y", i, j)

    # each customer: exactly one outgoing arc and one incoming arc
    for c in customers:
        out_expr = [(1.0, y(c, j)) for j in nodes if j != c]
        in_expr = [(1.0, y(i, c)) for i in nodes if i != c]
        q.add_quadratic_equality_constraint(out_expr, -1.0, w.degree)
        q.add_quadratic_equality_constraint(in_expr, -1.0, w.degree)

    # depot: number of outgoing arcs equals number of incoming arcs
    bal = [(1.0, y(depot, j)) for j in customers] + [(-1.0, y(i, depot)) for i in customers]
    q.add_quadratic_equality_constraint(bal, 0.0, w.depot_balance)

    # discourage 2-cycles i->j->i
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            i, j = nodes[a], nodes[b]
            q.add((y(i, j), y(j, i)), w.two_cycle)

    # forbid time-infeasible arcs; add the arc-cost objective
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            if j != depot and _arc_tw_infeasible(problem, i, j):
                q.add((y(i, j), y(i, j)), w.tw_forbid)
            q.add((y(i, j), y(i, j)), w.order * problem.costs[i][j])
    return q


def decode_ras(sample: dict, problem: VRPProblem) -> list:
    """Decode an arc sample into (partial) int-customer routes, depot excluded.

    Follows depot out-arcs along the selected successors; any customers left
    unvisited are filled in by the shared classical repair afterwards.
    """
    depot = problem.source_depot
    succ: dict = {}
    for var, val in sample.items():
        if val == 1 and isinstance(var, tuple) and len(var) == 3 and var[0] == "y":
            _, i, j = var
            succ.setdefault(i, []).append(j)
    customer_set = set(problem.dests)
    routes, visited = [], set()
    for start in list(succ.get(depot, [])):
        route, cur, guard = [], start, 0
        while cur != depot and cur not in visited and guard <= len(customer_set):
            guard += 1
            if cur in customer_set:
                route.append(cur)
                visited.add(cur)
            nxt = [j for j in succ.get(cur, []) if j == depot or j not in visited]
            if not nxt:
                break
            cur = nxt[0]
        if route:
            routes.append(route)
    return routes
