"""Feasibility-preserving local search (2-opt + relocate) for CVRPTW routes.

A shared cost-polishing step applied *equally* to any feasible solution, so we
can ask whether a coarsened-pipeline starting point leads to a better local
optimum than a cold greedy start. Moves are accepted only if they reduce total
distance and keep every route capacity- and time-window-feasible. Operates in
the integer node space of a ``VRPProblem`` (index 0 = depot).
"""
from __future__ import annotations

from .qubo import VRPProblem


def route_distance(p: VRPProblem, route) -> float:
    if not route:
        return 0.0
    d = p.source_depot
    total = p.costs[d][route[0]]
    for a, b in zip(route, route[1:]):
        total += p.costs[a][b]
    return total + p.costs[route[-1]][d]


def solution_distance(p: VRPProblem, routes) -> float:
    return sum(route_distance(p, r) for r in routes)


def route_tw_violations(p: VRPProblem, route) -> int:
    if not route:
        return 0
    d = p.source_depot
    v = 0
    t = max(0.0, p.time_windows[d][0]) + p.time_costs[d][route[0]]
    ready, due = p.time_windows[route[0]]
    if t > due:
        v += 1
    t = max(t, ready) + p.service_times[route[0]]
    for i in range(len(route) - 1):
        t += p.time_costs[route[i]][route[i + 1]]
        ready, due = p.time_windows[route[i + 1]]
        if t > due:
            v += 1
        t = max(t, ready) + p.service_times[route[i + 1]]
    return v


def route_demand(p: VRPProblem, route) -> float:
    return sum(p.weights.get(c, 0) for c in route)


def _cap_of(p: VRPProblem, idx: int) -> float:
    return p.capacities[idx] if idx < len(p.capacities) else p.capacities[0]


def _two_opt_pass(p: VRPProblem, routes) -> bool:
    """Best-improvement intra-route segment reversal on each route."""
    improved = False
    for ri, route in enumerate(routes):
        n = len(route)
        if n < 3:
            continue
        base = route_distance(p, route)
        best, best_d = route, base
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                dc = route_distance(p, cand)
                if dc < best_d - 1e-9 and route_tw_violations(p, cand) == 0:
                    best, best_d = cand, dc
        if best_d < base - 1e-9:
            routes[ri] = best
            improved = True
    return improved


def _relocate_pass(p: VRPProblem, routes) -> bool:
    """Best-improvement relocation of a single customer to another route."""
    best_delta, best_move = -1e-9, None
    for ai, ra in enumerate(routes):
        da = route_distance(p, ra)
        for pos, cust in enumerate(ra):
            ra_wo = ra[:pos] + ra[pos + 1:]
            da_wo = route_distance(p, ra_wo)
            demand = p.weights.get(cust, 0)
            for bi, rb in enumerate(routes):
                if bi == ai:
                    continue
                if route_demand(p, rb) + demand > _cap_of(p, bi):
                    continue
                db = route_distance(p, rb)
                for q in range(len(rb) + 1):
                    cand = rb[:q] + [cust] + rb[q:]
                    if route_tw_violations(p, cand) != 0:
                        continue
                    delta = (da_wo - da) + (route_distance(p, cand) - db)
                    if delta < best_delta:
                        best_delta, best_move = delta, (ai, pos, bi, q)
    if best_move is None:
        return False
    ai, pos, bi, q = best_move
    cust = routes[ai][pos]
    routes[ai] = routes[ai][:pos] + routes[ai][pos + 1:]
    routes[bi] = routes[bi][:q] + [cust] + routes[bi][q:]
    return True


def local_search(p: VRPProblem, routes, max_passes: int = 100):
    """Polish routes toward a local distance optimum (feasibility-preserving)."""
    routes = [list(r) for r in routes if r]
    for _ in range(max_passes):
        a = _two_opt_pass(p, routes)
        b = _relocate_pass(p, routes)
        if not (a or b):
            break
    return [r for r in routes if r]
