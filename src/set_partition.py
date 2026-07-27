"""Set-partitioning (route-selection) QUBO: one binary per pre-validated route.

An alternative to the position-based FQS/APS encoding. A pool of routes that
already satisfy capacity and time windows is generated classically; the QUBO then
selects a subset covering every customer exactly once at minimum cost:

    minimize   sum_r cost_r y_r
    subject to sum_{r covering c} y_r = 1   for every customer c

Because feasibility lives in the columns, there is no binary capacity slack and no
one-hot position block, so the QUBO is small (one variable per candidate route) and
its objective is directly the routing cost. This is the restricted set-partitioning
master problem of classical column generation. Coarsening shrinks the pool by
merging customers before route generation.

Reuses only the reference pipeline's VRPProblem and Qubo.
"""
from __future__ import annotations

import random
from collections import defaultdict

from src.qubo import Qubo


def route_cost(problem, route) -> float:
    d = problem.source_depot
    seq = [d] + list(route) + [d]
    return sum(problem.costs[seq[i]][seq[i + 1]] for i in range(len(seq) - 1))


def route_feasible(problem, route) -> bool:
    """Capacity and time-window feasibility of a single depot-to-depot route."""
    d = problem.source_depot
    if sum(problem.weights.get(j, 0) for j in route) > problem.capacities[0]:
        return False
    t = problem.time_windows[d][0]
    prev = d
    for j in route:
        t += problem.time_costs[prev][j]
        if t > problem.time_windows[j][1]:
            return False
        t = max(t, problem.time_windows[j][0]) + problem.service_times.get(j, 0)
        prev = j
    t += problem.time_costs[prev][d]
    return t <= problem.time_windows[d][1]


def generate_pool(problem, n_perms=60, seed=0, max_routes=250):
    """Diverse pool of feasible routes via randomized sequential insertion.

    Each of n_perms random customer orders is swept into routes greedily (a
    customer joins the current route if it stays feasible, else opens a new one).
    Every produced route is feasible. Singletons are added for any customer not
    otherwise covered so a full cover always exists. Capped to the cheapest
    max_routes to keep the QUBO tractable.
    """
    rng = random.Random(seed)
    custs = list(problem.dests)
    pool = set()
    for _ in range(n_perms):
        perm = custs[:]
        rng.shuffle(perm)
        route: list = []
        for c in perm:
            if route_feasible(problem, route + [c]):
                route.append(c)
            else:
                if route:
                    pool.add(tuple(route))
                route = [c] if route_feasible(problem, [c]) else []
        if route:
            pool.add(tuple(route))

    covered = set().union(*[set(r) for r in pool]) if pool else set()
    for c in custs:
        if c not in covered and route_feasible(problem, [c]):
            pool.add((c,))

    routes = sorted((list(r) for r in pool), key=lambda r: route_cost(problem, r))
    return routes[:max_routes]


def build_setpartition_qubo(problem, pool, penalty_scale=10.0) -> Qubo:
    """QUBO for min-cost exact cover over the route pool."""
    q = Qubo()
    costs = [route_cost(problem, r) for r in pool]
    A = penalty_scale * (max(costs) if costs else 1.0)

    for idx, c in enumerate(costs):
        q.add((f"r{idx}", f"r{idx}"), c)                 # objective

    covers = defaultdict(list)
    for idx, r in enumerate(pool):
        for c in r:
            covers[c].append(idx)

    for idxs in covers.values():                          # A * (sum y - 1)^2
        for i in idxs:
            q.add((f"r{i}", f"r{i}"), -A)
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                q.add((f"r{idxs[a]}", f"r{idxs[b]}"), 2 * A)
    return q


def decode_setpartition(sample, pool):
    """Selected routes (may over/under-cover before repair)."""
    return [pool[idx] for idx in range(len(pool)) if sample.get(f"r{idx}", 0) == 1]


def cover_status(selected, problem):
    """(exact_cover, n_missing, n_duplicate) for the selected routes."""
    seen = defaultdict(int)
    for r in selected:
        for c in r:
            seen[c] += 1
    custs = set(problem.dests)
    missing = custs - set(seen)
    dup = [c for c, n in seen.items() if n > 1]
    return (not missing and not dup), len(missing), len(dup)


def repair_cover(selected, problem):
    """Light fix to an exact cover: drop duplicate coverage, add singletons for
    uncovered customers. Keeps the experiment scorable when selection is imperfect."""
    seen = set()
    fixed = []
    for r in selected:
        kept = [c for c in r if c not in seen]
        seen.update(kept)
        if kept:
            fixed.append(kept)
    for c in problem.dests:
        if c not in seen and route_feasible(problem, [c]):
            fixed.append([c])
            seen.add(c)
    return fixed
