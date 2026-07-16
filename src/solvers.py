"""QUBO samplers + solution decoding/repair for CVRPTW.

Ported from the predecessor (quantum_solvers/vrp_solution.py, vrp_solvers.py,
DWaveSolvers_modified.py) into our source tree, with two deliberate changes:

  1. **Seeded sampling.** The reference never seeds the SA sampler; we thread a
     seed through so every run is reproducible (seeds everywhere).
  2. **Fixed, logged solver budget.** ``SolverConfig`` bundles backend +
     num_reads + num_sweeps + seed so all conditions share one budget -- without
     this every downstream comparison is confounded.

Backends: 'sa' (dwave-samplers SimulatedAnnealingSampler = modern neal),
'exact' (dimod.ExactSolver, tiny instances only), 'tabu', 'steepest'.
An optional 'leap' stub is intentionally left unimplemented for the main path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from typing import Optional

from .qubo import PenaltyWeights, VRPProblem


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
@dataclass
class SolverConfig:
    """Shared, logged solver budget. Held constant across conditions."""

    backend: str = "sa"          # 'sa' | 'exact' | 'tabu' | 'steepest'
    num_reads: int = 1000
    num_sweeps: int = 1000       # SA only
    seed: Optional[int] = None

    def as_record(self) -> dict:
        return {"backend": self.backend, "num_reads": self.num_reads,
                "num_sweeps": self.num_sweeps, "seed": self.seed}


def solve_qubo(qubo, cfg: SolverConfig, limit: int = 1) -> list[dict]:
    """Sample a QUBO. Returns up to ``limit`` lowest-energy samples as dicts."""
    backend = cfg.backend
    if backend == "sa":
        from dwave.samplers import SimulatedAnnealingSampler
        sampler = SimulatedAnnealingSampler()
        kwargs = dict(num_reads=cfg.num_reads, num_sweeps=cfg.num_sweeps)
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed
        response = sampler.sample_qubo(qubo.dict, **kwargs)
    elif backend == "exact":
        from dimod import ExactSolver
        response = ExactSolver().sample_qubo(qubo.dict)
    elif backend == "tabu":
        from dwave.samplers import TabuSampler
        kwargs = dict(num_reads=cfg.num_reads)
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed
        response = TabuSampler().sample_qubo(qubo.dict, **kwargs)
    elif backend == "steepest":
        from dwave.samplers import SteepestDescentSampler
        response = SteepestDescentSampler().sample_qubo(qubo.dict, num_reads=cfg.num_reads)
    elif backend == "leap":
        raise NotImplementedError(
            "Leap/QPU backend is an optional stub; the main path is simulator-only.")
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return [s for s in response.lowest()][:limit]


# --------------------------------------------------------------------------- #
# Solution decoding + repair (ported from vrp_solution.py)
# --------------------------------------------------------------------------- #
class VRPSolution:
    """Decode an FQS/APS sample into routes, then repair to a valid solution.

    ``raw_solution`` holds the routes decoded directly from the sample *before*
    repair (used to report pre-repair feasibility, ).
    """

    def __init__(self, problem: VRPProblem, sample, vehicle_k_limits, solution=None):
        self.problem = problem
        self.depot = problem.source_depot
        self.raw_solution: list = []

        if solution is not None:
            self.solution = solution
            return

        num_vehicles = len(problem.capacities)
        temp = {i: [] for i in range(num_vehicles)}
        for var, val in sample.items():
            if val == 1 and isinstance(var, tuple) and len(var) == 3 and isinstance(var[0], int):
                i, j, k = var
                if i < num_vehicles:
                    temp[i].append((k, j))
        decoded = []
        for i in range(num_vehicles):
            route = [j for _, j in sorted(temp[i], key=lambda x: x[0])]
            if route:
                decoded.append(route)
        self.raw_solution = [list(r) for r in decoded]

        repaired = self._repair_solution(decoded)
        self.solution = self._repair_time_windows(repaired)

    # -- repair helpers ---------------------------------------------------- #
    def _count_route_tw_violations(self, route) -> int:
        if not route:
            return 0
        p = self.problem
        v = 0
        t = max(0.0, p.time_windows[self.depot][0]) + p.time_costs[self.depot][route[0]]
        ready, due = p.time_windows[route[0]]
        if t > due:
            v += 1
        t = max(t, ready) + p.service_times[route[0]]
        for idx in range(len(route) - 1):
            t += p.time_costs[route[idx]][route[idx + 1]]
            ready, due = p.time_windows[route[idx + 1]]
            if t > due:
                v += 1
            t = max(t, ready) + p.service_times[route[idx + 1]]
        return v

    def _repair_solution(self, routes):
        p = self.problem
        all_customers = set(p.dests)
        visited = set()
        repaired = []
        for route in routes:
            clean = []
            for c in route:
                if c not in visited:
                    clean.append(c)
                    visited.add(c)
            if clean:
                repaired.append(clean)

        missing = all_customers - visited
        for customer in missing:
            if not repaired:
                repaired.append([customer])
                continue
            best_idx, best_pos, best_cost = -1, -1, float("inf")
            demand = p.weights.get(customer, 0)
            for idx, route in enumerate(repaired):
                load = sum(p.weights.get(c, 0) for c in route)
                cap = p.capacities[idx] if idx < len(p.capacities) else p.capacities[0]
                if load + demand > cap:
                    continue
                for pos in range(len(route) + 1):
                    cand = route[:pos] + [customer] + route[pos:]
                    if self._count_route_tw_violations(cand) > 0:
                        continue
                    prev = self.depot if pos == 0 else route[pos - 1]
                    nxt = self.depot if pos == len(route) else route[pos]
                    delta = p.costs[prev][customer] + p.costs[customer][nxt] - p.costs[prev][nxt]
                    if delta < best_cost:
                        best_cost, best_idx, best_pos = delta, idx, pos
            if best_idx != -1:
                r = repaired[best_idx]
                repaired[best_idx] = r[:best_pos] + [customer] + r[best_pos:]
            else:
                repaired.append([customer])
        return repaired

    def _repair_time_windows(self, routes):
        repaired = []
        for route in routes:
            if len(route) <= 1:
                repaired.append(route)
                continue
            best_route = list(route)
            best_v = self._count_route_tw_violations(best_route)
            if best_v == 0:
                repaired.append(best_route)
                continue
            if len(route) <= 8:
                for perm in permutations(route):
                    v = self._count_route_tw_violations(list(perm))
                    if v < best_v:
                        best_v, best_route = v, list(perm)
                        if best_v == 0:
                            break
            else:
                current = list(route)
                improved = True
                while improved:
                    improved = False
                    for a in range(len(current)):
                        for b in range(a + 1, len(current)):
                            cand = current[:]
                            cand[a], cand[b] = cand[b], cand[a]
                            if self._count_route_tw_violations(cand) < self._count_route_tw_violations(current):
                                current, improved = cand, True
                best_route = current
            repaired.append(best_route)
        return self._repair_inter_route(repaired)

    def _repair_inter_route(self, routes):
        p = self.problem
        routes = [list(r) for r in routes]

        def total_v(rts):
            return sum(self._count_route_tw_violations(r) for r in rts)

        improved = True
        while improved:
            improved = False
            if total_v(routes) == 0:
                break
            best_delta, best_move = 0, None
            for src_idx, src in enumerate(routes):
                if not src:
                    continue
                src_v = self._count_route_tw_violations(src)
                if src_v == 0:
                    continue
                for src_pos, customer in enumerate(src):
                    src_after = self._count_route_tw_violations(src[:src_pos] + src[src_pos + 1:])
                    demand = p.weights.get(customer, 0)
                    for dst_idx, dst in enumerate(routes):
                        if dst_idx == src_idx:
                            continue
                        cap = p.capacities[dst_idx] if dst_idx < len(p.capacities) else p.capacities[0]
                        if sum(p.weights.get(c, 0) for c in dst) + demand > cap:
                            continue
                        for dst_pos in range(len(dst) + 1):
                            dst_after = self._count_route_tw_violations(dst[:dst_pos] + [customer] + dst[dst_pos:])
                            delta = (src_after - src_v) + dst_after
                            if delta < best_delta:
                                best_delta, best_move = delta, (src_idx, src_pos, dst_idx, dst_pos)
            if best_move is not None:
                si, sp, di, dp = best_move
                c = routes[si][sp]
                routes[si] = routes[si][:sp] + routes[si][sp + 1:]
                routes[di] = routes[di][:dp] + [c] + routes[di][dp:]
                improved = True
        return [r for r in routes if r]

    # -- validation + cost ------------------------------------------------- #
    def check(self, routes=None) -> bool:
        """True iff routes are complete, non-duplicating, capacity- and TW-feasible."""
        p = self.problem
        routes = self.solution if routes is None else routes
        visited = set()
        for i, route in enumerate(routes):
            for c in route:
                if c in visited:
                    return False
                visited.add(c)
            if i < len(p.capacities) and sum(p.weights.get(c, 0) for c in route) > p.capacities[i]:
                return False
            if self._count_route_tw_violations(route) > 0:
                return False
        return visited == set(p.dests)

    def report(self, routes=None) -> dict:
        """Constraint report for a set of int-indexed routes (no repair applied).

        Used to measure *pre-repair* SA solution quality (pass ``raw_solution``):
        raw violation counts, duplicate/missing customers, and whether the raw
        decode is already feasible. Counting is done in problem/int space, so it
        works identically for full and coarsened graphs.
        """
        p = self.problem
        routes = self.raw_solution if routes is None else routes
        tw_viol = cap_viol = 0
        visits: list = []
        for i, route in enumerate(routes):
            visits.extend(route)
            cap = p.capacities[i] if i < len(p.capacities) else (p.capacities[0] if p.capacities else 0)
            if sum(p.weights.get(c, 0) for c in route) > cap:
                cap_viol += 1
            tw_viol += self._count_route_tw_violations(route)
        unique = set(visits)
        required = set(p.dests)
        duplicates = len(visits) - len(unique)
        missing = len(required - unique)
        feasible = (tw_viol == 0 and cap_viol == 0 and duplicates == 0
                    and missing == 0 and unique == required)
        return {
            "tw_violations": tw_viol,
            "cap_violations": cap_viol,
            "duplicates": duplicates,
            "missing": missing,
            "total_violations": tw_viol + cap_viol + duplicates + missing,
            "feasible": feasible,
        }

    def total_cost(self, routes=None) -> float:
        p = self.problem
        routes = self.solution if routes is None else routes
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += p.costs[self.depot][route[0]]
            for i in range(len(route) - 1):
                total += p.costs[route[i]][route[i + 1]]
            total += p.costs[route[-1]][self.depot]
        return total


# --------------------------------------------------------------------------- #
# FQS / APS solvers
# --------------------------------------------------------------------------- #
def _k_max_fqs(num_customers: int, num_vehicles: int) -> int:
    avg = math.ceil(num_customers / num_vehicles) if num_vehicles else 0
    return min(avg + 1, num_customers)


def _k_max_aps(num_customers: int, num_vehicles: int, limit_radius: int = 1) -> int:
    avg = math.ceil(num_customers / num_vehicles) if num_vehicles else 0
    return min(avg + limit_radius, num_customers)


def solve_vrp(problem: VRPProblem, solver: str, cfg: SolverConfig,
              penalties: PenaltyWeights | None = None,
              smart_capacity: bool = False) -> VRPSolution:
    """Solve a VRPProblem with 'FQS' or 'APS'. Returns a repaired VRPSolution."""
    penalties = penalties or PenaltyWeights()
    num_customers = len(problem.dests)
    num_vehicles = len(problem.capacities)
    if solver.upper() in ("FQS", "FULLQUBO"):
        k_max = _k_max_fqs(num_customers, num_vehicles)
    elif solver.upper() in ("APS", "AVERAGEPARTITION"):
        k_max = _k_max_aps(num_customers, num_vehicles)
    else:
        raise ValueError(f"Unknown solver: {solver}")
    vehicle_k_limits = [k_max] * num_vehicles

    qubo = problem.get_qubo(vehicle_k_limits, penalties, smart_capacity=smart_capacity)
    samples = solve_qubo(qubo, cfg, limit=1)
    if not samples:
        return VRPSolution(problem, {}, vehicle_k_limits, solution=[])
    return VRPSolution(problem, samples[0], vehicle_k_limits)
