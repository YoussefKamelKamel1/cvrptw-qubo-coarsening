"""QUBO formulations for CVRPTW.

Follows the FQS/APS QUBO of Borowski, Gora et al., ICCS 2020 ("New Hybrid Quantum
Annealing Algorithms for Solving VRP"). The variable is x_{i,j,k} = 1 iff vehicle
i visits customer j at step k.

The constraint hierarchy (strict ordering):
    visit-uniqueness > capacity / self-loop > time-window > continuity > distance
is expressed through ``PenaltyWeights``. Adaptive calibration scales these to the
instance's distance magnitude and per coarsening level; the static defaults are
the fixed baseline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .coarsening import Graph, euclidean


class Qubo:
    """Sparse QUBO as a dict of {(var,var) or (var1,var2): coeff}."""

    def __init__(self):
        self.dict: dict = {}

    def add(self, key, value):
        if isinstance(key, tuple) and len(key) == 2 and not isinstance(key[0], int):
            # a pair of variables (each variable is itself a 3-tuple or ('s',i,m))
            try:
                key = tuple(sorted(key))
            except TypeError:
                key = tuple(sorted(key, key=str))
        self.dict[key] = self.dict.get(key, 0) + value

    def add_quadratic_equality_constraint(self, linear_expression, constant, penalty):
        """penalty * (sum_a coeff_a x_a + constant)^2, dropping the constant^2 term."""
        for coeff, var in linear_expression:
            self.add((var, var), penalty * (coeff * coeff + 2 * constant * coeff))
        for a in range(len(linear_expression)):
            for b in range(a + 1, len(linear_expression)):
                c1, v1 = linear_expression[a]
                c2, v2 = linear_expression[b]
                self.add((v1, v2), penalty * (2 * c1 * c2))


@dataclass
class PenaltyWeights:
    """QUBO penalty hierarchy. Phase-1 defaults = predecessor's static values."""

    only_one: float = 10_000_000.0        # visit-uniqueness (top of hierarchy)
    capacity_penalty: float = 5_000_000.0
    time_window_penalty: float = 3_000_000.0
    vehicle_start_cost: float = 100_000.0
    order: float = 100.0                   # distance/objective weight (bottom)

    @property
    def continuity_penalty(self) -> float:
        return self.only_one * 0.1

    @property
    def selfloop_penalty(self) -> float:
        return self.only_one * 0.5


@dataclass
class PenaltyCalibration:
    """Phase-2 adaptive penalty calibration.

    Instead of fixed absolute penalties, scale them to the instance's objective
    magnitude ``B`` (an upper bound on the total routing objective) so the QUBO's
    dynamic range stays bounded regardless of instance size or coarsening level.
    Multipliers preserve the strict hierarchy
        unique > capacity/self-loop > time-window > continuity > distance.

    Because ``B`` is recomputed from whatever cost matrix is passed,
    ``adaptive_penalties`` naturally *re-scales per coarsening level* (a coarsened
    graph has fewer/smaller cost terms, so its penalties shrink in step).
    """

    order: float = 100.0            # distance weight = objective unit (fixed)
    tw_mult: float = 5.0            # time-window penalty  = tw_mult  * B
    cap_mult: float = 8.0          # capacity penalty     = cap_mult * B
    unique_mult: float = 15.0      # visit-uniqueness      = unique_mult * B
    vehicle_start_mult: float = 1.0  # * (order * max_edge), objective-scale
    # Phase-2b: when capacity genuinely binds (total demand > Q), scale the
    # capacity penalty up with the demand pressure total_demand/Q, capped so it
    # never exceeds the uniqueness weight. Fixes the RC1 regression where a
    # binding constraint plus coarsening's demand aggregation left cap too weak.
    binding_boost: float = 2.0


def adaptive_penalties(problem: "VRPProblem",
                       cal: PenaltyCalibration | None = None) -> PenaltyWeights:
    """Calibrate penalties to a problem's objective magnitude (Phase 2).

    B = order * max_edge * num_customers is an upper bound on the total routing
    objective; each penalty is a small multiple of B, so a single violation
    always outweighs any achievable objective gain (with margin) while keeping
    the coupling dynamic range ~O(unique_mult * num_customers) instead of the
    static ~1e7/objective (which grows unbounded as instances shrink).

    A binding capacity constraint is the one case that survives ``smart_capacity``
    and so keeps its log-slack; the weight is normalised by the squared slack
    coefficient there (see below) to hold the same dynamic range.
    """
    cal = cal or PenaltyCalibration()
    n = len(problem.costs)
    max_edge = 0.0
    for i in range(n):
        row = problem.costs[i]
        for j in range(n):
            if i != j and row[j] > max_edge:
                max_edge = row[j]
    max_edge = max(max_edge, 1.0)
    num_customers = max(1, len(problem.dests))
    B = cal.order * max_edge * num_customers

    # binding-aware capacity weight (Phase-2b)
    cap_mult = cal.cap_mult
    slack_norm = 1.0
    total_demand = sum(problem.weights.get(j, 0) for j in problem.dests)
    Q = problem.capacities[0] if problem.capacities else 0
    if Q > 0 and total_demand > Q:
        cap_mult = min(cal.unique_mult,
                       cal.cap_mult * cal.binding_boost * (total_demand / Q))
        # A binding constraint is kept rather than dropped, so it brings its log
        # slack along. The largest slack coupling scales as (2^m_max)^2, which
        # would lift the capacity block orders of magnitude above every other
        # term -- the dynamic range this calibration exists to control. Divide it
        # out so the block lands at the scale of the remaining penalties.
        slack_norm = float(2 ** math.floor(math.log2(max(1.0, Q)))) ** 2

    return PenaltyWeights(
        only_one=cal.unique_mult * B,
        capacity_penalty=cap_mult * B / slack_norm,
        time_window_penalty=cal.tw_mult * B,
        vehicle_start_cost=cal.vehicle_start_mult * cal.order * max_edge,
        order=cal.order,
    )


class VRPProblem:
    """Integer-indexed CVRPTW instance + FQS/APS QUBO builder.

    Nodes are integers 0..num_nodes-1 with ``source_depot`` the depot index.
    ``costs``/``time_costs`` are full matrices; ``weights``/``time_windows``/
    ``service_times`` are dicts keyed by node index.
    """

    def __init__(self, source_depot, costs, time_costs, capacities, dests,
                 weights, time_windows, service_times):
        self.source_depot = source_depot
        self.costs = costs
        self.time_costs = time_costs
        self.capacities = capacities
        self.dests = dests
        self.weights = weights
        self.time_windows = time_windows
        self.service_times = service_times

        # True earliest feasible arrival: cannot beat driving straight from depot.
        self.true_earliest = {}
        depot_start = self.time_windows[self.source_depot][0]
        for j in self.dests:
            self.true_earliest[j] = max(self.time_windows[j][0],
                                        depot_start + self.time_costs[self.source_depot][j])

    def get_qubo(self, vehicle_k_limits, penalties: PenaltyWeights,
                 smart_capacity: bool = False) -> Qubo:
        """Build the FQS/APS QUBO.

        smart_capacity (Phase 2): skip capacity constraints that cannot bind
        (total customer demand <= vehicle capacity) and tighten the log-slack to
        the realistic load range. This removes the dominant (2^m)^2 * cap_penalty
        couplings -- pure noise for high-capacity families -- which otherwise
        dominate the QUBO's internal dynamic range and hurt SA at equal budget.
        """
        num_vehicles = len(self.capacities)
        customer_nodes = self.dests
        pw = penalties
        qubo = Qubo()
        total_demand = sum(self.weights.get(j, 0) for j in customer_nodes)

        # 1. UNIQUE VISITS ---------------------------------------------------
        ps = pw.only_one
        for j in customer_nodes:
            variables = [(i, j, k) for i in range(num_vehicles)
                         for k in range(vehicle_k_limits[i])]
            for var in variables:
                qubo.add((var, var), -ps)
            for a in range(len(variables)):
                for b in range(a + 1, len(variables)):
                    qubo.add((variables[a], variables[b]), 2 * ps)
        # each vehicle at most one customer per step
        for i in range(num_vehicles):
            for k in range(vehicle_k_limits[i]):
                variables = [(i, j, k) for j in customer_nodes]
                for a in range(len(variables)):
                    for b in range(a + 1, len(variables)):
                        qubo.add((variables[a], variables[b]), ps)

        # 2. CONTINUITY + SELF-LOOPS ----------------------------------------
        cont = pw.continuity_penalty
        selfloop = pw.selfloop_penalty
        for i in range(num_vehicles):
            for k in range(1, vehicle_k_limits[i]):
                for j in customer_nodes:
                    var_k = (i, j, k)
                    qubo.add((var_k, var_k), cont)
                    for j_prev in customer_nodes:
                        qubo.add((var_k, (i, j_prev, k - 1)), -cont * 0.5)
            for k in range(vehicle_k_limits[i] - 1):
                for j in customer_nodes:
                    qubo.add(((i, j, k), (i, j, k + 1)), selfloop)

        # 3. CAPACITY (log-slack equality) ----------------------------------
        for i in range(num_vehicles):
            capacity = self.capacities[i]
            if capacity <= 0:
                continue
            # Phase-2: drop non-binding capacity constraints (no subset of
            # customers can exceed capacity), and size slack to the realistic
            # load range instead of the raw capacity.
            if smart_capacity and total_demand <= capacity:
                continue
            slack_cap = min(capacity, total_demand) if smart_capacity else capacity
            slack_cap = max(1.0, slack_cap)
            num_slack_bits = math.floor(math.log2(slack_cap)) + 1
            slack_vars = [("s", i, m) for m in range(num_slack_bits)]
            expr = []
            for j in customer_nodes:
                demand = self.weights.get(j, 0)
                for k in range(vehicle_k_limits[i]):
                    expr.append((demand, (i, j, k)))
            for m in range(num_slack_bits):
                expr.append((2 ** m, slack_vars[m]))
            qubo.add_quadratic_equality_constraint(expr, -capacity, pw.capacity_penalty)

        # 4. TIME WINDOWS (physics-aware, 3 tiers) --------------------------
        twp = pw.time_window_penalty
        # A. depot initial check
        for i in range(num_vehicles):
            for j1 in customer_nodes:
                if self.true_earliest[j1] > self.time_windows[j1][1]:
                    qubo.add(((i, j1, 0), (i, j1, 0)), twp)
        # B. pairwise k -> k+1
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            for j1 in customer_nodes:
                for j2 in customer_nodes:
                    if j1 == j2:
                        continue
                    earliest_dep_j1 = self.true_earliest[j1] + self.service_times[j1]
                    earliest_arr_j2 = earliest_dep_j1 + self.time_costs[j1][j2]
                    if earliest_arr_j2 > self.time_windows[j2][1]:
                        for k in range(k_max - 1):
                            qubo.add(((i, j1, k), (i, j2, k + 1)), twp)
                    else:
                        latest_dep_j1 = self.time_windows[j1][1] + self.service_times[j1]
                        latest_arr_j2 = latest_dep_j1 + self.time_costs[j1][j2]
                        if latest_arr_j2 > self.time_windows[j2][1]:
                            span = max(1.0, self.time_windows[j1][1] - self.time_windows[j1][0])
                            frac = min(1.0, (latest_arr_j2 - self.time_windows[j2][1]) / span)
                            tight = twp * frac * 0.5
                            for k in range(k_max - 1):
                                qubo.add(((i, j1, k), (i, j2, k + 1)), tight)
        # C. triangle lookahead k -> k+2
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            if k_max < 3:
                continue
            for j1 in customer_nodes:
                for j3 in customer_nodes:
                    if j1 == j3:
                        continue
                    earliest_leave_j1 = self.true_earliest[j1] + self.service_times[j1]
                    possible = False
                    for j2 in customer_nodes:
                        if j2 == j1 or j2 == j3:
                            continue
                        arr_j2 = earliest_leave_j1 + self.time_costs[j1][j2]
                        if arr_j2 > self.time_windows[j2][1]:
                            continue
                        leave_j2 = max(arr_j2, self.time_windows[j2][0]) + self.service_times[j2]
                        if leave_j2 + self.time_costs[j2][j3] <= self.time_windows[j3][1]:
                            possible = True
                            break
                    if not possible:
                        for k in range(k_max - 2):
                            qubo.add(((i, j1, k), (i, j3, k + 2)), twp)

        # 5. OBJECTIVE (Clarke-Wright savings) ------------------------------
        oc = pw.order
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            for k in range(k_max):
                for j in customer_nodes:
                    round_trip = self.costs[self.source_depot][j] + self.costs[j][self.source_depot]
                    val = round_trip * oc
                    if k == 0:
                        val += pw.vehicle_start_cost
                    qubo.add(((i, j, k), (i, j, k)), val)
            for k in range(k_max - 1):
                for j1 in customer_nodes:
                    for j2 in customer_nodes:
                        if j1 == j2:
                            continue
                        savings = (self.costs[j1][j2]
                                   - self.costs[j1][self.source_depot]
                                   - self.costs[self.source_depot][j2])
                        qubo.add(((i, j1, k), (i, j2, k + 1)), savings * oc)
        return qubo


def vrp_problem_from_graph(graph: Graph, depot_id: str, capacity: float,
                           num_vehicles: int | None = None) -> tuple[VRPProblem, list]:
    """Build an integer-indexed VRPProblem from a (possibly coarsened) Graph.

    Returns (problem, int_to_id) where int_to_id[k] is the original string id
    of integer node k (index 0 = depot). Fleet size defaults to the
    predecessor's ``max(2, num_customers // 2)``.
    """
    customer_ids = sorted([nid for nid in graph.nodes if nid != depot_id])
    int_to_id = [depot_id] + customer_ids
    id_to_int = {nid: i for i, nid in enumerate(int_to_id)}
    num_nodes = len(int_to_id)

    costs = [[0.0] * num_nodes for _ in range(num_nodes)]
    time_costs = [[0.0] * num_nodes for _ in range(num_nodes)]
    demands, time_windows, service_times = {}, {}, {}
    for u_id, u in graph.nodes.items():
        ui = id_to_int[u_id]
        demands[ui] = u.demand
        time_windows[ui] = (u.e, u.l)
        service_times[ui] = u.s
        for v_id, v in graph.nodes.items():
            vi = id_to_int[v_id]
            tau = 0.0 if u_id == v_id else euclidean(u, v)
            costs[ui][vi] = tau
            time_costs[ui][vi] = tau

    num_customers = len(customer_ids)
    if num_vehicles is None:
        num_vehicles = max(2, num_customers // 2)
    problem = VRPProblem(
        source_depot=id_to_int[depot_id], costs=costs, time_costs=time_costs,
        capacities=[capacity] * num_vehicles, dests=[id_to_int[c] for c in customer_ids],
        weights=demands, time_windows=time_windows, service_times=service_times,
    )
    return problem, int_to_id
