"""Route metrics: distance, capacity/TW violations, validity, feasibility.

Ported from the predecessor's ``calculate_route_metrics`` (utils.py) onto our
Graph, with an explicit split we keep:

  * **validity**   -- the solution is well-formed: every customer visited exactly
    once and every route starts and ends at the depot. (Structural.)
  * **feasibility** -- validity AND no capacity or time-window violations.

Routes are lists of string node ids including the depot at both ends,
e.g. ['0', '3', '7', '0'], evaluated on a Graph (original or coarsened).
"""
from __future__ import annotations

from .coarsening import Graph, euclidean


def calculate_route_metrics(graph: Graph, routes: list, depot_id: str,
                            vehicle_capacity: float) -> dict:
    """Aggregate metrics for a set of routes on ``graph``."""
    empty = {
        "total_distance": 0.0, "total_service_time": 0.0, "total_waiting_time": 0.0,
        "total_route_duration": 0.0, "time_window_violations": 0,
        "capacity_violations": 0, "is_feasible": False, "is_valid": False,
        "num_vehicles": 0, "total_demand_served": 0.0, "num_customers_visited": 0,
    }
    if not routes:
        return empty

    total_distance = total_service = total_waiting = total_duration = 0.0
    tw_violations = cap_violations = num_vehicles = 0
    total_demand = 0.0
    all_feasible = True
    visited: list = []

    for route in routes:
        if not route or len(route) < 2 or (len(route) == 2 and route[0] == route[1] == depot_id):
            continue
        num_vehicles += 1
        load = 0.0
        t = graph.nodes[depot_id].e
        depot_arrival = None
        for i in range(len(route) - 1):
            frm = graph.nodes[route[i]]
            to_id = route[i + 1]
            to = graph.nodes[to_id]
            if to_id != depot_id:
                load += to.demand
                if load > vehicle_capacity:
                    cap_violations += 1
                    all_feasible = False
            travel = euclidean(frm, to)
            total_distance += travel
            arrival = t + travel
            service_start = max(arrival, to.e)
            if service_start > to.l:
                tw_violations += 1
                all_feasible = False
            total_waiting += max(0.0, to.e - arrival)
            t = service_start + to.s
            if to_id != depot_id:
                total_service += to.s
                total_demand += to.demand
                visited.append(to_id)
            else:
                depot_arrival = arrival
        if route[-1] == depot_id:
            if depot_arrival is not None:
                total_duration += depot_arrival
        else:
            all_feasible = False

    # validity: every customer visited exactly once, routes closed at depot
    unique = set(visited)
    is_valid = (len(visited) == len(unique)) and all(
        r[0] == depot_id and r[-1] == depot_id for r in routes if len(r) >= 2)
    is_feasible = all_feasible and cap_violations == 0 and tw_violations == 0 and is_valid

    return {
        "total_distance": total_distance,
        "total_service_time": total_service,
        "total_waiting_time": total_waiting,
        "total_route_duration": total_duration,
        "time_window_violations": tw_violations,
        "capacity_violations": cap_violations,
        "is_feasible": is_feasible,
        "is_valid": is_valid,
        "num_vehicles": num_vehicles,
        "total_demand_served": total_demand,
        "num_customers_visited": len(unique),
    }


def relative_cost(cost: float, reference: float) -> float:
    """(cost - reference) / reference; the optimality/quality gap vs a reference."""
    if reference is None or reference == 0:
        return float("nan")
    return (cost - reference) / reference
