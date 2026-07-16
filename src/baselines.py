"""Classical reference solvers (classical reference solvers).

OR-Tools routing solver for CVRPTW, used to:
  1. answer the question "how does the pipeline compare to a strong
     classical solver?" (optimality gap), and
  2. provide the near-optimal cost used to *normalise the reward* in the Phase-4
     downstream-reward GNN training.

Distances/times are Euclidean (float); OR-Tools needs integers, so we scale by
``SCALE`` and round. Costs returned are in original (unscaled) units.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .datasets import VRPInstance

SCALE = 100  # 2-decimal precision for integer arc costs/times


@dataclass
class BaselineSolution:
    feasible: bool
    cost: float                 # total Euclidean distance (unscaled)
    routes: list                # list of routes (customer index lists, no depot)
    num_vehicles_used: int
    status: str


def solve_cvrptw_ortools(inst: VRPInstance, time_limit_s: float = 5.0,
                         num_vehicles: Optional[int] = None) -> BaselineSolution:
    """Near-optimal CVRPTW solve via OR-Tools routing.

    Returns a BaselineSolution; ``cost`` is the total Euclidean route distance,
    the reference we compare the QUBO pipeline against.
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    n = inst.coords.shape[0]              # nodes incl depot at 0
    ncust = n - 1
    if num_vehicles is None:
        num_vehicles = max(2, ncust)      # enough vehicles: feasibility = routing, not fleet
    depot = 0

    D = inst.distance_matrix()            # float Euclidean
    Dint = np.round(D * SCALE).astype(int)
    service = (inst.service_time if inst.has_time_windows
               else np.zeros(n)).astype(float)
    ready = (inst.ready_time if inst.has_time_windows else np.zeros(n)).astype(float)
    due = (inst.due_time if inst.has_time_windows else np.full(n, 1e9)).astype(float)
    demand = inst.demand.astype(float)
    cap = int(round(inst.capacity))

    mgr = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(mgr)

    # arc cost = scaled distance
    def dist_cb(i, j):
        return int(Dint[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    # capacity dimension
    def demand_cb(i):
        return int(round(demand[mgr.IndexToNode(i)]))
    dem_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        dem_idx, 0, [cap] * num_vehicles, True, "Capacity")

    # time dimension: transit = travel + service(at from-node)
    def time_cb(i, j):
        fi = mgr.IndexToNode(i)
        return int(Dint[fi][mgr.IndexToNode(j)] + round(service[fi] * SCALE))
    time_idx = routing.RegisterTransitCallback(time_cb)
    horizon = int(round(max(due) * SCALE)) + 1
    routing.AddDimension(time_idx, horizon, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(n):
        idx = mgr.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(int(round(ready[node] * SCALE)),
                                        int(round(due[node] * SCALE)))
    # allow waiting: slack already permits it; minimize nothing extra on time

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return BaselineSolution(False, float("inf"), [], 0, "INFEASIBLE")

    routes, used = [], 0
    total = 0.0
    for v in range(num_vehicles):
        idx = routing.Start(v)
        route = []
        prev_node = mgr.IndexToNode(idx)
        while not routing.IsEnd(idx):
            nxt = sol.Value(routing.NextVar(idx))
            node = mgr.IndexToNode(idx)
            nnode = mgr.IndexToNode(nxt)
            if node != depot:
                route.append(node)
            total += D[node][nnode]
            idx = nxt
        if route:
            routes.append(route)
            used += 1
    return BaselineSolution(True, float(total), routes, used, "OK")
