"""Apply the repair once more to stored expanded routes, with the full time-window
check including the return to the depot.

  python experiments/fullcheck_repair.py <traces.jsonl> <dataset> <out.csv>
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import pandas as pd
from review_runner import load
from src.coarsening import graph_from_instance
from src.metrics import calculate_route_metrics
from src.qubo import vrp_problem_from_graph
from src.solvers import VRPSolution


class FullCheck(VRPSolution):
    def _count_route_tw_violations(self, route) -> int:
        v = super()._count_route_tw_violations(route)
        if not route:
            return v
        p = self.problem
        t = max(0.0, p.time_windows[self.depot][0]) + p.time_costs[self.depot][route[0]]
        t = max(t, p.time_windows[route[0]][0]) + p.service_times[route[0]]
        for a, b in zip(route, route[1:]):
            t += p.time_costs[a][b]
            t = max(t, p.time_windows[b][0]) + p.service_times[b]
        t += p.time_costs[route[-1]][self.depot]
        if t > p.time_windows[self.depot][1]:
            v += 1
        return v


def repair(cls, graph, depot, capacity, routes):
    nc = len(graph.nodes) - 1
    nv = max(2, nc // 2, len(routes))
    prob, int_to_id = vrp_problem_from_graph(graph, depot, capacity, num_vehicles=nv)
    id_to_int = {nid: i for i, nid in enumerate(int_to_id)}
    int_routes = [[id_to_int[c] for c in r if c != depot] for r in routes]
    sol = cls(prob, {}, None, solution=[])
    rep = sol._repair_time_windows(sol._repair_solution(int_routes))
    return [[depot] + [int_to_id[c] for c in r] + [depot] for r in rep if r]


traces, dataset, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows, cache = [], {}
for line in open(traces):
    r = json.loads(line)
    k = (r["instance"], r["N"])
    if k not in cache:
        inst = load(r["instance"], r["dataset"], r["c104"]).truncate(r["N"])
        g, d = graph_from_instance(inst)
        cache[k] = (g, d, inst.capacity)
    g, d, cap = cache[k]
    fin = r["final_routes"]
    m0 = calculate_route_metrics(g, fin, d, cap)
    t0 = time.perf_counter()
    rep = repair(FullCheck, g, d, cap, fin)
    dt = (time.perf_counter() - t0) * 1000
    m1 = calculate_route_metrics(g, rep, d, cap)
    rows.append(dict(instance=r["instance"], N=r["N"], seed=r["seed"], arm=r["arm"],
                     final_feasible=int(m0["is_feasible"]), final_tw=m0["time_window_violations"],
                     final_distance=m0["total_distance"], final_vehicles=m0["num_vehicles"],
                     full_feasible=int(m1["is_feasible"]), full_tw=m1["time_window_violations"],
                     full_cap=m1["capacity_violations"], full_visited=m1["num_customers_visited"],
                     full_distance=m1["total_distance"], full_vehicles=m1["num_vehicles"],
                     old_rep_feasible=int(r["final_rep"]["is_feasible"]), repair_ms=dt))
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", len(rows))
