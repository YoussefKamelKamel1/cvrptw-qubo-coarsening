"""Adaptive calibration with each multiplier halved or doubled (GNN coarsening,
held-out instances, N = 10 and 20, three seeds; Supplementary Table S6).
"""
import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

from review_runner import load                                     # noqa: E402
from src.coarsening import graph_from_instance                     # noqa: E402
from src.datasets import train_test_split_instances                # noqa: E402
from src.experiment import solve_on_graph                          # noqa: E402
from src.gnn_coarsening import GNNCoarsener, load_model            # noqa: E402
from src.qubo import PenaltyCalibration, PenaltyWeights            # noqa: E402
from src.solvers import SolverConfig                               # noqa: E402
from src.utils import set_global_seed                              # noqa: E402

BASE = dict(unique_mult=15.0, cap_mult=8.0, tw_mult=5.0, cont_ratio=0.1)
SETTINGS = [("published", {})]
for k in BASE:
    for f in (0.5, 2.0):
        SETTINGS.append((f"{k} x{f:g}", {k: BASE[k] * f}))

model = load_model(str(ROOT / "results" / "gnn_model_rl.pt"), device="cpu")
rows = []
t_all = time.perf_counter()
for label, change in SETTINGS:
    s = {**BASE, **change}
    ratio = s["cont_ratio"]
    PenaltyWeights.continuity_penalty = property(lambda self, r=ratio: self.only_one * r)
    cal = PenaltyCalibration(unique_mult=s["unique_mult"], cap_mult=s["cap_mult"],
                             tw_mult=s["tw_mult"])
    for name in train_test_split_instances()[1]:
        for N in (10, 20):
            for seed in (0, 1, 2):
                set_global_seed(seed)
                inst = load(name, "solomon", "fixed").truncate(N)
                graph, depot = graph_from_instance(inst)
                co = GNNCoarsener(graph, P=0.5, radiusCoeff=0.5, depot_id=depot,
                                  model=model, device="cpu")
                coarse, _ = co.coarsen()
                _, m, _ = solve_on_graph(coarse, depot, inst.capacity, "FQS",
                                         SolverConfig("sa", 200, 200, seed=seed),
                                         penalty_mode="adaptive", penalty_cal=cal,
                                         coarsener=co)
                other = m["pre_total_viol"] - m["pre_tw_viol"] - m["pre_cap_viol"]
                rows.append(dict(setting=label, instance=name, N=N, seed=seed,
                                 pre_feasible=bool(m["pre_feasible"]),
                                 pre_tw=int(m["pre_tw_viol"]), pre_cap=int(m["pre_cap_viol"]),
                                 pre_other=int(other), final_feasible=bool(m["is_feasible"]),
                                 vehicles=int(m["num_vehicles"]), distance=float(m["total_distance"])))
    print(label, f"{time.perf_counter() - t_all:.0f}s", flush=True)
out = HERE / "results" / "penalty_sensitivity.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"wrote {out} in {time.perf_counter() - t_all:.0f}s")
