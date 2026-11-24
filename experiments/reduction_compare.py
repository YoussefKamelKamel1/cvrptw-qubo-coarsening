"""Roof duality on the position QUBOs, with and without coarsening (held-out instances,
N = 10, 20, 40, static and adaptive penalties; Supplementary Table S7).
"""
import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import dimod                                                        # noqa: E402
from dwave.preprocessing import roof_duality                       # noqa: E402

from run_qpu import build_problem                                  # noqa: E402
from src.datasets import train_test_split_instances                # noqa: E402
from src.gnn_coarsening import load_model                          # noqa: E402
from src.utils import set_global_seed                              # noqa: E402

model = load_model(str(ROOT / "results" / "gnn_model_rl.pt"), device="cpu")

# Use the corrected C104 (the raw file lacks customer 36), as in the main runs.
import run_qpu                                                      # noqa: E402
from review_runner import load as _rr_load                          # noqa: E402
run_qpu.load_solomon = lambda name: _rr_load(name, "solomon", "fixed")
rows = []
t_all = time.perf_counter()
for name in train_test_split_instances()[1]:
    for N in (10, 20, 40):
        for coarsen in ("none", "heuristic", "gnn"):
            for pen in ("static", "adaptive"):
                set_global_seed(0)
                *_, qubo, _ = build_problem(name, N, coarsen, pen, model, "cpu")
                bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.dict)
                n = bqm.num_variables
                t0 = time.perf_counter()
                _, fixed_s = roof_duality(bqm, strict=True)
                _, fixed_w = roof_duality(bqm, strict=False)
                rows.append(dict(instance=name, N=N, coarsen=coarsen, penalties=pen, n_vars=n,
                                 fixed_strict=len(fixed_s), fixed_weak=len(fixed_w),
                                 rd_time_s=round(time.perf_counter() - t0, 3)))
                print(rows[-1], flush=True)
out = HERE / "results" / "reduction_compare.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"wrote {out} in {time.perf_counter() - t_all:.0f}s")
