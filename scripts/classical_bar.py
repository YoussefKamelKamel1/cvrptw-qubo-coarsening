"""Matched classical samplers on the same coarsened QUBOs: the bar a QPU must clear.

For each coarsening (none / heuristic / gnn) the adaptive-penalty QUBO is solved by
simulated annealing, tabu search, steepest descent, and uniform random (a floor),
all on the identical logical problem. Quality is scored with the same full-
distribution, pre-repair metrics as the QPU runner (raw feasibility, mean/min raw
violations, post-repair feasibility, best/mean feasible distance, best energy) plus
wall-clock time, so "kernel utility" and "equal wall-clock" claims have a reference.

  python scripts/classical_bar.py --N 10 20 --n-instances 3 --num-reads 1000
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch                                               # noqa: E402
from dwave.samplers import (SimulatedAnnealingSampler,     # noqa: E402
                            TabuSampler, SteepestDescentSampler)
from dimod import RandomSampler                            # noqa: E402

from run_qpu import build_problem, evaluate_samples        # noqa: E402
from src.gnn_coarsening import load_model                  # noqa: E402
from src.datasets import train_test_split_instances        # noqa: E402


def _call(sampler, Q, **kw):
    """Sample, dropping kwargs the sampler does not accept (e.g. seed)."""
    try:
        return sampler.sample_qubo(Q, **kw)
    except TypeError:
        kw.pop("seed", None)
        return sampler.sample_qubo(Q, **kw)


def run_sampler(name, Q, reads, sweeps, seed):
    t0 = time.perf_counter()
    if name == "sa":
        ss = _call(SimulatedAnnealingSampler(), Q, num_reads=reads,
                   num_sweeps=sweeps, seed=seed)
    elif name == "tabu":
        ss = _call(TabuSampler(), Q, num_reads=reads, seed=seed)
    elif name == "greedy":
        ss = _call(SteepestDescentSampler(), Q, num_reads=reads, seed=seed)
    elif name == "random":
        ss = _call(RandomSampler(), Q, num_reads=reads)
    else:
        raise ValueError(name)
    return ss, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--coarsen", nargs="+", default=["none", "heuristic", "gnn"])
    ap.add_argument("--samplers", nargs="+", default=["sa", "tabu", "greedy", "random"])
    ap.add_argument("--instances", nargs="+", default=None)
    ap.add_argument("--n-instances", type=int, default=3)
    ap.add_argument("--num-reads", type=int, default=1000)
    ap.add_argument("--num-sweeps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="results/gnn_model_rl.pt")
    ap.add_argument("--out", default="results/classical_bar.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ROOT / args.model, device=device)
    _, test = train_test_split_instances()
    names = args.instances or test[:args.n_instances]
    print(f"instances={names} | N={args.N} | coarsen={args.coarsen} | "
          f"samplers={args.samplers} | reads={args.num_reads}\n", flush=True)

    rows = []
    t0 = time.perf_counter()
    for N in args.N:
        for coarsen in args.coarsen:
            for name in names:
                inst, depot, coarsener, problem, int_to_id, qubo, k_limits = build_problem(
                    name, N, coarsen, "adaptive", model, device)
                nv = len({v for key in qubo.dict for v in key})
                for s in args.samplers:
                    ss, dt = run_sampler(s, qubo.dict, args.num_reads,
                                         args.num_sweeps, args.seed)
                    ev = evaluate_samples(ss, problem, coarsener, inst,
                                          int_to_id, depot, k_limits)
                    row = dict(N=N, coarsen=coarsen, instance=name, sampler=s,
                               n_vars=nv, time_s=round(dt, 3), **ev)
                    rows.append(row)
                    print(f"  N={N:2d} {coarsen:9s} {name:5s} {s:7s} "
                          f"vars={nv:4d} t={dt:6.2f}s raw_feas={ev['raw_feas_frac']} "
                          f"post_feas={ev['post_feas_frac']} best_dist={ev['best_dist']}",
                          flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\ndone in {time.perf_counter()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
