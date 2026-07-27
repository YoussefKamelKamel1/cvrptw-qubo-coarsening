"""Classical reverse-annealing proxy: can reheating escape a classical basin?

Reverse annealing on a QPU starts from a known good state, melts partially, and
refreezes. The classical analog here: take the lowest-energy forward-SA sample as
the seed, then re-run SA *from that seed* at a partial reheat temperature (a
restricted neighborhood), for several reheat strengths. If reheating never reaches
a lower energy / higher raw-feasibility than forward search from scratch, quantum
reverse annealing is unlikely to help on this landscape either -- a cheap way to
decide whether the QPU reverse-anneal experiment is worth running.

Reheat is a fraction of the forward run's own beta range on a log scale:
frac=0 is full melt (equivalent to forward), frac=1 stays frozen at the seed.

  python scripts/reverse_anneal_proxy.py --N 10 20 --n-instances 3
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch                                               # noqa: E402
from dwave.samplers import SimulatedAnnealingSampler       # noqa: E402

from run_qpu import build_problem, evaluate_samples        # noqa: E402
from src.gnn_coarsening import load_model                  # noqa: E402
from src.datasets import train_test_split_instances        # noqa: E402


def _beta_range(sampleset, Q):
    """Forward run's beta range, from info if present else a coefficient estimate."""
    br = sampleset.info.get("beta_range") if hasattr(sampleset, "info") else None
    if br and len(br) == 2 and br[0] > 0:
        return float(br[0]), float(br[1])
    mags = [abs(v) for v in Q.values() if v != 0]
    if not mags:
        return 0.1, 10.0
    return 1.0 / max(mags), 1.0 / min(mags)


def reheat_beta(bmin, bmax, frac):
    """Log-interp start temperature: frac=0 -> bmin (hot), frac=1 -> bmax (cold)."""
    return math.exp(math.log(bmin) + frac * (math.log(bmax) - math.log(bmin)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--coarsen", nargs="+", default=["gnn"])
    ap.add_argument("--instances", nargs="+", default=None)
    ap.add_argument("--n-instances", type=int, default=3)
    ap.add_argument("--num-reads", type=int, default=200)
    ap.add_argument("--num-sweeps", type=int, default=200)
    ap.add_argument("--reheat-fracs", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="results/gnn_model_rl.pt")
    ap.add_argument("--out", default="results/reverse_anneal.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ROOT / args.model, device=device)
    _, test = train_test_split_instances()
    names = args.instances or test[:args.n_instances]
    sa = SimulatedAnnealingSampler()
    print(f"instances={names} | N={args.N} | coarsen={args.coarsen} | "
          f"reheat={args.reheat_fracs}\n", flush=True)

    rows = []
    t0 = time.perf_counter()
    for N in args.N:
        for coarsen in args.coarsen:
            for name in names:
                inst, depot, coarsener, problem, int_to_id, qubo, k_limits = build_problem(
                    name, N, coarsen, "adaptive", model, device)
                Q = qubo.dict

                # forward baseline
                fwd = sa.sample_qubo(Q, num_reads=args.num_reads,
                                     num_sweeps=args.num_sweeps, seed=args.seed)
                ev_f = evaluate_samples(fwd, problem, coarsener, inst,
                                        int_to_id, depot, k_limits)
                seed_sample = dict(next(iter(fwd.lowest().samples())))
                bmin, bmax = _beta_range(fwd, Q)
                rows.append(dict(N=N, coarsen=coarsen, instance=name, mode="forward",
                                 reheat="-", best_energy=ev_f["best_energy"],
                                 raw_feas=ev_f["raw_feas_frac"],
                                 min_raw_viol=ev_f["min_raw_viol"],
                                 best_dist=ev_f["best_dist"]))
                print(f"  N={N:2d} {coarsen} {name:5s} forward      "
                      f"E={ev_f['best_energy']} raw_feas={ev_f['raw_feas_frac']} "
                      f"best_dist={ev_f['best_dist']}", flush=True)

                # reverse proxy at several reheat strengths
                for frac in args.reheat_fracs:
                    b0 = reheat_beta(bmin, bmax, frac)
                    try:
                        rev = sa.sample_qubo(
                            Q, num_reads=args.num_reads, num_sweeps=args.num_sweeps,
                            beta_range=[b0, bmax], initial_states=seed_sample,
                            initial_states_generator="tile", seed=args.seed)
                    except TypeError:
                        rev = sa.sample_qubo(
                            Q, num_reads=args.num_reads, num_sweeps=args.num_sweeps,
                            beta_range=[b0, bmax], initial_states=seed_sample)
                    ev_r = evaluate_samples(rev, problem, coarsener, inst,
                                            int_to_id, depot, k_limits)
                    rows.append(dict(N=N, coarsen=coarsen, instance=name,
                                     mode="reverse", reheat=frac,
                                     best_energy=ev_r["best_energy"],
                                     raw_feas=ev_r["raw_feas_frac"],
                                     min_raw_viol=ev_r["min_raw_viol"],
                                     best_dist=ev_r["best_dist"]))
                    better = (ev_r["best_energy"] is not None
                              and ev_f["best_energy"] is not None
                              and ev_r["best_energy"] < ev_f["best_energy"] - 1e-9)
                    print(f"  N={N:2d} {coarsen} {name:5s} reverse f={frac} "
                          f"E={ev_r['best_energy']} raw_feas={ev_r['raw_feas_frac']} "
                          f"best_dist={ev_r['best_dist']}"
                          f"{'  <-- beats forward' if better else ''}", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\ndone in {time.perf_counter()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
