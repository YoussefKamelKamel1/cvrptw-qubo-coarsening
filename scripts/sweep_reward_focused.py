"""Do the reward tiebreaker terms actually change the outcome, or only the oracle choice?

The full sweep in `sweep_reward.py` showed that the reward's coefficient magnitudes
barely change which candidate wins: quadrupling the gap weight moves 3 oracle
choices out of 129, and a tenfold violation weight moves none. Those settings
therefore cannot differ downstream for any causal reason, and retraining them
measures training noise rather than an effect.

Exactly one setting genuinely differs. Dropping both tiebreaker terms
("feasibility only") flips 74 of 129 oracle choices and collapses the GNN's
oracle win rate from 35% to 2%, because with feasibility alone most candidates tie
and ties fall to whichever came first in the pool, which is the heuristic. That
would reduce expert iteration to plain imitation.

This script measures that one comparison properly: both settings, several training
seeds each, so the difference can be read against the seed-to-seed spread instead
of a single run. Everything else is held fixed.

  python scripts/sweep_reward_focused.py --train-seeds 0 1 2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

from compare_architectures import train_one, downstream               # noqa: E402
from sweep_reward import SETTINGS, labels_for                         # noqa: E402
from src.datasets import train_test_split_instances                   # noqa: E402
from src.utils import set_global_seed                                 # noqa: E402

DEFAULT_PAIR = ["published", "feasibility only"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="results/label_cache.pt")
    ap.add_argument("--settings", nargs="+", default=DEFAULT_PAIR)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--Ns", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--num-reads", type=int, default=200)
    ap.add_argument("--num-sweeps", type=int, default=200)
    ap.add_argument("--out", default="results/reward_sweep_focused.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(ROOT / args.cache, weights_only=False)
    cache = blob["cache"]
    _, test = train_test_split_instances()
    chosen = [s for s in SETTINGS if s[0] in args.settings]
    if len(chosen) != len(args.settings):
        raise SystemExit(f"unknown setting; available: {[s[0] for s in SETTINGS]}")

    print(f"device={device} | {len(cache)} instance-size pairs | "
          f"settings={[c[0] for c in chosen]} | train seeds={args.train_seeds}\n",
          flush=True)

    rows = []
    for name, *w in chosen:
        examples, sel = labels_for(cache, w)
        print(f"== {name}  weights={w} ==")
        print(f"  oracle picked GNN {100*sel['gnn_win_rate']:.0f}% of the time, "
              f"{len(examples)} labelled graphs", flush=True)
        for ts in args.train_seeds:
            targs = SimpleNamespace(epochs=args.epochs, lr=3e-3, hidden=64,
                                    layers=2, val_frac=0.15, seed=ts)
            model, _ = train_one("sage", examples, targs, device)
            model.eval()
            for N in args.Ns:
                got = []
                for inst_name in test:
                    for sd in args.seeds:
                        set_global_seed(sd)
                        r = downstream(model, inst_name, N, device, sd,
                                       args.num_reads, args.num_sweeps)
                        rows.append(dict(setting=name, train_seed=ts, N=N,
                                         instance=inst_name, seed=sd, **r))
                        got.append(r)
                print(f"    seed {ts} N={N:3d}  "
                      f"feasible={100*np.mean([g['feasible'] for g in got]):5.1f}%  "
                      f"pre-feasible={100*np.mean([g['pre_feasible'] for g in got]):5.1f}%",
                      flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    def stat(name, N, field):
        per_seed = [100 * np.mean([r[field] for r in rows
                                   if r["setting"] == name and r["train_seed"] == ts
                                   and r["N"] == N])
                    for ts in args.train_seeds]
        return float(np.mean(per_seed)), float(np.std(per_seed))

    print(f"\n=== downstream feasibility (%), mean over {len(args.train_seeds)} "
          "training seeds, +/- spread ===")
    names = [c[0] for c in chosen]
    print(f"{'N':>4} " + " ".join(f"{n:>20}" for n in names))
    for N in args.Ns:
        line = f"{N:>4} "
        for n in names:
            m, s = stat(n, N, "feasible")
            line += f"  {m:8.1f} +/-{s:5.1f}   "
        print(line)

    print("\n=== pre-repair feasibility (%) ===")
    print(f"{'N':>4} " + " ".join(f"{n:>20}" for n in names))
    for N in args.Ns:
        line = f"{N:>4} "
        for n in names:
            m, s = stat(n, N, "pre_feasible")
            line += f"  {m:8.1f} +/-{s:5.1f}   "
        print(line)

    if len(names) == 2:
        print("\n=== verdict: is the difference bigger than the seed spread? ===")
        for N in args.Ns:
            a, sa = stat(names[0], N, "feasible")
            b, sb = stat(names[1], N, "feasible")
            d = a - b
            pooled = max(sa, sb)
            v = "clear" if abs(d) > 2 * pooled else "within noise"
            print(f"  N={N:3d}: {names[0]} - {names[1]} = {d:+.1f} pp, "
                  f"spread {pooled:.1f} pp -> {v}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
