"""Does the expert-iteration reward depend on its coefficients, or only on the ordering?

The reference reward is
    r = 2*feasible + 1*pre_feasible - 0.5*min(gap,2) - 0.02*min(violations,50)
and the coefficients were set by hand to encode a priority ordering rather than
tuned. This sweep tests whether they matter.

Using the cache from `gen_labels_cached.py`, each weighting is applied offline to
already-measured candidates, so only retraining and evaluation repeat. For each
setting we rebuild the label set, retrain the scorer, and run the real pipeline on
held-out instances.

Reads the cache only; trains with the same loop as the architecture comparison.
No existing module is modified.

  python scripts/gen_labels_cached.py --out results/label_cache.pt
  python scripts/sweep_reward.py --cache results/label_cache.pt
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

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

from compare_architectures import train_one, downstream                # noqa: E402
from src.datasets import train_test_split_instances                    # noqa: E402
from src.utils import set_global_seed                                  # noqa: E402

# (name, w_feasible, w_pre_feasible, w_gap, w_violations)
SETTINGS = [
    ("published",      2.0, 1.0, 0.50, 0.02),
    ("feasibility only", 2.0, 1.0, 0.00, 0.00),
    ("gap heavy",      2.0, 1.0, 2.00, 0.02),
    ("violation heavy", 2.0, 1.0, 0.50, 0.20),
    ("flat feasibility", 1.0, 1.0, 0.50, 0.02),
    ("pre-repair first", 1.0, 2.0, 0.50, 0.02),
    ("cost first",      1.0, 0.5, 2.00, 0.02),
]


def score(metrics, ref_cost, w_feas, w_pre, w_gap, w_viol) -> float:
    r = w_feas * float(metrics["is_feasible"]) + w_pre * float(metrics["pre_feasible"])
    if metrics["is_feasible"] and ref_cost and ref_cost > 0:
        gap = max(0.0, (metrics["total_distance"] - ref_cost) / ref_cost)
        r -= w_gap * min(gap, 2.0)
    r -= w_viol * min(metrics["pre_total_viol"], 50)
    return r


def labels_for(cache, w) -> tuple[list, dict]:
    """Pick the winning candidate per instance-size under weighting w."""
    examples, wins, n = [], {"heur": 0, "gnn": 0}, 0
    for entry in cache:
        best, best_r = None, None
        for cand in entry["candidates"]:
            r = score(cand["metrics"], entry["ref_cost"], *w)
            if best_r is None or r > best_r:
                best, best_r = cand, r
        examples.extend(best["examples"])
        wins[best["tag"]] += 1
        n += 1
    return examples, {"gnn_win_rate": wins["gnn"] / max(1, n), **wins}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="results/label_cache.pt")
    ap.add_argument("--Ns", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--num-reads", type=int, default=200)
    ap.add_argument("--num-sweeps", type=int, default=200)
    ap.add_argument("--out", default="results/reward_sweep.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(ROOT / args.cache, weights_only=False)
    cache = blob["cache"]
    _, test = train_test_split_instances()
    print(f"device={device} | {len(cache)} instance-size pairs | "
          f"{sum(len(c['candidates']) for c in cache)} cached candidates\n", flush=True)

    targs = SimpleNamespace(epochs=args.epochs, lr=3e-3, hidden=64, layers=2,
                            val_frac=0.15, seed=0)

    # Which candidate wins is what the weights control. Settings that pick the same
    # winners produce byte-identical labels, so they need training only once.
    import hashlib
    picks, base_key = {}, None
    for name, *w in SETTINGS:
        idx = []
        for e in cache:
            best, br = None, None
            for i, c in enumerate(e["candidates"]):
                r = score(c["metrics"], e["ref_cost"], *w)
                if br is None or r > br:
                    best, br = i, r
            idx.append(best)
        picks[name] = (hashlib.md5(str(idx).encode()).hexdigest()[:8], idx)
        if name == "published":
            base_key = idx
    print("=== how much does each weighting change the oracle choice? ===")
    for name, (h, idx) in picks.items():
        d = sum(a != b for a, b in zip(base_key, idx))
        print(f"  {name:>18}: {d:3d} of {len(idx)} instance-size pairs differ "
              f"from the published weighting")
    print(f"  distinct label sets: {len({v[0] for v in picks.values()})} "
          f"of {len(SETTINGS)} settings\n", flush=True)

    rows, summary, trained = [], [], {}
    for name, *w in SETTINGS:
        examples, sel = labels_for(cache, w)
        key = picks[name][0]
        print(f"== {name}  weights={w} ==")
        print(f"  oracle picked GNN {100*sel['gnn_win_rate']:.0f}% of the time, "
              f"{len(examples)} labelled graphs", flush=True)
        if key in trained:
            model, stats = trained[key]
            print(f"  identical labels to '{stats['_from']}', reusing that model",
                  flush=True)
        else:
            model, stats = train_one("sage", examples, targs, device)
            stats["_from"] = name
            trained[key] = (model, stats)
        model.eval()
        per_N = {}
        for N in args.Ns:
            got = []
            for inst_name in test:
                for sd in args.seeds:
                    set_global_seed(sd)
                    r = downstream(model, inst_name, N, device, sd,
                                   args.num_reads, args.num_sweeps)
                    rows.append(dict(setting=name, w_feas=w[0], w_pre=w[1],
                                     w_gap=w[2], w_viol=w[3], N=N,
                                     instance=inst_name, seed=sd, **r))
                    got.append(r)
            per_N[N] = (100 * np.mean([g["feasible"] for g in got]),
                        100 * np.mean([g["pre_feasible"] for g in got]))
            print(f"  N={N:3d}  feasible={per_N[N][0]:5.1f}%  "
                  f"pre-feasible={per_N[N][1]:5.1f}%", flush=True)
        summary.append((name, sel["gnn_win_rate"], per_N))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    print("\n=== downstream coarsened feasibility (%) by reward setting ===")
    print(f"{'setting':>18} {'GNN oracle':>11} " + " ".join(f"{'N='+str(N):>8}" for N in args.Ns))
    for name, gwr, per_N in summary:
        print(f"{name:>18} {100*gwr:10.0f}% " +
              " ".join(f"{per_N[N][0]:8.1f}" for N in args.Ns))
    base = next(s for s in summary if s[0] == "published")
    spread = {N: max(s[2][N][0] for s in summary) - min(s[2][N][0] for s in summary)
              for N in args.Ns}
    print(f"\nspread across settings: " +
          ", ".join(f"N={N}: {v:.1f} pp" for N, v in spread.items()))
    print("Read this spread with care. Each setting is trained once, so it mixes the\n"
          "effect of the weights with training-seed noise. Settings whose label sets\n"
          "are near-identical (see the table above) cannot plausibly differ downstream\n"
          "for causal reasons, so a large spread among those is evidence of noise, not\n"
          "of sensitivity. The reliable evidence about the weights is the label-choice\n"
          "table: it is exact and needs no training. Use --train-seeds-style repetition\n"
          "if the downstream column has to carry weight.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
