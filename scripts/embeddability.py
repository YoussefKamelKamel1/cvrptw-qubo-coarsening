"""Offline minor-embedding study: does coarsening make the QUBO hardware-tractable?

For each coarsening (none / heuristic / gnn) the adaptive-penalty QUBO is embedded
into the ideal Pegasus (Advantage) and Zephyr (Advantage2) graphs with minorminer.
No Leap account is used. Reports logical-variable count, embedding success rate,
and chain length / physical-qubit count. Chains longer than a device can hold
reliably (roughly >7) are what make an otherwise-fine QUBO fail on real hardware,
so this is the quantity coarsening has to bring down.

Targets are FULL-YIELD ideal graphs; real devices have a few percent of dead
qubits, so these figures are an optimistic bound (embedding is only harder in
practice). Run:

  python scripts/embeddability.py --N 10 20 40 --n-instances 3
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dwave_networkx as dnx           # noqa: E402
import torch                           # noqa: E402
from minorminer import find_embedding  # noqa: E402

from run_qpu import build_problem, source_graph          # noqa: E402
from src.gnn_coarsening import load_model                # noqa: E402
from src.datasets import train_test_split_instances      # noqa: E402


def best_embedding(S, T, seeds, timeout):
    """Best (shortest max-chain) embedding of source graph S into target T."""
    best = None
    for s in seeds:
        emb = find_embedding(S, T, random_seed=s, timeout=timeout, tries=1)
        if not emb:
            continue
        chains = [len(c) for c in emb.values()]
        mc = max(chains)
        if best is None or mc < best["max_chain"]:
            best = dict(max_chain=mc, phys_qubits=sum(chains),
                        mean_chain=round(statistics.mean(chains), 2))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--coarsen", nargs="+", default=["none", "heuristic", "gnn"])
    ap.add_argument("--instances", nargs="+", default=None)
    ap.add_argument("--n-instances", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--timeout", type=int, default=90, help="seconds per embedding try")
    ap.add_argument("--max-vars", type=int, default=1500,
                    help="skip embedding above this logical size (marks too_large)")
    ap.add_argument("--model", default="results/gnn_model_rl.pt")
    ap.add_argument("--out", default="results/embeddability.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ROOT / args.model, device=device)
    _, test = train_test_split_instances()
    names = args.instances or test[:args.n_instances]

    print("building target topologies ...", flush=True)
    targets = {"pegasus_adv": dnx.pegasus_graph(16),
               "zephyr_adv2": dnx.zephyr_graph(15)}
    for k, g in targets.items():
        print(f"  {k}: {g.number_of_nodes()} qubits, {g.number_of_edges()} couplers")
    print(f"instances: {names} | N={args.N} | coarsen={args.coarsen}\n", flush=True)

    per_inst = []          # raw rows
    t0 = time.perf_counter()
    for N in args.N:
        for coarsen in args.coarsen:
            for name in names:
                *_, qubo, _ = build_problem(name, N, coarsen, "adaptive", model, device)
                S = source_graph(qubo)
                n_vars = S.number_of_nodes()
                row = dict(N=N, coarsen=coarsen, instance=name, n_vars=n_vars)
                for topo, T in targets.items():
                    if n_vars > args.max_vars:
                        emb = None
                        tag = "too_large"
                    else:
                        emb = best_embedding(S, T, args.seeds, args.timeout)
                        tag = "ok" if emb else "FAIL"
                    row[f"{topo}_ok"] = emb is not None
                    row[f"{topo}_max_chain"] = emb["max_chain"] if emb else None
                    row[f"{topo}_phys"] = emb["phys_qubits"] if emb else None
                    print(f"  N={N:2d} {coarsen:9s} {name:5s} vars={n_vars:4d} "
                          f"{topo}: {tag}"
                          + (f" max_chain={emb['max_chain']} phys={emb['phys_qubits']}"
                             if emb else ""), flush=True)
                per_inst.append(row)

    # per-instance CSV
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_inst[0].keys()))
        w.writeheader()
        w.writerows(per_inst)

    # summary table
    print("\n" + "=" * 78)
    print(f"{'N':>3} {'coarsen':10} {'vars':>5}  "
          f"{'peg_ok':>6} {'peg_chain':>9} {'peg_qb':>7}  "
          f"{'zep_ok':>6} {'zep_chain':>9} {'zep_qb':>7}")
    print("-" * 78)

    def med(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.median(vals), 1) if vals else None

    for N in args.N:
        for coarsen in args.coarsen:
            grp = [r for r in per_inst if r["N"] == N and r["coarsen"] == coarsen]
            if not grp:
                continue
            v = round(statistics.mean(r["n_vars"] for r in grp))
            po = sum(r["pegasus_adv_ok"] for r in grp)
            zo = sum(r["zephyr_adv2_ok"] for r in grp)
            n = len(grp)
            print(f"{N:>3} {coarsen:10} {v:>5}  "
                  f"{po:>3}/{n:<2} {str(med(r['pegasus_adv_max_chain'] for r in grp)):>9} "
                  f"{str(med(r['pegasus_adv_phys'] for r in grp)):>7}  "
                  f"{zo:>3}/{n:<2} {str(med(r['zephyr_adv2_max_chain'] for r in grp)):>9} "
                  f"{str(med(r['zephyr_adv2_phys'] for r in grp)):>7}")
    print("=" * 78)
    print(f"\ndone in {time.perf_counter()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
