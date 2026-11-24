"""Test of the outcome-based selector on Solomon, Gehring-Homberger and EURO instances.

  python experiments/v3_test.py --dataset solomon|gh|euro [--capped] --workers 4
"""
import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

R = HERE / "results" / "v3"
SIZES = (10, 20, 40, 50, 60, 80, 100)
_M = None
CAPPED = False


def task(t):
    global _M
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from gnn_ensemble import load_spec
    from src.gnn_coarsening import build_features
    from v3_common import load_instance, make_candidates, run_candidate, global_feats, qubo_vars, FIXED_KEY
    from v3_data import policy_spec
    from v3_train import ValueNet, predict
    dataset, name, N, capped = t
    global CAPPED
    CAPPED = capped
    f = R / ("test_capped" if capped else "test") / f"{dataset}__{name}__{N}.json"
    if f.exists():
        return "skip"
    if _M is None:
        vals = []
        for s in range(5):
            m = ValueNet(); m.load_state_dict(torch.load(R / f"value_seed{s}.pt")); vals.append(m.eval())
        _M = (load_spec(policy_spec(), device="cpu"), vals,
              json.loads((R / "value_calibration.json").read_text())["lam"])
    policy, vals, lam = _M
    inst = load_instance(dataset, name)
    if len(inst.coords) - 1 < N:
        return "too small"
    inst = inst.truncate(N)
    cands = make_candidates(inst, policy, seed=hash((name, N)) % (1 << 30))
    feats, nvs = [], []
    for key, co, coarse in cands:
        nv = qubo_vars(coarse, co.depot_id, inst.capacity)
        fb = build_features(coarse, co.depot_id)
        feats.append(dict(node_feat=fb.node_feat, mp_edge_index=fb.mp_edge_index, feas=0.0,
                          glob=torch.tensor(global_feats(inst, coarse, co.depot_id, key, nv),
                                            dtype=torch.float32)))
        nvs.append(nv)
    p = predict(vals, feats)
    fx = [i for i, c in enumerate(cands) if c[0] == FIXED_KEY][0]
    if CAPPED:   # hard size cap: only candidates no larger than heuristic-fixed
        k = int(np.argmax(np.where(np.array(nvs) <= nvs[fx], p, -np.inf)))
    else:
        k = int(np.argmax(p - lam * np.log(nvs)))
    key, co, coarse = cands[k]
    runs = [dict(seed=s, **run_candidate(inst, co, coarse, s)) for s in (0, 1, 2)]
    f.write_text(json.dumps(dict(dataset=dataset, name=name, N=N, family=inst.family, chosen=key,
                                 chosen_vars=nvs[k], fixed_vars=nvs[fx],
                                 pred={c[0]: float(x) for c, x in zip(cands, p)},
                                 vars={c[0]: int(v) for c, v in zip(cands, nvs)}, runs=runs)))
    return "ok"


def names(dataset):
    if dataset == "solomon":
        from src.datasets import train_test_split_instances
        return train_test_split_instances()[1]
    if dataset == "gh":
        from review_runner import gh_names
        return gh_names()
    from euro_data import euro_names
    return euro_names()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["solomon", "gh", "euro"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--capped", action="store_true",
                    help="choose only among candidates no larger than heuristic-fixed")
    a = ap.parse_args()
    (R / ("test_capped" if a.capped else "test")).mkdir(parents=True, exist_ok=True)
    ts = [(a.dataset, n, N, a.capped) for N in sorted(SIZES, reverse=True) for n in names(a.dataset)]
    t0 = time.time()
    with get_context("spawn").Pool(a.workers, maxtasksperchild=10) as pool:
        for k, st in enumerate(pool.imap_unordered(task, ts), 1):
            if k % 25 == 0:
                print(f"  [{k}/{len(ts)}] {time.time() - t0:.0f}s", flush=True)
    print(f"done {a.dataset}: {len(ts)} tasks in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
