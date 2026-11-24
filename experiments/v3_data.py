"""Outcomes of every candidate coarsening on the training and validation instances.

  python experiments/v3_data.py --workers 4
"""
import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v3_common import VAL, gh400_names, euro_train_names            # noqa: E402

OUT = HERE / "results" / "v3" / "data"
SIZES = (10, 20, 40)
SEEDS = (0, 1)
POLICY = None


def policy_spec():
    import json
    sel = json.loads((HERE / "results" / "v2" / "r2_selected.json").read_text())
    return "ens:" + ",".join(str(HERE / "results" / v["ckpt"]) for v in sel.values())


def task(t):
    global POLICY
    import torch
    torch.set_num_threads(1)
    from gnn_ensemble import load_spec
    from src.gnn_coarsening import build_features
    from v3_common import load_instance, make_candidates, run_candidate, global_feats, qubo_vars
    dataset, name, N, split = t
    f = OUT / f"{dataset}__{name}__{N}.pt"
    if f.exists():
        return f.name, "skip"
    if POLICY is None:
        POLICY = load_spec(policy_spec(), device="cpu")
    inst = load_instance(dataset, name)
    if len(inst.coords) - 1 < N:
        return f.name, "too small"
    inst = inst.truncate(N)
    recs = []
    for key, co, coarse in make_candidates(inst, POLICY, seed=hash((name, N)) % (1 << 30)):
        nv = qubo_vars(coarse, co.depot_id, inst.capacity)
        outs = [run_candidate(inst, co, coarse, s) for s in SEEDS]
        fb = build_features(coarse, co.depot_id)
        recs.append(dict(key=key, node_feat=fb.node_feat, mp_edge_index=fb.mp_edge_index,
                         glob=torch.tensor(global_feats(inst, coarse, co.depot_id, key, nv),
                                           dtype=torch.float32),
                         nvars=nv, feas=sum(o["final_feasible"] for o in outs) / len(outs),
                         pre=sum(o["pre_feasible"] for o in outs) / len(outs)))
    torch.save(dict(dataset=dataset, name=name, N=N, split=split, family=inst.family,
                    cands=recs), f)
    return f.name, "ok"


def tasks():
    from src.datasets import train_test_split_instances
    train, test = train_test_split_instances()
    out = []
    for n in train:
        out.append(("solomon", n, "val" if n in VAL["solomon"] else "train"))
    gh = gh400_names()
    for fam in ("C1", "C2", "R1", "R2", "RC1", "RC2"):
        names = sorted([n for n in gh if n.split("_")[0] == fam], key=lambda s: int(s.split("_")[-1]))
        out.append(("gh400", names[0], "val"))
        out += [("gh400", n, "train") for n in names[1:6]]
    eu = euro_train_names()
    out += [("euro", n, "val") for n in eu[:6]] + [("euro", n, "train") for n in eu[6:30]]
    assert not {n for d, n, s in out if d == "solomon"} & set(test)
    return [(d, n, N, s) for N in SIZES for d, n, s in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ts = tasks()
    t0 = time.time()
    with get_context("spawn").Pool(a.workers, maxtasksperchild=10) as pool:
        for k, (n, st) in enumerate(pool.imap_unordered(task, ts), 1):
            if k % 20 == 0 or st not in ("ok", "skip"):
                print(f"  [{k}/{len(ts)}] {time.time() - t0:.0f}s {n} {st}", flush=True)
    print(f"done {len(ts)} tasks in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
