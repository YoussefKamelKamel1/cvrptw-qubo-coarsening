"""Leave-one-family-out generalisation of the GNN merge scorer.

For each Solomon family F, the scorer is retrained with scripts/train_gnn.py
on the expert-iteration labels (results/gnn_labels_rl.pt) minus
every graph from a family-F instance, then evaluated on the held-out test
instances of family F. A control model is retrained on all labels with the same
script and seed, so the comparison isolates the missing family.

Caveat: the labels include rollouts of the multi-size model, which saw every
family; the held-out family's own graphs are removed entirely.

  python scripts/run_lofo.py train   (GPU, sequential; models in results/lofo/)
  python scripts/run_lofo.py eval    (CPU, parallel)    -> results/lofo.csv
"""
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUTD = ROOT / "results" / "lofo"
FAMILIES = ("C1", "C2", "R1", "R2", "RC1", "RC2")
PY = sys.executable


def family_of(key: str) -> str:
    m = re.match(r"([A-Z]+)(\d)", key)
    return m.group(1) + m.group(2)


def train(fams=("ALL",) + FAMILIES):
    import torch
    OUTD.mkdir(parents=True, exist_ok=True)
    data = torch.load(ROOT / "results" / "gnn_labels_rl.pt", weights_only=False)
    for fam in fams:
        keep = [e for e in data["examples"] if fam == "ALL" or family_of(e["instance"]) != fam]
        lab = OUTD / f"labels_minus_{fam}.pt"
        torch.save(dict(data, examples=keep), lab)
        model = OUTD / f"model_minus_{fam}.pt"
        print(f"[{fam}] {len(keep)} graphs, "
              f"{len({e['instance'] for e in keep})} instance-size keys", flush=True)
        t = time.perf_counter()
        log = OUTD / f"train_minus_{fam}.log"
        with open(log, "w") as fh:
            subprocess.run([PY, str(ROOT / "scripts" / "train_gnn.py"),
                            "--labels", str(lab.relative_to(ROOT)),
                            "--out", str(model.relative_to(ROOT)), "--seed", "0"],
                           cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, check=True)
        print(f"   trained in {time.perf_counter()-t:.0f}s -> {model.name}", flush=True)


def _job(spec):
    from src.runner_parallel import run_one_job
    row = run_one_job({k: v for k, v in spec.items() if k != "model_tag"})
    row["model_tag"] = spec["model_tag"]
    return row


def evaluate():
    import pandas as pd
    from src.datasets import train_test_split_instances
    from src.runner_parallel import build_jobs
    _, test = train_test_split_instances()
    jobs = []
    for name in test:
        fam = family_of(name)
        for tag, path in (("lofo", OUTD / f"model_minus_{fam}.pt"),
                          ("control_all", OUTD / "model_minus_ALL.pt"),
                          ("deployed", ROOT / "results" / "gnn_model_rl.pt")):
            for j in build_jobs([name], [10, 20, 40], [0, 1, 2], ["gnn"],
                                model_path=str(path)):
                jobs.append(dict(j, model_tag=tag))
    nw = min(10, os.cpu_count())
    print(f"{len(jobs)} jobs on {nw} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(nw) as ex:
        for fu in as_completed([ex.submit(_job, j) for j in jobs]):
            rows.append(fu.result())
    out = ROOT / "results" / "lofo.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(tuple(sys.argv[2:]) or ("ALL",) + FAMILIES)   # optional family subset
    else:
        evaluate()
