"""OR-Tools reference cache.

Solve each (instance, N) with OR-Tools once, persist to CSV, and reuse the
near-optimal cost everywhere (label rewards, optimality gaps). Removes repeated
2-3 s OR-Tools solves from every downstream run.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "results" / "ortools_cache.csv"


def ortools_cost(name: str, N: int, time_limit: float = 3.0) -> Optional[float]:
    from .datasets import load_solomon
    from .baselines import solve_cvrptw_ortools
    inst = load_solomon(name).truncate(N)
    s = solve_cvrptw_ortools(inst, time_limit_s=time_limit)
    return float(s.cost) if s.feasible else None


def _worker(args):
    name, N, tl = args
    return (name, N, ortools_cost(name, N, tl))


def load_cache(csv_path=DEFAULT_CACHE) -> dict:
    d: dict = {}
    p = Path(csv_path)
    if p.exists():
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                c = row["cost"]
                d[(row["instance"], int(row["N"]))] = (
                    None if c in ("", "None") else float(c))
    return d


def build_cache(pairs, csv_path=DEFAULT_CACHE, time_limit: float = 3.0,
                n_workers: Optional[int] = None, serial: bool = False) -> dict:
    """Ensure OR-Tools cost is cached for every (name, N) in ``pairs``."""
    csv_path = Path(csv_path)
    cache = load_cache(csv_path)
    todo = [(n, N, time_limit) for (n, N) in dict.fromkeys(pairs)
            if (n, N) not in cache]
    if todo:
        if serial or n_workers == 1:
            results = [_worker(a) for a in todo]
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_worker, todo))
        for name, N, cost in results:
            cache[(name, N)] = cost
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["instance", "N", "cost"])
            for (name, N), cost in sorted(cache.items()):
                w.writerow([name, N, "" if cost is None else cost])
    return cache
