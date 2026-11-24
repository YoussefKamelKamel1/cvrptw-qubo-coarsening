"""EURO Meets NeurIPS 2022 VRPTW instances.

The instances give an asymmetric road travel-time matrix next to planar coordinates.
The pipeline derives travel times from coordinates, so we solve a Euclidean
approximation; fit_report() measures how far it is from the road matrix.
Data: github.com/ortec/euro-neurips-vrp-2022-quickstart (instances/, CC BY-NC 4.0),
placed in experiments/data_euro/raw/.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EURO_DIR = HERE / "data_euro" / "raw"
SAMPLE = HERE / "data_euro" / "sample60.txt"


def euro_names() -> list:
    return [ln.strip()[:-4] for ln in SAMPLE.read_text().splitlines() if ln.strip()]


@lru_cache(maxsize=None)
def _parse(name: str) -> dict:
    lines = (EURO_DIR / f"{name}.txt").read_text().splitlines()
    head, sec, cur = {}, {}, None
    for ln in lines:
        s = ln.strip()
        if not s or s == "EOF":
            continue
        if s.endswith("SECTION"):
            cur = s.replace("_SECTION", "")
            sec[cur] = []
        elif cur is None and ":" in s:
            k, v = s.split(":", 1)
            head[k.strip()] = v.strip()
        else:
            sec[cur].append([float(x) for x in s.split()])
    n = int(head["DIMENSION"])
    T = np.array(sec["EDGE_WEIGHT"])
    assert T.shape == (n, n), (name, T.shape)
    xy = np.array(sec["NODE_COORD"])[:, 1:3]
    dem = np.array(sec["DEMAND"])[:, 1]
    tw = np.array(sec["TIME_WINDOW"])[:, 1:3]
    svc = np.array(sec["SERVICE_TIME"])[:, 1]
    depot = int(sec["DEPOT"][0][0])
    assert depot == 1 and len(xy) == len(dem) == len(tw) == len(svc) == n, name
    assert "RELEASE_TIME" not in sec, name
    return dict(n=n, T=T, xy=xy, dem=dem, tw=tw, svc=svc,
                cap=float(head["CAPACITY"]), veh=int(head["VEHICLES"]))


def _euclid(xy):
    d = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


@lru_cache(maxsize=None)
def scale_factor(name: str) -> float:
    p = _parse(name)
    E, M = _euclid(p["xy"]), 0.5 * (p["T"] + p["T"].T)
    off = ~np.eye(p["n"], dtype=bool)
    return float((E[off] * M[off]).sum() / (E[off] ** 2).sum())


def fit_report(name: str, N: int | None = None) -> dict:
    """Agreement of the scaled Euclidean times with the road matrix (first N customers)."""
    p = _parse(name)
    k = p["n"] if N is None else N + 1
    E = _euclid(p["xy"][:k]) * scale_factor(name)
    T = p["T"][:k, :k]
    off = ~np.eye(k, dtype=bool)
    rel = np.abs(E[off] - T[off]) / np.maximum(T[off], 1.0)
    asym = np.abs(T - T.T)[off] / np.maximum(0.5 * (T + T.T)[off], 1.0)
    return dict(instance=name, N=N or p["n"] - 1, scale=scale_factor(name),
                corr=float(np.corrcoef(E[off], T[off])[0, 1]),
                median_rel_err=float(np.median(rel)), p90_rel_err=float(np.quantile(rel, 0.9)),
                median_asym=float(np.median(asym)))


def load_euro(name: str):
    from src.datasets import VRPInstance
    p = _parse(name)
    return VRPInstance(name=name, coords=p["xy"] * scale_factor(name), demand=p["dem"],
                       capacity=p["cap"], n_vehicles=p["veh"], ready_time=p["tw"][:, 0],
                       due_time=p["tw"][:, 1], service_time=p["svc"], family="EURO",
                       meta={"source": "euro-neurips-2022 (euclidean approx.)",
                             "scale": scale_factor(name)})
