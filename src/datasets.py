"""Dataset loaders for VRP benchmarks.

Currently supports:
  * Solomon CVRPTW instances (C1/C2, R1/R2, RC1/RC2)  -> load_solomon()
  * Christofides CMT CVRP instances (CMT01-05, 11, 12) -> load_christofides()

Solomon data format
-------------------
We use the predecessor project's data layout,
which stores each instance as a **CSV** under ``data/solomon/<FAMILY>/`` with
columns::

    CUST NO.,XCOORD.,YCOORD.,DEMAND,READY TIME,DUE DATE,SERVICE TIME

Two format quirks we must preserve for reproduction compatibility:
  1. The CSV does **not** contain the vehicle capacity. Capacity is resolved
     from the Solomon family (C1=200, C2=700, R1=200, R2=1000, RC1=200,
     RC2=1000) via ``SOLOMON_FAMILY_CAPACITY`` -- exactly as the predecessor's
     ``_solomon_capacity`` does.
  2. The **first data row is the depot** (its DEMAND is 0). It is often
     labelled "1" rather than "0"; we always place it at index 0 internally.

A parser for the classic whitespace ``.txt`` Solomon format is also provided
(``parse_solomon_txt``) as a fallback for externally sourced instances.

Quantum tier
------------
Following the papers (arXiv:2510.22329, IEEE Access 2025), the quantum tier
takes the *first N customers* (N=5, 10), preserving family structure, rather
than generating random graphs. Use ``VRPInstance.truncate(n)``.

Distances are Euclidean by default (Solomon convention); callers may round.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Repo layout: this file is src/datasets.py, data lives at ../data
_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# Solomon vehicle capacity by family (the CSVs omit it). Mirrors the
# predecessor's _solomon_capacity(); values are the canonical Solomon settings.
SOLOMON_FAMILY_CAPACITY: dict[str, float] = {
    "C1": 200.0, "C2": 700.0,
    "R1": 200.0, "R2": 1000.0,
    "RC1": 200.0, "RC2": 1000.0,
}

# Quantum-tier (FQS/APS) coarsening hyperparameters, tuned per family at N=10
# in the predecessor project (graph-coarsening/utils.py FAMILY_HYPERPARAMS).
# Kept here so Phase-1 reproduction can look them up by instance; the GNN's
# whole point (Phase 3) is to make these per-family knobs unnecessary.
SOLOMON_FAMILY_HYPERPARAMS: dict[str, dict] = {
    "C1":  {"alpha": 1.0, "beta": 1.0, "P": 0.5, "radiusCoeff": 2.0},
    "C2":  {"alpha": 1.0, "beta": 1.0, "P": 0.5, "radiusCoeff": 2.0},
    "R1":  {"alpha": 1.0, "beta": 0.6, "P": 0.4, "radiusCoeff": 0.5},
    "R2":  {"alpha": 1.0, "beta": 0.6, "P": 0.4, "radiusCoeff": 0.5},
    "RC1": {"alpha": 0.7, "beta": 0.8, "P": 0.7, "radiusCoeff": 2.0},
    "RC2": {"alpha": 0.7, "beta": 0.8, "P": 0.7, "radiusCoeff": 2.0},
}


@dataclass
class VRPInstance:
    """A capacitated VRP (optionally with time windows).

    Node 0 is always the depot. Arrays are indexed 0..n_customers, i.e.
    they include the depot at index 0.
    """

    name: str
    coords: np.ndarray          # (N+1, 2) float, node 0 = depot
    demand: np.ndarray          # (N+1,)   float, demand[0] == 0
    capacity: float
    n_vehicles: int
    # Time-window fields (None for pure CVRP instances like Christofides)
    ready_time: Optional[np.ndarray] = None    # e_i, (N+1,)
    due_time: Optional[np.ndarray] = None       # l_i, (N+1,)
    service_time: Optional[np.ndarray] = None   # s_i, (N+1,)
    family: Optional[str] = None                # fine family: 'C1'|'C2'|'R1'|'R2'|'RC1'|'RC2'
    meta: dict = field(default_factory=dict)

    @property
    def n_customers(self) -> int:
        return self.coords.shape[0] - 1

    @property
    def has_time_windows(self) -> bool:
        return self.ready_time is not None

    @property
    def family_group(self) -> Optional[str]:
        """Coarse family used for reporting: 'C' | 'R' | 'RC' | None."""
        if self.family is None:
            return None
        f = self.family.upper()
        if f.startswith("RC"):
            return "RC"
        if f.startswith("R"):
            return "R"
        if f.startswith("C"):
            return "C"
        return None

    def distance_matrix(self, rounding: Optional[str] = None) -> np.ndarray:
        """Euclidean distance matrix over all nodes (depot included).

        rounding: None (exact float), 'floor', 'round', or 'trunc1'
        (one-decimal truncation, a common Solomon convention).
        """
        d = self.coords
        diff = d[:, None, :] - d[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        if rounding == "floor":
            dist = np.floor(dist)
        elif rounding == "round":
            dist = np.round(dist)
        elif rounding == "trunc1":
            dist = np.trunc(dist * 10) / 10.0
        return dist

    def truncate(self, n: int) -> "VRPInstance":
        """Return a copy keeping the depot + first ``n`` customers.

        Mirrors the paper's 'first N customers' quantum-tier subsampling.
        """
        if n > self.n_customers:
            raise ValueError(f"{self.name}: only {self.n_customers} customers, asked {n}")
        sl = slice(0, n + 1)  # keep depot (0) + first n customers

        def _cut(a):
            return None if a is None else a[sl].copy()

        return VRPInstance(
            name=f"{self.name}[:{n}]",
            coords=self.coords[sl].copy(),
            demand=self.demand[sl].copy(),
            capacity=self.capacity,
            n_vehicles=self.n_vehicles,
            ready_time=_cut(self.ready_time),
            due_time=_cut(self.due_time),
            service_time=_cut(self.service_time),
            family=self.family,
            meta={**self.meta, "truncated_to": n},
        )


# --------------------------------------------------------------------------- #
# Solomon CVRPTW
# --------------------------------------------------------------------------- #
_SOLOMON_HEADERS = ["CUST NO.", "XCOORD.", "YCOORD.", "DEMAND",
                    "READY TIME", "DUE DATE", "SERVICE TIME"]


def family_from_name(name: str) -> Optional[str]:
    """Fine Solomon family ('RC1','C2',...) from an instance name like 'RC205'.

    Check RC before R/C to avoid the prefix clash.
    """
    stem = Path(name).stem.upper()
    for fam in ("RC1", "RC2", "C1", "C2", "R1", "R2"):
        # e.g. 'RC2' matches 'RC205'; require the family digit then more digits
        prefix = fam
        if stem.startswith(prefix) and stem[len(prefix):len(prefix) + 1].isdigit():
            return fam
    return None


def solomon_capacity(name: str) -> Optional[float]:
    """Vehicle capacity for a Solomon instance by name (family map)."""
    fam = family_from_name(name)
    return SOLOMON_FAMILY_CAPACITY.get(fam) if fam else None


def parse_solomon_csv(text: str, name: str) -> VRPInstance:
    """Parse the predecessor's Solomon CSV format.

    Header row contains 'CUST NO.'; data rows follow. The first data row is
    the depot (demand 0). Capacity is resolved from the family map, not the
    file. ``name`` (e.g. 'C101') determines family + capacity.
    """
    cap = solomon_capacity(name)
    if cap is None:
        raise ValueError(
            f"parse_solomon_csv: unknown Solomon instance '{name}' -- "
            "cannot determine family/capacity."
        )
    lines = text.splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if "CUST NO." in ln), None)
    if header_idx is None:
        raise ValueError(f"parse_solomon_csv({name}): no 'CUST NO.' header found")

    reader = csv.DictReader(
        io.StringIO("\n".join(lines[header_idx + 1:])),
        fieldnames=_SOLOMON_HEADERS, delimiter=",", skipinitialspace=True,
    )
    rows = []
    for row in reader:
        cleaned = {k.strip(): v.strip() for k, v in row.items()
                   if k and v and str(k).strip() and str(v).strip()}
        if not cleaned:
            continue
        try:
            rows.append([
                float(cleaned[_SOLOMON_HEADERS[1]]),  # x
                float(cleaned[_SOLOMON_HEADERS[2]]),  # y
                float(cleaned[_SOLOMON_HEADERS[3]]),  # demand
                float(cleaned[_SOLOMON_HEADERS[4]]),  # ready
                float(cleaned[_SOLOMON_HEADERS[5]]),  # due
                float(cleaned[_SOLOMON_HEADERS[6]]),  # service
            ])
        except (KeyError, ValueError):
            continue
    if not rows:
        raise ValueError(f"parse_solomon_csv({name}): no data rows parsed")

    arr = np.array(rows, dtype=float)  # first row is depot, order preserved
    return VRPInstance(
        name=Path(name).stem,
        coords=arr[:, 0:2],
        demand=arr[:, 2],
        capacity=cap,
        n_vehicles=0,  # Solomon fixes a fleet size per family; solver decides here
        ready_time=arr[:, 3],
        due_time=arr[:, 4],
        service_time=arr[:, 5],
        family=family_from_name(name),
        meta={"source": "solomon_csv"},
    )


def parse_solomon_txt(text: str, name: Optional[str] = None) -> VRPInstance:
    """Parse the classic whitespace Solomon .txt format (fallback source).

    Layout: a NAME line, a ``NUMBER CAPACITY`` block, then 7-column customer
    rows (CUST NO. 0 = depot). Capacity/fleet are read from the file here.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    parsed_name = lines[idx].strip() if idx < len(lines) else (name or "solomon")
    if name is None:
        name = parsed_name

    tokens_by_line = [ln.split() for ln in lines]
    n_vehicles = capacity = None
    for i, toks in enumerate(tokens_by_line):
        if len(toks) >= 2 and toks[0].upper() == "NUMBER" and toks[1].upper() == "CAPACITY":
            for j in range(i + 1, len(tokens_by_line)):
                t = tokens_by_line[j]
                if len(t) >= 2 and _all_numeric(t[:2]):
                    n_vehicles = int(float(t[0]))
                    capacity = float(t[1])
                    break
            break

    rows = [[float(x) for x in toks] for toks in tokens_by_line
            if len(toks) == 7 and _all_numeric(toks)]
    if not rows:
        raise ValueError(f"parse_solomon_txt({name}): no 7-column customer rows found")
    arr = np.array(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]  # sort by CUST NO. so depot (0) first
    if capacity is None:
        capacity = solomon_capacity(name)
    if capacity is None:
        raise ValueError(f"parse_solomon_txt({name}): capacity unknown")
    return VRPInstance(
        name=Path(name).stem, coords=arr[:, 1:3], demand=arr[:, 3],
        capacity=capacity, n_vehicles=int(n_vehicles or 0),
        ready_time=arr[:, 4], due_time=arr[:, 5], service_time=arr[:, 6],
        family=family_from_name(name), meta={"source": "solomon_txt"},
    )


def _find_solomon_file(name: str, root: Path) -> Optional[Path]:
    """Locate an instance file by name, searching family subfolders."""
    stem = Path(name).stem
    for ext in (".csv", ".txt"):
        # direct, then family-subfolder, then recursive case-insensitive
        cand = root / f"{stem}{ext}"
        if cand.exists():
            return cand
    matches = [p for p in root.rglob("*")
               if p.is_file() and p.stem.lower() == stem.lower()
               and p.suffix.lower() in (".csv", ".txt")]
    return matches[0] if matches else None


def load_solomon(name: str, data_dir: Optional[Path] = None) -> VRPInstance:
    """Load a cached Solomon instance by name, e.g. 'C101', 'R101', 'RC101'."""
    root = Path(data_dir) if data_dir else _DATA_ROOT / "solomon"
    path = _find_solomon_file(name, root)
    if path is None:
        raise FileNotFoundError(f"Solomon instance not found under {root}: {name}")
    text = path.read_text()
    if path.suffix.lower() == ".csv":
        return parse_solomon_csv(text, name=path.stem)
    return parse_solomon_txt(text, name=path.stem)


def list_solomon(data_dir: Optional[Path] = None) -> list[str]:
    """Return sorted instance names available under data/solomon."""
    root = Path(data_dir) if data_dir else _DATA_ROOT / "solomon"
    if not root.exists():
        return []
    names = {p.stem for p in root.rglob("*") if p.suffix.lower() in (".csv", ".txt")}
    return sorted(names)


def train_test_split_instances(test_every: int = 4,
                               data_dir: Optional[Path] = None
                               ) -> tuple[list[str], list[str]]:
    """Deterministic family-stratified split of Solomon instances.

    Within each fine family (C1/C2/R1/R2/RC1/RC2), every ``test_every``-th
    instance goes to test. Guarantees all families appear in both splits and
    that the GNN is evaluated on instances it never trained on.
    """
    from collections import defaultdict
    by_family: dict = defaultdict(list)
    for name in list_solomon(data_dir):
        by_family[family_from_name(name) or "?"].append(name)
    train, test = [], []
    for fam in sorted(by_family):
        for i, name in enumerate(sorted(by_family[fam])):
            (test if i % test_every == test_every - 1 else train).append(name)
    return sorted(train), sorted(test)


# --------------------------------------------------------------------------- #
# Christofides CMT (CVRP, no time windows)
# --------------------------------------------------------------------------- #
def parse_christofides(text: str, name: Optional[str] = None) -> VRPInstance:
    """Parse the classic Christofides/Mingozzi/Toth CMT text format.

    Header line: ``N  Q  MAX_ROUTE_TIME  DROP(SERVICE)_TIME``
    Then one depot line and N customer lines: ``X  Y  DEMAND``.

    Some distributions place demand differently; this parser reads the
    common ``x y demand`` layout used by CVRPLIB text files. It is
    validated against the actual downloaded files before use.
    """
    toks = text.split()
    nums = [t for t in toks if _is_number(t)]
    if len(nums) < 4:
        raise ValueError(f"parse_christofides({name}): too few numeric tokens")
    n = int(float(nums[0]))
    capacity = float(nums[1])
    # nums[2] = max route time, nums[3] = drop/service time (often 0)
    body = nums[4:]
    # depot: x y  (+ possibly demand 0) ; customers: x y demand
    # Robustly read triples for customers; the depot may have 2 or 3 fields.
    # Common CMT layout: depot 'x y', then n rows of 'x y demand'.
    depot = body[:2]
    rest = body[2:]
    if len(rest) < 3 * n:
        raise ValueError(
            f"parse_christofides({name}): expected {3*n} customer values, got {len(rest)}"
        )
    rest = rest[: 3 * n]
    cust = np.array(rest, dtype=float).reshape(n, 3)
    coords = np.vstack([[float(depot[0]), float(depot[1])], cust[:, :2]])
    demand = np.concatenate([[0.0], cust[:, 2]])
    return VRPInstance(
        name=name or "cmt",
        coords=coords,
        demand=demand,
        capacity=capacity,
        n_vehicles=0,  # unspecified in CMT; solver decides
        family=None,
        meta={"source": "christofides", "max_route_time": float(nums[2]),
              "drop_time": float(nums[3])},
    )


def load_christofides(name: str, data_dir: Optional[Path] = None) -> VRPInstance:
    root = Path(data_dir) if data_dir else _DATA_ROOT / "christofides"
    stem = name if "." in name else f"{name}.txt"
    path = root / stem
    if not path.exists():
        matches = [p for p in root.glob("*") if p.stem.lower() == Path(stem).stem.lower()]
        if not matches:
            raise FileNotFoundError(f"Christofides instance not found: {path}")
        path = matches[0]
    return parse_christofides(path.read_text(), name=path.stem)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _all_numeric(toks) -> bool:
    return all(_is_number(t) for t in toks)


if __name__ == "__main__":
    # Smoke test against whatever is cached.
    names = list_solomon()
    print(f"{len(names)} Solomon instances cached under {_DATA_ROOT / 'solomon'}")
    sample = ["C101", "R101", "RC101", "C201", "R201", "RC201"]
    for name in sample:
        if name not in names:
            continue
        inst = load_solomon(name)
        assert inst.demand[0] == 0.0, f"{name}: depot demand must be 0"
        assert inst.n_customers == 100, f"{name}: expected 100 customers, got {inst.n_customers}"
        d = inst.distance_matrix()
        t5 = inst.truncate(5)
        print(f"  {inst.name}: N={inst.n_customers}, Q={inst.capacity}, "
              f"family={inst.family}/{inst.family_group}, TW={inst.has_time_windows}, "
              f"depot=({inst.coords[0,0]:.0f},{inst.coords[0,1]:.0f}), "
              f"win0=[{inst.ready_time[0]:.0f},{inst.due_time[0]:.0f}]  "
              f"| truncate(5): N={t5.n_customers}, d[0,1]={d[0,1]:.2f}")
    print("datasets.py smoke test OK")
