"""Shared utilities: reproducibility, config loading, small helpers.

Seeds are set and logged everywhere: every run seeds numpy, torch, and the
sampler, and is driven by a config file.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = False) -> int:
    """Seed Python, numpy, and (if available) torch. Returns the seed.

    ``deterministic_torch`` trades speed for exact reproducibility on GPU.
    The D-Wave samplers take their own ``seed=`` argument at sample time;
    pass the same seed there (see solvers.py).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config into a plain dict."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class RunContext:
    """Bundles the identity of a single reproducible run for logging."""

    seed: int
    config_path: Optional[str] = None
    extra: Optional[dict] = None

    def as_record(self) -> dict:
        rec = {"seed": self.seed, "config_path": self.config_path}
        if self.extra:
            rec.update(self.extra)
        return rec
