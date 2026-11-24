"""Large-neighbourhood search with route selection: improvement per iteration."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 else "pilot"
it = pd.read_csv(HERE / "results" / f"exp7b_{tag}_iterations.csv")
sm = pd.read_csv(HERE / "results" / f"exp7b_{tag}_summary.csv")

it["sel_gain"] = 100 * (it.best_single - it.exact) / it.best_single     # selection vs best single
it["improves"] = it.exact < it.current - 1e-9
it["single_beats_cur"] = it.best_single < it.current - 1e-9
it["sa_hits"] = np.isclose(it.sa, it.exact, rtol=1e-6)
it["sa_vs_single"] = it.sa <= it.best_single + 1e-9

print("=== per iteration, identical pool for all selectors ===")
print(f"iterations: {len(it)}   mean pool size: {it.pool.mean():.0f}")
print(f"exact selection beats the best single OR-Tools solution: "
      f"{(it.exact < it.best_single - 1e-9).mean()*100:.0f}% of iterations, "
      f"mean gain {it.sel_gain.mean():.1f}%")
print(f"best single OR-Tools re-solve beats the current routes: "
      f"{it.single_beats_cur.mean()*100:.0f}%")
print(f"exact selection improves the current routes: {it.improves.mean()*100:.0f}%")
print(f"SA reaches the exact selection: {it.sa_hits.mean()*100:.0f}%   "
      f"SA at least as good as best single: {it.sa_vs_single.mean()*100:.0f}%")

print("\n=== final solutions ===")
sm["lns_%"] = 100 * (sm.lns_exact - sm.start) / sm.start
sm["ortlong_%"] = 100 * (sm.ort_long - sm.start) / sm.start
print(sm[["instance", "N", "start", "lns_exact", "lns_%", "ort_long", "ortlong_%",
          "lns_wall", "sa_wall"]].round(2).to_string(index=False))
