"""fig_ablation_levers, including the cap-crush control -> figures/.

Two panels: (left) pre-repair violations per penalty mode with pre-repair
feasibility annotated; (right) QUBO variable count per mode. The key visual is
that cap-crush sits at the STATIC variable count yet collapses violations, so the
gain is dynamic-range conditioning, not the smaller search space of smart-cap.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})

MODES = ["static", "scale_only", "smartcap_only", "adaptive", "capcrush"]
LABELS = ["static", "scale\nonly", "smart-cap\nonly", "adaptive", "cap-crush\n(control)"]
# grey = keeps static var count; blue = drops capacity vars; red = the control
COLORS = ["#7f7f7f", "#7f7f7f", "#1f77b4", "#1f77b4", "#d62728"]


def main():
    df = pd.read_csv(ROOT / "results" / "phase2_confound.csv")
    agg = df.groupby("penalty_mode").agg(
        viol=("pre_total_viol", "mean"),
        feas=("pre_feasible", lambda s: 100 * s.mean()),
        vars=("num_qubo_vars", "mean")).reindex(MODES)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    b1 = ax1.bar(LABELS, agg["viol"], color=COLORS)
    ax1.set_ylabel("pre-repair violations (mean)")
    ax1.set_title("Raw constraint violations")
    # label the value as well: the adaptive and cap-crush bars are near zero on a
    # 0-33 axis and would otherwise read as missing data
    for rect, v, f in zip(b1, agg["viol"], agg["feas"]):
        ax1.annotate(f"{v:.2f}\n{f:.0f}% feas.",
                     (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                     ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylim(0, max(agg["viol"]) * 1.30)

    b2 = ax2.bar(LABELS, agg["vars"], color=COLORS)
    ax2.set_ylabel("QUBO variables (mean)")
    ax2.set_title("Problem size")
    ax2.axhline(agg.loc["static", "vars"], ls="--", lw=1, color="#7f7f7f")
    for rect, v in zip(b2, agg["vars"]):
        ax2.annotate(f"{v:.0f}", (rect.get_x() + rect.get_width() / 2,
                     rect.get_height()), ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, max(agg["vars"]) * 1.15)

    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation_levers.pdf")
    fig.savefig(OUT / "fig_ablation_levers.png", dpi=140)
    plt.close(fig)
    print("wrote fig_ablation_levers (with cap-crush control)")
    print(agg.round(2).to_string())


if __name__ == "__main__":
    main()
