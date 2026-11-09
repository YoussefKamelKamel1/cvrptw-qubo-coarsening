"""fig_learned_stop -> figures/.

Two panels vs N: coarsened feasibility and gap to the OR-Tools reference, for a
fixed coarsening depth (P=0.5) versus a learned stopping threshold. Learned
stopping lowers the gap but also lowers feasibility -- it shifts along the
feasibility/cost frontier rather than dominating fixed depth.
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


def main():
    d = pd.read_csv(ROOT / "results" / "learned_stop.csv")
    Ns = sorted(d.N.unique())
    styles = {"gnn_P0.5": ("fixed depth ($P{=}0.5$)", "#1f77b4", "o-"),
              "gnn_tau0.95": ("learned stop ($\\tau{=}0.95$)", "#d62728", "s--")}

    feas = (d.groupby(["N", "variant"]).is_feasible.mean().mul(100).unstack("variant"))
    fe = d[d.is_feasible]
    gap = (fe.groupby(["N", "variant"]).opt_gap.mean().mul(100).unstack("variant"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4))
    for v, (lab, col, ls) in styles.items():
        if v in feas:
            a1.plot(feas.index, feas[v], ls, color=col, label=lab)
        if v in gap:
            a2.plot(gap.index, gap[v], ls, color=col, label=lab)
    a1.set_xlabel("customers $N$"); a1.set_ylabel("coarsened feasibility (%)")
    a1.set_title("Feasibility"); a1.set_xticks(Ns); a1.legend(fontsize=9)
    a2.set_xlabel("customers $N$"); a2.set_ylabel("gap to OR-Tools reference (%)")
    a2.set_title("Cost"); a2.set_xticks(Ns); a2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_learned_stop.pdf")
    fig.savefig(OUT / "fig_learned_stop.png", dpi=140)
    plt.close(fig)
    print("wrote fig_learned_stop (gap to OR-Tools label)")
    print("feasibility %:\n", feas.round(0).to_string())
    print("gap %:\n", gap.round(0).to_string())


if __name__ == "__main__":
    main()
