"""Hardware, embeddability and alternative-encoding figures -> figures/.

F1 fig_qpu_conditioning : raw feasibility per instance for the three penalty arms,
                          plus mean violations with the QUBO variable count
                          annotated. Static and cap-crush carry the SAME variable
                          count, so the gap between them is conditioning alone.
F2 fig_qpu_mechanism    : dynamic range vs raw violations on hardware.
F3 fig_embeddability    : chain length and physical qubits vs N on Pegasus/Zephyr.
F4 fig_encodings        : pre-repair feasibility of the position vs set-partition
                          encoding.

The adaptive arm is read from the capacity-normalised runs where capacity binds
(RC104, RC108) and from the main sweep elsewhere; the normalisation is a no-op on
non-binding instances, so both sources describe the same calibration.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})

# validated categorical palette (all six checks pass in light mode)
ARM_C = {"static": "#ff7f0e", "capcrush": "#d62728", "adaptive": "#1f77b4"}
ARM_H = {"static": "//", "capcrush": "\\\\", "adaptive": ""}   # print/B&W fallback
ARM_L = {"static": "static", "capcrush": "cap-crush (control)", "adaptive": "adaptive"}
ARMS = ["static", "capcrush", "adaptive"]
COND_C = {"none": "#7f7f7f", "heuristic": "#1f77b4", "gnn": "#d62728"}
COND_M = {"none": "o", "heuristic": "s", "gnn": "^"}
FAM = lambda s: "RC" if s.startswith("RC") else s[0]        # noqa: E731


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {name}")


def load_qpu() -> pd.DataFrame:
    """Main sweep, with the adaptive arm superseded by the capacity-normalised runs."""
    main = pd.concat([pd.read_csv(RES / f"qpu_gnn_seed{s}.csv") for s in (0, 1, 2)],
                     ignore_index=True)
    fix_files = [RES / f"qpu_fix_seed{s}.csv" for s in (0, 1, 2)]
    if all(f.exists() for f in fix_files):
        fix = pd.concat([pd.read_csv(f) for f in fix_files], ignore_index=True)
        superseded = set(zip(fix.instance, fix.penalty_mode))
        keep = ~main.apply(lambda r: (r.instance, r.penalty_mode) in superseded, axis=1)
        main = pd.concat([main[keep], fix], ignore_index=True)
    main["family"] = main.instance.map(FAM)
    return main


# --------------------------------------------------------------------------- #
def f1_conditioning(df):
    order = (df[["instance", "family"]].drop_duplicates()
             .sort_values(["family", "instance"]).instance.tolist())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3),
                                   gridspec_kw={"width_ratios": [2.4, 1]})

    x = np.arange(len(order))
    w = 0.26
    for i, arm in enumerate(ARMS):
        sub = df[df.penalty_mode == arm].groupby("instance")["raw_feas_frac"]
        mean = sub.mean().reindex(order).values * 100
        err = sub.std().reindex(order).fillna(0).values * 100
        ax1.bar(x + (i - 1) * w, mean, w, yerr=err, capsize=2,
                color=ARM_C[arm], hatch=ARM_H[arm], edgecolor="white", linewidth=0.6,
                error_kw={"lw": 0.8}, label=ARM_L[arm])
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=60, ha="right", fontsize=8.5)
    ax1.set_ylabel("pre-repair feasible samples (%)")
    ax1.set_title("Raw feasibility on Advantage2 (3 seeds)")
    ax1.set_ylim(-6, 108)
    ax1.legend(frameon=False, fontsize=9, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.02))
    # family separators
    fams = [FAM(s) for s in order]
    for k in range(1, len(order)):
        if fams[k] != fams[k - 1]:
            ax1.axvline(k - 0.5, color="0.75", lw=0.8, ls=":")
    # the static bars are exactly zero, so say so next to the baseline
    ax1.annotate("static below 0.1%\non every instance", (0.015, 0.74),
                 xycoords="axes fraction", fontsize=9, color=ARM_C["static"],
                 fontweight="bold")

    # right: mean violations per arm, with the variable count annotated
    g = df.groupby("penalty_mode").agg(viol=("mean_raw_viol", "mean"),
                                       nv=("n_vars", "mean")).reindex(ARMS)
    bars = ax2.bar([ARM_L[a] for a in ARMS], g["viol"],
                   color=[ARM_C[a] for a in ARMS],
                   hatch=[ARM_H[a] for a in ARMS], edgecolor="white", linewidth=0.6)
    for rect, nv in zip(bars, g["nv"]):
        ax2.annotate(f"{nv:.0f} vars", (rect.get_x() + rect.get_width() / 2,
                     rect.get_height()), ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("pre-repair violations per sample (mean)")
    ax2.set_title("Same size, different conditioning")
    ax2.set_ylim(0, g["viol"].max() * 1.25)
    ax2.set_xticks(range(len(ARMS)))
    ax2.set_xticklabels([ARM_L[a].replace(" (control)", "\n(control)") for a in ARMS],
                        fontsize=9)
    save(fig, "fig_qpu_conditioning")


def f2_mechanism(df):
    """Ising scale, not the raw coefficient ratio, is the hardware-relevant measure:
    it is the factor the problem is compressed by to fit the device h/J range, so it
    sets how much of the objective falls below coupler precision."""
    mk = {"static": "s", "capcrush": "^", "adaptive": "o"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm in ARMS:
        s = df[df.penalty_mode == arm]
        for ax, y in ((ax1, s.mean_raw_viol), (ax2, s.raw_feas_frac * 100)):
            ax.scatter(s.ising_scale, y, s=36, alpha=0.75, color=ARM_C[arm],
                       marker=mk[arm], edgecolor="white", linewidth=0.5,
                       label=ARM_L[arm] if ax is ax1 else None)
    for ax, ylab in ((ax1, "pre-repair violations per sample"),
                     (ax2, "pre-repair feasible samples (%)")):
        ax.set_xscale("log")
        ax.set_xlabel("Ising rescaling factor (log scale)")
        ax.set_ylabel(ylab)
    ax1.set_title("More compression, more violations")
    ax2.set_title("More compression, fewer feasible samples")
    ax1.legend(frameon=False, fontsize=9)
    save(fig, "fig_qpu_mechanism")


def f3_embeddability():
    f = RES / "embeddability.csv"
    if not f.exists():
        print("  skip fig_embeddability (no embeddability.csv)")
        return
    df = pd.read_csv(f)
    conds = ["none", "heuristic", "gnn"]
    names = {"none": "no coarsening", "heuristic": "heuristic", "gnn": "GNN"}
    Ns = sorted(df.N.unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    x = np.arange(len(Ns))
    w = 0.26

    # panel A: Zephyr chain length; failures drawn as an explicit marker
    for i, cond in enumerate(conds):
        vals, fails = [], []
        for N in Ns:
            s = df[(df.coarsen == cond) & (df.N == N)]
            ok = s[s.zephyr_adv2_ok == True]                       # noqa: E712
            vals.append(ok.zephyr_adv2_max_chain.median() if len(ok) else 0)
            fails.append(len(s) > 0 and len(ok) == 0)
        pos = x + (i - 1) * w
        ax1.bar(pos, vals, w, color=COND_C[cond], hatch=ARM_H["static"] if cond == "none" else "",
                edgecolor="white", linewidth=0.6, label=names[cond])
        for j, (p, v, fl) in enumerate(zip(pos, vals, fails)):
            all_fail = all(df[(df.coarsen == c) & (df.N == Ns[j])].zephyr_adv2_ok.eq(False).all()
                           for c in conds)
            if fl and not all_fail:
                ax1.annotate("no\nembed", (p, 0.4), ha="center", va="bottom",
                             fontsize=7.5, color=COND_C[cond], fontweight="bold")
            elif v:
                ax1.annotate(f"{v:.0f}", (p, v), ha="center", va="bottom", fontsize=8.5)
    # where every condition fails, one label reads better than three
    for j, N in enumerate(Ns):
        if all(df[(df.coarsen == c) & (df.N == N)].zephyr_adv2_ok.eq(False).all() for c in conds):
            ax1.annotate("no embedding found", (x[j], 0.6), ha="center", va="bottom",
                         fontsize=8.5, color="0.35", fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(Ns)
    ax1.set_xlabel("customers N")
    ax1.set_ylabel("max chain length (median)")
    ax1.set_title("Embedding into Advantage2 (Zephyr)")
    ax1.set_ylim(0, 20)
    ax1.legend(frameon=False, fontsize=9)

    # panel B: Zephyr vs Pegasus, only where both embed
    both = []
    for N in Ns:
        for cond in conds:
            s = df[(df.coarsen == cond) & (df.N == N)]
            z = s[s.zephyr_adv2_ok == True]                          # noqa: E712
            p = s[s.pegasus_adv_ok == True]                          # noqa: E712
            if len(z) and len(p):
                both.append((f"{names[cond]}\nN={N}", z.zephyr_adv2_max_chain.median(),
                             p.pegasus_adv_max_chain.median()))
    if both:
        lbl = [b[0] for b in both]
        xb = np.arange(len(both))
        ax2.bar(xb - 0.19, [b[1] for b in both], 0.38, color="#1f77b4",
                edgecolor="white", linewidth=0.6, label="Advantage2 (Zephyr)")
        ax2.bar(xb + 0.19, [b[2] for b in both], 0.38, color="#7f7f7f", hatch="//",
                edgecolor="white", linewidth=0.6, label="Advantage (Pegasus)")
        ax2.set_xticks(xb); ax2.set_xticklabels(lbl, fontsize=8)
        ax2.set_ylabel("max chain length (median)")
        ax2.set_title("Zephyr gives shorter chains")
        ax2.legend(frameon=False, fontsize=9)
    save(fig, "fig_embeddability")


def f4_encodings():
    sp = RES / "set_partition.csv"
    cb = RES / "classical_bar_grid.csv"
    if not sp.exists():
        print("  skip fig_encodings (no set_partition.csv)")
        return
    d_sp = pd.read_csv(sp)
    # full-distribution cover rate, comparable with the position encoding's
    # pre-repair feasibility (both are over all samples, not best-of-N)
    col = "raw_cover_frac" if "raw_cover_frac" in d_sp.columns else "exact_cover"
    d_sp = d_sp[d_sp.coarsen == "gnn"].groupby("N")[col].mean() * 100
    series = [("set-partition", d_sp, "#2ca02c", "^")]
    if cb.exists():
        d_cb = pd.read_csv(cb)
        d_cb = (d_cb[(d_cb.coarsen == "gnn") & (d_cb.sampler == "sa")]
                .groupby("N")["raw_feas_frac"].mean() * 100)
        series.insert(0, ("position (FQS/APS)", d_cb, "#1f77b4", "o"))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, s, c, m in series:
        ax.plot(s.index, s.values, marker=m, ms=7, lw=1.8, color=c, label=name)
        for xi, yi in zip(s.index, s.values):
            ax.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8.5, color=c)
    ax.set_xlabel("customers N")
    ax.set_ylabel("pre-repair feasible samples (%)")
    ax.set_title("Feasibility before repair, by encoding")
    ax.set_ylim(-5, 112)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig_encodings")


def main():
    df = load_qpu()
    print(f"QPU rows: {len(df)}  arms={sorted(df.penalty_mode.unique())}  "
          f"instances={df.instance.nunique()}")
    f1_conditioning(df)
    f2_mechanism(df)
    f3_embeddability()
    f4_encodings()
    print("\nsummary by arm:")
    print(df.groupby("penalty_mode")[["raw_feas_frac", "mean_raw_viol",
                                      "n_vars", "qubo_dynamic_range"]]
            .mean().reindex(ARMS).round(3).to_string())


if __name__ == "__main__":
    main()
