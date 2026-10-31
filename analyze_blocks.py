"""Summarise the block-count sweep: accuracy against number of blocks, with equivalence tests.

Reads every 160-epoch run in results/, groups the staggered runs by their schedule, and compares
each group with the synchronous AC/DC baseline using the same Welch t-test and two-one-sided-tests
procedure the report uses, taken from the notebook so the statistics cannot drift.

No training. Safe to run at any time: groups with fewer than two seeds are reported as incomplete
rather than tested, so it is useful while the sweep is still filling in.

Usage:  python analyze_blocks.py
Writes: results/blocks_accuracy.csv  and  figures/eps/07c_blocks_accuracy.eps
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import make_thesis_figures as M  # noqa: E402  (house figure style, notebook loader)
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

EPOCHS = 160            # the valid sweep; shorter presets are ignored
MARGIN = 1.0            # equivalence margin in percentage points, fixed in the proposal
LEADING_B = 4           # leading candidate of the exploratory sweep, ringed in the figure


def load_welch_and_tost():
    """Take the report's own test out of the notebook, so the statistics cannot drift.

    make_thesis_figures only executes the cells it needs for figures, and the statistics cell
    is not one of them, so it is extracted here. The cell continues into code that expects a
    populated RESULTS dict, so it is cut at the first thing after the function.
    """
    import math  # noqa: F401  (used by the extracted code)

    import scipy.stats as stats  # noqa: F401

    cells = ["".join(c["source"]) for c in
             json.load(open(M.NOTEBOOK, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    hits = [s for s in cells if "def welch_and_tost" in s]
    if len(hits) != 1:
        sys.exit(f"expected exactly one welch_and_tost cell, found {len(hits)}")
    src = hits[0].split("def acc_vector")[0]
    g = {"np": np, "math": math, "stats": stats}
    exec(src, g)
    return g["welch_and_tost"]


def load_records():
    """One row per finished run, with the schedule fields that define its configuration."""
    rows = []
    for path in sorted(glob.glob(os.path.join("results", "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"  skipping unreadable {os.path.basename(path)}: {e}")
            continue
        cfg = d.get("cfg")
        if not cfg or cfg.get("epochs") != EPOCHS:
            continue
        rows.append(dict(
            file=os.path.basename(path),
            method=cfg["method"], seed=cfg["seed"],
            B=cfg["n_blocks"], dense_len=cfg["acdc_dense_len"], cycle=cfg["acdc_cycle"],
            top1=d["top1"], peak_MiB=d["peak_param_mib"],
            ramp=cfg.get("compression_ramp_steps", 0)))
    return pd.DataFrame(rows)


def main():
    welch_and_tost = load_welch_and_tost()

    df = load_records()
    if df.empty:
        sys.exit("No 160-epoch results found in results/.")
    print(f"{len(df)} runs at {EPOCHS} epochs\n")

    # Baseline: synchronous AC/DC. Its peak and trajectory do not depend on the block count,
    # so the runs from the main sweep are the correct comparator for every block count.
    base = df[df.method == "acdc"]
    if len(base) < 2:
        sys.exit("Need at least two AC/DC seeds as the baseline.")
    base_acc = base.top1.to_numpy()
    print(f"baseline AC/DC: {base_acc.mean():.2f}% over {len(base_acc)} seeds "
          f"(peak {base.peak_MiB.mean():.2f} MiB)\n")

    out = []
    staggered = df[df.method.isin(["async_acdc", "async_pp"])]
    for (method, B, dl), g in staggered.groupby(["method", "B", "dense_len"], sort=True):
        acc = g.top1.to_numpy()
        row = dict(method=method, B=B, dense_len=dl, cycle=int(g.cycle.iloc[0]),
                   n_seeds=len(acc), top1_mean=acc.mean(),
                   top1_std=acc.std(ddof=1) if len(acc) > 1 else np.nan,
                   peak_MiB=g.peak_MiB.mean(),
                   reduction_pct=100 * (1 - g.peak_MiB.mean() / base.peak_MiB.mean()))
        if len(acc) >= 2:
            st = welch_and_tost(acc, base_acc, margin=MARGIN)
            row.update(diff_vs_acdc=st["diff"], p_welch=st["p_welch"],
                       tost_p=st["tost_p"], equivalent=st["equivalent"])
        else:
            row.update(diff_vs_acdc=acc.mean() - base_acc.mean(),
                       p_welch=np.nan, tost_p=np.nan, equivalent=None)
        out.append(row)

    res = pd.DataFrame(out).sort_values(["dense_len", "B", "method"], ascending=[False, True, True])
    pd.set_option("display.width", 200)
    show = res.copy()
    for c in ("top1_mean", "top1_std", "peak_MiB", "reduction_pct", "diff_vs_acdc"):
        show[c] = show[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    for c in ("p_welch", "tost_p"):
        show[c] = show[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    print(show.to_string(index=False))

    res.to_csv(os.path.join("results", "blocks_accuracy.csv"), index=False)
    print("\nwrote results/blocks_accuracy.csv")

    incomplete = res[res.n_seeds < 3]
    if len(incomplete):
        print(f"\n{len(incomplete)} group(s) still have fewer than three seeds:")
        for _, r in incomplete.iterrows():
            print(f"   {r.method} B={r.B} dense_len={r.dense_len}: {r.n_seeds} seed(s)")

    make_figure(res, base_acc)


def make_figure(res, base_acc):
    """Accuracy against block count, with the one-percentage-point equivalence band."""
    main = res[res.dense_len == 5]
    if main.empty or main.n_seeds.max() < 2:
        print("not enough data for the accuracy figure yet; skipping")
        return

    fig, ax = plt.subplots(figsize=(M.WIDE, 3.0))
    ax.yaxis.grid(True)
    mu = base_acc.mean()
    # Solid tint, not alpha: EPS cannot store transparency.
    ax.axhspan(mu - 1, mu + 1, facecolor=M.BAND, edgecolor="none", zorder=1)
    ax.axhline(mu, color=M.COL["acdc"], lw=1.2, ls=(0, (4, 2)), zorder=2)
    ax.text(8.35, mu + 1.04, f"AC/DC mean {mu:.2f}% $\\pm$ 1 point", ha="right", va="bottom",
            fontsize=M.FS_ANN, color=M.INK2)

    handles = []
    for key in ("async_acdc", "async_pp"):
        g = main[main.method == key].sort_values("B")
        if g.empty:
            continue
        err = g.top1_std.fillna(0).to_numpy()
        ax.errorbar(g.B, g.top1_mean, yerr=err, color=M.COL[key], lw=1.6,
                    marker=M.MARK[key], markersize=5.5, markeredgecolor="white",
                    markeredgewidth=0.8, capsize=3, zorder=3)
        handles.append(Line2D([], [], color=M.COL[key], lw=1.6, marker=M.MARK[key],
                              markersize=5.5, markeredgecolor="white", markeredgewidth=0.8,
                              label=M.NAME[key]))
    # Ring the leading candidate of the exploratory sweep. It is not labelled "equivalent": under
    # the paired analysis with a multiplicity correction (reanalysis_revision.py) no block count
    # above three is established as equivalent to AC/DC, so the choice still has to be confirmed.
    lead = main[main.B == LEADING_B]
    if not lead.empty:
        for _, r in lead.iterrows():
            ax.scatter([r.B], [r.top1_mean], s=120, facecolors="none", edgecolors=M.INK,
                       linewidths=0.9, zorder=5)
        # Placed well below the ring, in the empty band under the curve, so the leader line
        # and the text both stay clear of the neighbouring error bars.
        ax.annotate(f"$B = {LEADING_B}$: leading candidate of the\nexploratory sweep (not confirmed)",
                    xy=(LEADING_B, lead.top1_mean.min()), xytext=(6, -52),
                    textcoords="offset points", ha="left", va="top", fontsize=M.FS_ANN,
                    color=M.INK,
                    arrowprops=dict(arrowstyle="-", color=M.MUTED, lw=0.5, shrinkA=1, shrinkB=8))

    ax.set_xlabel("Number of blocks, $B$")
    ax.set_ylabel("Final top-1 accuracy (%)")
    ax.set_xlim(2.6, 8.4)
    ax.set_xticks(range(3, 9))
    lo, hi = ax.get_ylim()           # headroom at the bottom for the annotation
    ax.set_ylim(lo - 0.35, hi)
    if handles:
        M.legend_above(ax, handles, ncol=len(handles))
    M.add_title(ax, "Accuracy against number of blocks (ResNet-20)")
    M.save(fig, "07c_blocks_accuracy")


if __name__ == "__main__":
    main()
