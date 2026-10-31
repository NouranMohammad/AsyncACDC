"""Column-width figures for the IEEE conference version of the report.

The report's figures are drawn 6.2 in wide in a 9 pt sans-serif face. An IEEE two-column page has
3.5 in columns and asks for 8 pt Times New Roman in figures, so the four figures the paper uses are
redrawn here at 3.45 in with every axes placed explicitly in inches; nothing is scaled at print
time and no label position depends on matplotlib's automatic layout.

Data and colours come from the existing pipeline, so the numbers cannot drift from the report:
make_thesis_figures (M) for the house colours, the run loader and the notebook's schedule and
memory-ledger code, and reanalysis_revision (R) for the onset-aligned disturbance traces.
No training is performed.

Usage:  python make_ieee_figures.py
Writes: figures/ieee/ieee_schedule_memory.pdf, ieee_pareto.pdf, ieee_disturbance.pdf,
        ieee_blocks.pdf; set FIG_PREVIEW_DIR=<dir> to also write 300 dpi PNG previews there.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Anaconda + PyTorch both ship OpenMP

import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import make_thesis_figures as M  # noqa: E402  (house colours, loaders, notebook code)
import reanalysis_revision as R  # noqa: E402  (onset-aligned disturbance traces)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

OUT = os.path.join(HERE, "figures", "ieee")
PREVIEW = os.environ.get("FIG_PREVIEW_DIR")
W = 3.45                 # printed width of every figure (in); the IEEE column is 3.5 in
FS = 8                   # IEEE asks for 8 pt figure text
LEFT, RIGHT = 0.47, 0.07  # axes margins (in): room for tick labels and a rotated axis label
AX_W = W - LEFT - RIGHT
BAND = "#e8e7e1"         # a shade darker than M.BAND, so shaded regions survive printing

# Importing M set the report's style (Arial 9 pt); the paper needs Times 8 pt.
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0, "axes.labelpad": 2.5, "legend.handlelength": 1.6,
    "legend.columnspacing": 0.9, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.0,
    "pdf.fonttype": 42,
})


def axes_in(fig, left, bottom, width, height):
    """Axes placed in inches from the lower-left corner of the figure."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, bottom / fh, width / fw, height / fh])


def panel_label(fig, text, top_in):
    """Panel letter at the left edge of the figure, its baseline just above an axes."""
    fig.text(0.03 / W, top_in / fig.get_size_inches()[1], text, ha="left", va="bottom",
             fontsize=FS, color=M.INK)


def legend_row(fig, handles, top_in, ncol):
    """Frameless legend centred across the figure, its top edge at top_in."""
    fig.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, top_in / fig.get_size_inches()[1]))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), format="pdf")      # exact size, no tight bbox
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=300)
    plt.close(fig)
    print(f"  wrote figures/ieee/{name}.pdf")


# ----------------------------------------------------------------------------- figures
def fig_schedule_memory(plan_sync, plan_async, tr_sync, tr_async, dense, floor):
    """(a) synchronous and (b) staggered schedules, (c) the modelled memory of both."""
    E, B = plan_sync.dense.shape
    aw = AX_W - 0.06      # room for the half of the "160" tick label that overhangs the axes
    fig = plt.figure(figsize=(W, 2.95))
    for plan, key, bottom, label in ((plan_sync, "acdc", 2.14, "(a) AC/DC"),
                                     (plan_async, "async_acdc", 1.50, "(b) Async-AC/DC")):
        ax = axes_in(fig, LEFT, bottom, aw, 0.45)
        for b in range(B):
            row = B - 1 - b                                   # block 1 on top
            ax.add_patch(Rectangle((0, row + 0.12), E, 0.76, linewidth=0, facecolor=M.SPARSE_CELL))
            for a, z in M.spans(plan.dense[:, b]):
                ax.add_patch(Rectangle((a, row + 0.12), z - a, 0.76, linewidth=0,
                                       facecolor=M.COL[key]))
        ax.set_xlim(0, E)
        ax.set_ylim(0, B)
        ax.set_yticks([B - 1 - b + 0.5 for b in range(B)])
        ax.set_yticklabels([f"Block {b + 1}" for b in range(B)])
        ax.set_xticks(range(0, E + 1, 20))
        ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_label(fig, label, bottom + 0.47)

    ax = axes_in(fig, LEFT, 0.36, aw, 0.95)
    ax.yaxis.grid(True)
    x = np.arange(E + 1)
    ax.axhline(dense, color=M.INK2, lw=0.6, ls=(0, (4, 2)), zorder=1)
    ax.axhline(floor, color=M.INK2, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    ax.step(x, np.append(tr_sync, tr_sync[-1]), where="post", color=M.COL["acdc"], lw=1.1, zorder=3)
    ax.step(x, np.append(tr_async, tr_async[-1]), where="post", color=M.COL["async_acdc"],
            lw=1.3, zorder=4)
    ax.text(E - 1.5, dense + 0.08, f"Dense ({dense:.2f} MiB)", ha="right", va="bottom",
            fontsize=FS, color=M.INK2)
    # Two short lines right of epoch 120, so the drop of the staggered trace cannot cross it.
    ax.text(E - 1.5, floor + 0.08, f"Fully sparse\n({floor:.2f} MiB)", ha="right", va="bottom",
            fontsize=FS, color=M.INK2, linespacing=1.1)
    peak = tr_async.max()
    t_last = int(np.where(tr_async == peak)[0][-1]) + 1
    # Raised above the "fully sparse" note, with a leader to the last peak plateau.
    ax.annotate(f"Peak {peak:.2f} MiB\n({M.MINUS}{100 * (1 - peak / tr_sync.max()):.0f}%)",
                xy=(t_last - 1.5, peak), xytext=(t_last + 4, peak + 0.68), ha="left",
                va="center", fontsize=FS, color=M.INK2, linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color=M.MUTED, lw=0.5, shrinkA=0, shrinkB=1.5))
    ax.set_xlim(0, E)
    ax.set_ylim(0, 3.6)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticks(range(0, E + 1, 20))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Memory (MiB)")
    panel_label(fig, "(c) Modeled parameter-state memory", 1.33)

    legend_row(fig, [Patch(facecolor=M.COL["acdc"], label="AC/DC"),
                     Patch(facecolor=M.COL["async_acdc"], label="Async-AC/DC"),
                     Patch(facecolor=M.SPARSE_CELL, label="Sparse epoch")], 2.93, ncol=3)
    save(fig, "ieee_schedule_memory")


def fig_pareto(agg):
    YLO, YHI = 87.5, 93.25
    fig = plt.figure(figsize=(W, 2.6))
    ax = axes_in(fig, LEFT, 0.36, AX_W, 1.78)
    ax.grid(True)
    pts = agg.set_index("key")

    order = sorted(pts.index, key=lambda k: (pts.loc[k, "peak_MiB"], -pts.loc[k, "top1_mean"]))
    front, best = [], -np.inf
    for k in order:
        if pts.loc[k, "top1_mean"] > best + 1e-12:
            front.append(k)
            best = pts.loc[k, "top1_mean"]
    fx = [pts.loc[k, "peak_MiB"] for k in front]
    fy = [pts.loc[k, "top1_mean"] for k in front]
    ax.step(fx + [3.45], fy + [fy[-1]], where="post", color="#b9b8b1", lw=0.9, zorder=1)

    # (text x, text y, ha); None keeps the point's own height. SNIP and GraSP share one label.
    label_at = {
        "set": (0.55, 90.95, "left"), "rigl": (0.55, 89.85, "left"),
        "grasp": (0.55, 88.30, "left"), "snip": None,
        "async_pp": (1.66, 91.62, "left"), "async_acdc": (1.66, 90.12, "left"),
        "dense": (2.96, None, "right"), "gmp": (2.96, None, "right"),
        "acdc": (2.96, 90.42, "right"), "nm": (2.96, None, "right"),
        "topkast": (0.86, YLO + 0.36, "left"),
    }
    names = dict(M.NAME, grasp="SNIP, GraSP")
    for k in M.ORDER:
        x, y, sd = pts.loc[k, "peak_MiB"], pts.loc[k, "top1_mean"], pts.loc[k, "top1_std"]
        off_scale = y < YLO
        colour = M.COL.get(k, M.OTHER)
        marker = "v" if off_scale else M.MARK.get(k, "o")
        yy = YLO + 0.13 if off_scale else y
        if not off_scale:
            ax.errorbar(x, y, yerr=sd, fmt="none", ecolor=colour, elinewidth=0.8, capsize=2,
                        capthick=0.8, zorder=3)
        ax.scatter(x, yy, s=34 if k in M.HIGHLIGHT else 22, marker=marker, color=colour,
                   edgecolors="white", linewidths=0.7, clip_on=False,
                   zorder=5 if k == "async_acdc" else 4)
        if label_at[k] is None:
            continue
        tx, ty, ha = label_at[k]
        ty = yy if ty is None else ty
        text = f"{names[k]} ({y:.1f}%)" if off_scale else names[k]
        ax.annotate(text, xy=(x, yy), xytext=(tx, ty), ha=ha, va="center", fontsize=FS,
                    color=M.INK, fontweight="bold" if k in ("async_acdc", "async_pp") else "normal",
                    arrowprops=dict(arrowstyle="-", color=M.MUTED, lw=0.5, shrinkA=0, shrinkB=3))
    ax.set_xlim(0.0, 3.45)
    ax.set_ylim(YLO, YHI)
    ax.set_xlabel("Peak parameter-state memory (MiB)")
    ax.set_ylabel("Top-1 accuracy (%)")
    legend_row(fig, [M.dot_key(M.COL["async_pp"], "D", "Async-AC/DC++"),
                     M.dot_key(M.COL["async_acdc"], "o", "Async-AC/DC"),
                     M.dot_key(M.COL["acdc"], "s", "AC/DC"),
                     M.dot_key(M.COL["dense"], "^", "Dense"),
                     M.dot_key(M.OTHER, "o", "Other baselines"),
                     Line2D([], [], color="#b9b8b1", lw=0.9, label="Pareto frontier")], 2.58, ncol=3)
    save(fig, "ieee_pareto")


def fig_disturbance(traces):
    x = np.arange(-R.RAMP - R.BASE_W, R.RECOVERY)
    fig = plt.figure(figsize=(W, 2.3))
    ax = axes_in(fig, LEFT, 0.36, AX_W, 1.52)
    ax.yaxis.grid(True)
    ax.axvspan(-R.RAMP, 0, facecolor=BAND, edgecolor="none", zorder=0)
    ax.axhline(0, color=M.INK2, lw=0.5, zorder=1)
    ax.axvline(0, color=M.MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    k = np.ones(R.SMOOTH) / R.SMOOTH
    handles = []
    for m in ("acdc", "async_acdc", "async_pp"):
        sm = np.convolve(np.median(traces[m], axis=0), k, mode="same")
        sm[:R.SMOOTH // 2] = np.nan                    # the centred window runs off both ends
        sm[-(R.SMOOTH // 2):] = np.nan
        ax.plot(x, sm, color=M.COL[m], lw=1.2, zorder=3)
        handles.append(Line2D([], [], color=M.COL[m], lw=1.2, label=M.NAME[m]))
    handles.append(Patch(facecolor=BAND, label="Ramp of Async-AC/DC++"))
    # Notes sit at the bottom, clear of the post-boundary peaks.
    ax.text(-R.RAMP / 2, -10.6, "Ramp", ha="center", va="center", fontsize=FS, color=M.INK2)
    ax.text(6, -10.6, "Block fully sparse", ha="left", va="center", fontsize=FS, color=M.INK2)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-12.5, 22)
    ax.set_xlabel("Optimizer steps relative to full sparsity")
    ax.set_ylabel("Training loss change (%)")
    legend_row(fig, handles, 2.28, ncol=2)
    save(fig, "ieee_disturbance")


def fig_blocks(trade, acc, acdc_mean):
    fig = plt.figure(figsize=(W, 3.55))

    # (a) modelled peak against the number of blocks
    ax = axes_in(fig, LEFT, 2.15, AX_W, 1.1)
    ax.yaxis.grid(True)
    bs = trade.B.to_numpy()
    peaks = trade.peak_MiB.to_numpy()
    sync = float(trade.loc[trade.B == 1, "peak_MiB"].iloc[0])
    floor = sync - float(trade.loc[trade.B == 1, "excess_over_floor_MiB"].iloc[0])
    ax.axhline(sync, color=M.INK2, lw=0.6, ls=(0, (4, 2)), zorder=1)
    ax.axhline(floor, color=M.INK2, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    ax.plot(bs, floor + (sync - floor) / bs, color=M.MUTED, lw=0.9, ls=(0, (5, 2, 1, 2)), zorder=2)
    ax.plot(bs, peaks, color=M.COL["async_acdc"], lw=1.2, marker="o", markersize=3.6,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    for b in (3, 4):
        ax.scatter([b], [peaks[bs == b][0]], s=62, facecolors="none", edgecolors=M.INK,
                   linewidths=0.8, zorder=4)
    p3, p4 = peaks[bs == 3][0], peaks[bs == 4][0]
    ax.annotate(f"3 blocks: {M.MINUS}{100 * (1 - p3 / sync):.0f}%", xy=(3, p3), xytext=(5.0, 2.3),
                ha="left", va="center", fontsize=FS, color=M.INK,
                arrowprops=dict(arrowstyle="-", color=M.MUTED, lw=0.5, shrinkA=0, shrinkB=4))
    ax.annotate(f"4 blocks (candidate): {M.MINUS}{100 * (1 - p4 / sync):.0f}%", xy=(4, p4),
                xytext=(5.4, 1.62), ha="left", va="center", fontsize=FS, color=M.INK,
                arrowprops=dict(arrowstyle="-", color=M.MUTED, lw=0.5, shrinkA=0, shrinkB=4))
    ax.text(12.3, sync - 0.09, f"AC/DC ({sync:.2f} MiB)", ha="right", va="top", fontsize=FS,
            color=M.INK2)
    ax.text(12.3, floor - 0.09, f"Fully sparse ({floor:.2f} MiB)", ha="right", va="top",
            fontsize=FS, color=M.INK2)
    ax.set_xlim(0.6, 12.4)
    ax.set_ylim(0, 3.5)
    ax.set_xticks(range(1, 13))
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("Number of blocks")
    ax.set_ylabel("Peak memory (MiB)")
    legend_row(fig, [Line2D([], [], color=M.COL["async_acdc"], lw=1.2, marker="o", markersize=3.6,
                            markeredgecolor="white", markeredgewidth=0.6, label="Staggered peak"),
                     Line2D([], [], color=M.MUTED, lw=0.9, ls=(0, (5, 2, 1, 2)),
                            label="Perfectly equal blocks")], 3.53, ncol=2)
    panel_label(fig, "(a)", 3.29)

    # (b) measured accuracy against the number of blocks
    ax = axes_in(fig, LEFT, 0.36, AX_W, 1.1)
    ax.yaxis.grid(True)
    ax.axhspan(acdc_mean - 1, acdc_mean + 1, facecolor=BAND, edgecolor="none", zorder=0)
    ax.axhline(acdc_mean, color=M.COL["acdc"], lw=1.0, ls=(0, (4, 2)), zorder=1)
    handles = []
    for key, dx in (("async_acdc", -0.06), ("async_pp", 0.06)):
        g = acc[acc.method == key].sort_values("B")
        ax.errorbar(g.B + dx, g.top1_mean, yerr=g.top1_std, color=M.COL[key], lw=1.1,
                    marker=M.MARK[key], markersize=3.8, markeredgecolor="white",
                    markeredgewidth=0.6, elinewidth=0.8, capsize=2, capthick=0.8, zorder=3)
        handles.append(Line2D([], [], color=M.COL[key], lw=1.1, marker=M.MARK[key], markersize=3.8,
                              markeredgecolor="white", markeredgewidth=0.6, label=M.NAME[key]))
    lead = acc[acc.B == 4].top1_mean.mean()
    ax.scatter([4], [lead], s=150, facecolors="none", edgecolors=M.INK, linewidths=0.8, zorder=4)
    ax.annotate("Leading candidate\n(not confirmed)", xy=(4, lead), xytext=(4.28, 89.5),
                ha="left", va="center", fontsize=FS, color=M.INK, linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color=M.MUTED, lw=0.5, shrinkA=0, shrinkB=6))
    ax.set_xlim(2.6, 8.4)
    ax.set_ylim(89.2, 92.1)
    ax.set_xticks(range(3, 9))
    ax.set_xlabel("Number of blocks")
    ax.set_ylabel("Top-1 accuracy (%)")
    # The key overlays the band and the dashed mean line, exactly as they are drawn.
    band_key = (Patch(facecolor=BAND, edgecolor="none"),
                Line2D([], [], color=M.COL["acdc"], lw=1.0, ls=(0, (4, 2))))
    fig.legend(handles=handles + [band_key],
               labels=[h.get_label() for h in handles] + ["AC/DC mean " + "±" + " 1 point"],
               handler_map={tuple: HandlerTuple(ndivide=1)}, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.74 / fig.get_size_inches()[1]))
    panel_label(fig, "(b)", 1.5)
    save(fig, "ieee_blocks")


def main():
    print("loading results and notebook code ...")
    nb = M.load_notebook_code()
    runs = M.load_runs()
    agg = M.aggregate(runs)

    cfg = nb["Config"](**runs["async_acdc"][M.SEEDS[0]]["cfg"])
    plan_sync, plan_async = nb["plan_sync_acdc"](cfg), nb["plan_async_acdc"](cfg)
    sm = nb["SparseModel"](nb["build_model"](cfg, 10), cfg)
    nb["assign_blocks"](sm, cfg)
    tr_sync = nb["simulate_memory_trace"](sm, cfg, plan_sync)
    tr_async = nb["simulate_memory_trace"](sm, cfg, plan_async)
    dense = nb["dense_reference_memory"](sm, cfg).total_mib
    floor = nb["fully_sparse_memory"](sm, cfg, 1 - cfg.target_sparsity).total_mib
    fig_schedule_memory(plan_sync, plan_async, tr_sync, tr_async, dense, floor)

    fig_pareto(agg)

    runs_all, _ = R.load()
    _, traces = R.disturbance(runs_all)
    fig_disturbance(traces)

    trade = pd.read_csv(os.path.join("results", "blocks_tradeoff.csv"))
    acc = pd.read_csv(os.path.join("results", "blocks_accuracy.csv"))
    acc = acc[(acc.dense_len == 5) & acc.B.between(3, 8)]
    acdc_mean = float(agg.set_index("key").loc["acdc", "top1_mean"])
    fig_blocks(trade, acc, acdc_mean)

    print("\ncross-checks:")
    print(f"  peaks sync {tr_sync.max():.2f}  async {tr_async.max():.2f} MiB; dense {dense:.2f}, "
          f"floor {floor:.2f}")
    print("  B-sweep accuracy\n", acc[["method", "B", "top1_mean", "top1_std"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
