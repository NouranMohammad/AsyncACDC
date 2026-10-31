"""Regenerate the Async-AC/DC thesis figures as EPS, one file per panel.

Reads the finished 160-epoch, three-seed sweep (results/*.json) and reuses the notebook's own
schedule and memory-ledger code for the analytical figures, so every number matches the runs.
No training is performed: the notebook cells that are executed only define functions.

Design rules: every figure is authored at the width it is printed (the report's text width is
6.2 in) so that its 9-point text is never scaled down; no titles or author names inside the
figures (captions belong in the thesis); one colour per method across every figure; other
baselines recede to gray; legends sit outside the plotting area; marker shapes back up colour
for grayscale printing; no transparency anywhere (EPS cannot store it).

Usage:  python make_thesis_figures.py          -> figures/eps/*.eps and results/summary_full.csv
        set FIG_PREVIEW_DIR=<dir> to also write PNG previews there.
"""
import contextlib
import glob
import io
import json
import os
import types
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figures", "eps")
# The notebook holds the training code. In the GitHub repository it sits next to this script;
# in the original project folder it sits two levels up. ASYNC_ACDC_NOTEBOOK overrides both.
_NOTEBOOK_CANDIDATES = (
    os.environ.get("ASYNC_ACDC_NOTEBOOK", ""),
    os.path.join(HERE, "AsyncACDC_Implementation.ipynb"),
    os.path.join(HERE, "..", "..", "2644540_Nouran_Mohammed_AsyncACDC_ImplementationColab.ipynb"),
)
NOTEBOOK = next((os.path.abspath(p) for p in _NOTEBOOK_CANDIDATES if p and os.path.exists(p)),
                os.path.abspath(_NOTEBOOK_CANDIDATES[-1]))
PREVIEW = os.environ.get("FIG_PREVIEW_DIR")

EPOCHS = 160            # the main sweep; result files from shorter presets are ignored
SEEDS = (0, 1, 2)
REF_BLOCKS = 3          # the schedule the method-by-method figures compare at; the block-count
REF_DENSE_LEN = 5       # sweep lives in the same results/ directory and is filtered out here
WIDE = 6.2              # printed width of a full-width figure (inches)
HALF = 3.05             # printed width of one of two side-by-side panels
FS_ANN = 8.5            # font size of annotations inside the axes

# ----------------------------------------------------------------------------- style
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#a8a79f"
SPARSE_CELL = "#ebeae4"
BAND = "#f1f0eb"
OTHER, OTHER_LINE = "#9a9993", "#cfcec8"
BAND_GREEN = "#d6f0e5"   # solid tint of the Async-AC/DC green; EPS cannot store transparency

COL = {"async_pp": "#2a78d6", "acdc": "#eb6834", "async_acdc": "#1baf7a", "dense": "#52514e"}
MARK = {"async_pp": "D", "acdc": "s", "async_acdc": "o", "dense": "^"}
NAME = {"dense": "Dense", "gmp": "GMP", "snip": "SNIP", "grasp": "GraSP", "set": "SET",
        "rigl": "RigL", "topkast": "Top-KAST", "nm": "N:M",
        "acdc": "AC/DC", "async_acdc": "Async-AC/DC", "async_pp": "Async-AC/DC++"}
ORDER = ["dense", "gmp", "snip", "grasp", "set", "rigl", "topkast", "nm",
         "acdc", "async_acdc", "async_pp"]
HIGHLIGHT = ["dense", "acdc", "async_acdc", "async_pp"]
SPARSITY_RAMP = {"80%": ("#86b6ef", "o"), "90%": ("#2a78d6", "s"), "95%": ("#104281", "D")}
MINUS = "−"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.major.pad": 3.0, "ytick.major.pad": 3.0,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.columnspacing": 1.3,
    "legend.handletextpad": 0.5, "legend.borderaxespad": 0.0,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    # Type 3 glyphs are plain PostScript drawing procedures, so every EPS viewer renders the
    # text. Type 42 (embedded TrueType) needs a TrueType rasterizer in the viewer; viewers
    # without one draw the lines but silently drop every label and legend.
    "ps.fonttype": 3, "pdf.fonttype": 42,
})


def save(fig, name, tight=True):
    os.makedirs(OUT, exist_ok=True)
    kw = dict(bbox_inches="tight", pad_inches=0.03) if tight else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(os.path.join(OUT, name + ".eps"), format="eps", **kw)
    for w in caught:
        if issubclass(w.category, DeprecationWarning):      # library housekeeping, not the figure
            continue
        print(f"  WARNING [{name}]: {w.message}")
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=200, **kw)
    plt.close(fig)
    print(f"  wrote figures/eps/{name}.eps")


def legend_above(ax, handles, ncol):
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=ncol)


SHOW_TITLES = False       # titles live in the thesis captions; set True to draw them in the figure
TITLE_ROOM = 0.25 if SHOW_TITLES else 0.0   # extra inches for fixed-layout figures (05a, 05b)


def add_title(ax, text):
    """Left-aligned title placed above everything that already sits above the axes
    (legend, annotations), so it can never overlap them."""
    if not SHOW_TITLES:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_axes = ax.transAxes.inverted()
    top = 1.0
    for artist in [ax.get_legend(), *ax.texts]:
        if artist is None or not artist.get_visible():
            continue
        y1 = artist.get_window_extent(renderer).y1
        top = max(top, to_axes.transform((0, y1))[1])
    ax.annotate(text, xy=(0, top), xycoords="axes fraction", xytext=(0, 5),
                textcoords="offset points", ha="left", va="bottom", fontsize=10, color=INK,
                annotation_clip=False)


def line_key(key, label=None, **kw):
    style = dict(color=COL.get(key, OTHER), marker=MARK.get(key), markersize=5,
                 markeredgecolor="white", markeredgewidth=0.6, label=label or NAME[key])
    style.update(kw)
    return Line2D([], [], **style)


def dot_key(color, marker, label):
    return Line2D([], [], linestyle="none", marker=marker, markersize=6, color=color,
                  markeredgecolor="white", markeredgewidth=0.7, label=label)


# ----------------------------------------------------------------------------- data
def load_notebook_code():
    """Execute the notebook's schedule / ledger / shock definitions (no training)."""
    cells = ["".join(c["source"]) for c in json.load(open(NOTEBOOK, encoding="utf-8"))["cells"]
             if c["cell_type"] == "code"]

    def pick(marker, cut=None):
        hits = [s for s in cells if marker in s]
        assert len(hits) == 1, f"notebook cell not found uniquely: {marker!r}"
        return hits[0].split(cut)[0] if cut else hits[0]

    g = {"__name__": "notebook", "_nc": 10, "RunResult": object}
    specs = [("# --- Core imports", None), ("# --- Determinism", None), ("class Config:", None),
             ("class BasicBlock", None), ("class _STEMask", None), ("class LayerHandle", None),
             ("def score_magnitude", None), ("def topk_mask_from_scores", None),
             ("class MemoryBreakdown", None),
             ("def optimal_contiguous_partition", "# ---- demonstrate the partitioner"),
             ("class PhasePlan", "# ---- visualise the two schedules"),
             ("def simulate_memory_trace", "_sm_probe = SparseModel"),
             ("SHOCK_COLUMNS = [", "_spe = len(")]
    with contextlib.redirect_stdout(io.StringIO()):
        for marker, cut in specs:
            exec(pick(marker, cut), g)
    return g


def load_runs():
    """{method: {seed: result}} for the 160-epoch main sweep; every method must have all seeds.

    The block-count sweep (analyze_blocks.py) stores its runs in the same directory, so a
    method and seed no longer identify a run on their own. Every figure in this script compares
    methods at the reference schedule, so runs with any other block count or dense-phase length
    are skipped here rather than colliding with the main sweep.
    """
    runs = {}
    for path in glob.glob(os.path.join(RESULTS, "*.json")):
        d = json.load(open(path))
        cfg = d["cfg"]
        if cfg["epochs"] != EPOCHS:
            continue
        if (cfg["n_blocks"], cfg["acdc_dense_len"]) != (REF_BLOCKS, REF_DENSE_LEN):
            continue
        seed = cfg["seed"]
        assert seed not in runs.get(d["method_key"], {}), f"two results for {d['method_key']} seed {seed}"
        runs.setdefault(d["method_key"], {})[seed] = d
    incomplete = [k for k in ORDER if set(runs.get(k, {})) != set(SEEDS)]
    assert not incomplete, f"incomplete results for: {incomplete}"
    return runs


def aggregate(runs):
    """Mean and standard deviation over seeds; memory columns are set by the schedule."""
    rows = []
    for k in ORDER:
        rs = [runs[k][s] for s in SEEDS]
        top1 = np.array([r["top1"] for r in rs])
        top5 = np.array([r["top5"] for r in rs])
        rows.append(dict(key=k, method=NAME[k], n=len(rs),
                         top1_mean=top1.mean(), top1_std=top1.std(ddof=1),
                         top5_mean=top5.mean(), top5_std=top5.std(ddof=1),
                         sparsity=np.mean([r["final_sparsity"] for r in rs]),
                         peak_MiB=max(r["peak_param_mib"] for r in rs),
                         mean_MiB=np.mean([r["mean_param_mib"] for r in rs]),
                         final_MiB=np.mean([r["final_param_mib"] for r in rs]),
                         PFLOPs=np.mean([r["train_flops"] for r in rs]) / 1e15))
    agg = pd.DataFrame(rows)
    agg["peak_vs_dense"] = 100 * agg["peak_MiB"] / agg.loc[agg.key == "dense", "peak_MiB"].iloc[0]
    agg.to_csv(os.path.join(RESULTS, "summary_full.csv"), index=False)
    return agg


def mean_curve(runs, key, field):
    return np.mean([runs[key][s][field] for s in SEEDS], axis=0)


def spans(mask):
    """Half-open [start, end) index ranges over which a boolean vector is True."""
    out, start = [], None
    for t, v in enumerate(mask):
        if v and start is None:
            start = t
        if not v and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


# ----------------------------------------------------------------------------- figures
def fig_schedule(nb, plan, cfg, key, name, title):
    E, B = plan.dense.shape
    W, M = nb["_middle_window"](cfg)
    # 5.4 in of axes plus the legend on the right gives the 6.2 in printed width.
    fig, ax = plt.subplots(figsize=(5.4, 1.9))
    for b in range(B):
        row = B - 1 - b                               # block 1 on top
        ax.add_patch(Rectangle((0, row + 0.12), E, 0.76, linewidth=0, facecolor=SPARSE_CELL))
        for a, z in spans(plan.dense[:, b]):          # one patch per dense window
            ax.add_patch(Rectangle((a, row + 0.12), z - a, 0.76, linewidth=0, facecolor=COL[key]))
    ax.set_xlim(0, E)
    ax.set_ylim(0, B + 1.05)
    ax.set_yticks([B - 1 - b + 0.5 for b in range(B)])
    ax.set_yticklabels([f"Block {b + 1}" for b in range(B)])
    ax.set_xticks(range(0, E + 1, 20))
    ax.set_xlabel("Epoch")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)

    y = B + 0.32
    for x0, x1, label in [(0, W, "Warm-up"), (W, W + M, "Alternating phase"),
                          (W + M, E, "Sparse fine-tuning")]:
        a, z = x0 + 0.5, x1 - 0.5
        ax.plot([a, z], [y, y], color=MUTED, lw=0.6, solid_capstyle="butt", clip_on=False)
        for xx in (a, z):
            ax.plot([xx, xx], [y - 0.14, y], color=MUTED, lw=0.6, solid_capstyle="butt",
                    clip_on=False)
        ax.text((x0 + x1) / 2, y + 0.12, label, ha="center", va="bottom", fontsize=FS_ANN,
                color=INK2)

    ax.legend(handles=[Patch(facecolor=COL[key], label="Dense"),
                       Patch(facecolor=SPARSE_CELL, label="Sparse")],
              loc="center left", bbox_to_anchor=(1.01, 0.38), handlelength=1.0, handleheight=1.0)
    add_title(ax, title)
    save(fig, name)


def fig_memory_analytical(tr_sync, tr_async, dense, floor):
    E = len(tr_sync)
    x = np.arange(E + 1)
    fig, ax = plt.subplots(figsize=(WIDE, 3.0))
    ax.yaxis.grid(True)
    ax.axhline(dense, color=INK2, lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.axhline(floor, color=INK2, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
    ax.step(x, np.append(tr_sync, tr_sync[-1]), where="post", color=COL["acdc"], zorder=3)
    ax.step(x, np.append(tr_async, tr_async[-1]), where="post", color=COL["async_acdc"], zorder=4)
    ax.text(E - 1.5, dense + 0.07, f"Dense training ({dense:.2f} MiB)", ha="right", va="bottom",
            fontsize=FS_ANN, color=INK2)
    # Below the floor line, so it cannot touch the vertical drop of the traces at epoch 120.
    ax.text(E - 1.5, floor - 0.07, f"Fully sparse ({floor:.2f} MiB)", ha="right", va="top",
            fontsize=FS_ANN, color=INK2)
    peak = tr_async.max()
    t_last = int(np.where(tr_async == peak)[0][-1]) + 1
    ax.text(t_last + 3.5, peak,
            f"Peak {peak:.2f} MiB\n({MINUS}{100 * (1 - peak / tr_sync.max()):.0f}% vs. AC/DC)",
            ha="left", va="center", fontsize=FS_ANN, color=INK2, linespacing=1.3)
    ax.set_xlim(0, E)
    ax.set_ylim(0, 3.6)
    ax.set_xticks(range(0, E + 1, 20))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameter-state memory (MiB)")
    legend_above(ax, [line_key("acdc", marker=None), line_key("async_acdc", marker=None)], ncol=2)
    add_title(ax, "Analytical parameter-state memory during training")
    save(fig, "02_memory_trace_analytical")


def fig_pareto(agg):
    YLO, YHI = 87.5, 93.25
    fig, ax = plt.subplots(figsize=(WIDE, 4.0))
    ax.grid(True)
    pts = agg.set_index("key")

    # Pareto frontier (minimise memory, maximise accuracy) drawn as the step it really is.
    order = sorted(pts.index, key=lambda k: (pts.loc[k, "peak_MiB"], -pts.loc[k, "top1_mean"]))
    front, best = [], -np.inf
    for k in order:
        if pts.loc[k, "top1_mean"] > best + 1e-12:
            front.append(k)
            best = pts.loc[k, "top1_mean"]
    fx = [pts.loc[k, "peak_MiB"] for k in front]
    fy = [pts.loc[k, "top1_mean"] for k in front]
    ax.step(fx + [3.45], fy + [fy[-1]], where="post", color="#b9b8b1", lw=1.0, zorder=1)

    label_at = {  # key: (text x, text y, ha); None = at the point's own height
        "set": (0.60, 90.78, "left"), "rigl": (0.60, 89.96, "left"),
        "grasp": (0.60, 88.68, "left"), "snip": (0.60, 87.92, "left"),
        "async_pp": (1.70, 91.32, "left"), "async_acdc": (1.70, 90.38, "left"),
        "dense": (3.00, None, "right"), "gmp": (3.00, None, "right"),
        "acdc": (3.00, None, "right"), "nm": (3.00, None, "right"),
        "topkast": (1.10, YLO + 0.30, "left"),
    }
    for k in ORDER:
        x, y, sd = pts.loc[k, "peak_MiB"], pts.loc[k, "top1_mean"], pts.loc[k, "top1_std"]
        off_scale = y < YLO
        colour = COL.get(k, OTHER)
        marker = "v" if off_scale else MARK.get(k, "o")
        yy = YLO + 0.14 if off_scale else y
        if not off_scale:                             # one standard deviation over the seeds
            ax.errorbar(x, y, yerr=sd, fmt="none", ecolor=colour, elinewidth=0.9, capsize=2.5,
                        capthick=0.9, zorder=3)
        # The two staggered methods nearly coincide; the proposed method is drawn on top so
        # that both markers stay visible.
        ax.scatter(x, yy, s=56 if k in HIGHLIGHT else 38, marker=marker, color=colour,
                   edgecolors="white", linewidths=0.9, clip_on=False,
                   zorder=5 if k == "async_acdc" else 4)
        tx, ty, ha = label_at[k]
        ty = yy if ty is None else ty
        text = f"{NAME[k]} ({y:.1f}%)" if off_scale else NAME[k]
        weight = "bold" if k in ("async_acdc", "async_pp") else "normal"
        ax.annotate(text, xy=(x, yy), xytext=(tx, ty), ha=ha, va="center", fontsize=9,
                    color=INK, fontweight=weight,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=4))
    ax.set_xlim(0.0, 3.45)
    ax.set_ylim(YLO, YHI)
    ax.set_xlabel("Peak parameter-state memory (MiB)")
    ax.set_ylabel("Top-1 accuracy (%), mean of three seeds")
    legend_above(ax, [dot_key(COL["async_pp"], "D", "Async-AC/DC++"),
                      dot_key(COL["async_acdc"], "o", "Async-AC/DC"),
                      dot_key(COL["acdc"], "s", "AC/DC"),
                      dot_key(COL["dense"], "^", "Dense"),
                      dot_key(OTHER, "o", "Other baselines"),
                      dot_key(OTHER, "v", "Below plotted range"),
                      Line2D([], [], color="#b9b8b1", lw=1.0, label="Pareto frontier"),
                      Line2D([], [], color=INK2, lw=0.9, marker="_", markersize=6,
                             label="One standard deviation")], ncol=4)
    add_title(ax, "Accuracy vs. peak training memory (ResNet-20, CIFAR-10, 90% sparsity)")
    save(fig, "03_pareto")


def fig_accuracy(runs):
    E = EPOCHS
    x = np.arange(1, E + 1)
    fig, ax = plt.subplots(figsize=(WIDE, 3.0))
    ax.yaxis.grid(True)
    for k in ORDER:
        if k in HIGHLIGHT:
            continue
        ax.plot(x, mean_curve(runs, k, "acc_curve"), color=OTHER_LINE, lw=0.8, zorder=1)
    for k in HIGHLIGHT:
        ax.plot(x, mean_curve(runs, k, "acc_curve"), color=COL[k], lw=1.3, zorder=3,
                marker=MARK[k], markevery=(19, 20), markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.6)
    acc = mean_curve(runs, "topkast", "acc_curve")
    ax.text(E - 1.5, acc[-1] + 2.5, NAME["topkast"], ha="right", va="bottom", fontsize=FS_ANN,
            color=INK2)
    ax.set_xlim(0, E + 2)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, E + 1, 20))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test top-1 accuracy (%)")
    legend_above(ax, [line_key(k) for k in HIGHLIGHT]
                 + [Line2D([], [], color=OTHER_LINE, lw=1.2, label="Other baselines")], ncol=5)
    add_title(ax, "Test accuracy")
    save(fig, "04a_test_accuracy")


def fig_memory_measured(runs):
    fig, ax = plt.subplots(figsize=(WIDE, 3.0))
    ax.yaxis.grid(True)
    styles = [("dense", dict(color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=1)),
              ("rigl", dict(color=MUTED, lw=0.9, ls=(0, (1, 1.5)), zorder=1)),
              ("acdc", dict(color=COL["acdc"], lw=1.3, zorder=2)),
              ("async_acdc", dict(color=COL["async_acdc"], lw=1.9, zorder=3)),
              ("async_pp", dict(color=COL["async_pp"], lw=1.2, ls=(0, (3, 2)), zorder=4))]
    handles = []
    for k, st in styles:
        r = runs[k][SEEDS[0]]                          # the trace is fixed by the schedule
        ax.plot(r["mem_epochs"], r["mem_mib"], **st)
        handles.append(Line2D([], [], label=NAME[k], **{a: b for a, b in st.items() if a != "zorder"}))
    ax.set_xlim(0, EPOCHS)
    ax.set_ylim(0, 3.6)
    ax.set_xticks(range(0, EPOCHS + 1, 20))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameter-state memory (MiB)")
    legend_above(ax, handles, ncol=5)
    add_title(ax, "Measured parameter-state memory")
    save(fig, "04b_memory_trace_measured")


def fig_seed_dots(runs, agg):
    means = agg.set_index("key")["top1_mean"]
    keys = sorted([k for k in ORDER if k != "topkast"], key=lambda k: means[k])   # best on top
    acdc_mean = means["acdc"]
    fig, ax = plt.subplots(figsize=(WIDE, 3.4))
    ax.xaxis.grid(True)
    ax.add_patch(Rectangle((acdc_mean - 1, -0.6), 2.0, len(keys) + 0.2, facecolor=BAND,
                           linewidth=0, zorder=0))
    ax.axvline(acdc_mean, color=COL["acdc"], lw=0.8, ls=(0, (4, 2)), zorder=1)
    for i, k in enumerate(keys):
        vals = [runs[k][s]["top1"] for s in SEEDS]
        ax.plot([np.mean(vals)] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.4, solid_capstyle="butt",
                zorder=3)
        ax.scatter(vals, [i] * len(vals), s=36, marker=MARK.get(k, "o"), color=COL.get(k, OTHER),
                   edgecolors="white", linewidths=0.7, zorder=4)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([NAME[k] for k in keys])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(87.5, 93.25)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlabel("Final top-1 accuracy (%)")
    legend_above(ax, [dot_key(OTHER, "o", "One seed"),
                      Line2D([], [], color=INK, lw=1.4, label="Mean of three seeds"),
                      Line2D([], [], color=COL["acdc"], lw=0.8, ls=(0, (4, 2)), label="AC/DC mean"),
                      Patch(facecolor=BAND, label="AC/DC mean ± 1 point")], ncol=4)
    add_title(ax, "Final accuracy of every seed")
    save(fig, "09_seed_accuracy")


def feasible_winners(agg):
    lo, hi = agg["final_MiB"].min() * 0.98, agg["peak_MiB"].max() * 1.02   # notebook's range
    edges = sorted({lo, hi, *[p for p in agg["peak_MiB"] if lo <= p <= hi]})
    segs = []
    for a, b in zip(edges[:-1], edges[1:]):
        feas = agg[agg["peak_MiB"] <= a + 1e-12]
        if feas.empty:
            continue
        row = feas.loc[feas["top1_mean"].idxmax()]
        if segs and segs[-1]["key"] == row.key:
            segs[-1]["x1"] = b
        else:
            segs.append({"key": row.key, "x0": a, "x1": b, "acc": row.top1_mean})
    return lo, hi, segs


# Shared axes geometry of 05a and 05b (inches), so the two panels align exactly when stacked.
ISO_LEFT, ISO_BOTTOM, ISO_PLOT_W = 0.8, 0.5, 5.3


def fig_iso_accuracy(agg):
    lo, hi, segs = feasible_winners(agg)
    plot_h = 2.25
    h = ISO_BOTTOM + plot_h + 0.12 + TITLE_ROOM
    fig = plt.figure(figsize=(WIDE, h))
    ax = fig.add_axes([ISO_LEFT / WIDE, ISO_BOTTOM / h, ISO_PLOT_W / WIDE, plot_h / h])
    ax.yaxis.grid(True)
    for i, s in enumerate(segs):
        c = COL.get(s["key"], OTHER)
        ax.plot([s["x0"], s["x1"]], [s["acc"], s["acc"]], color=c, lw=2.6,
                solid_capstyle="butt", zorder=3)
        if i + 1 < len(segs):
            ax.plot([s["x1"], s["x1"]], [s["acc"], segs[i + 1]["acc"]], color=MUTED, lw=0.7,
                    ls=(0, (1, 1.5)), zorder=2)
        ax.scatter(s["x0"], s["acc"], s=42, marker=MARK.get(s["key"], "o"), color=c,
                   edgecolors="white", linewidths=0.9, zorder=4)
    label_x = {"async_pp": 2.05}                       # keep clear of the AC/DC leader line
    for s in segs:
        k, mid = s["key"], (s["x0"] + s["x1"]) / 2
        label = f"{NAME[k]} {s['acc']:.1f}%"
        if s["x1"] - s["x0"] < 0.15 and k != "dense":  # too narrow for a label: leader line
            ax.annotate(label, xy=(mid, s["acc"]), xytext=(1.34, 91.55), ha="right", va="center",
                        fontsize=FS_ANN, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=3))
        elif k == "dense":
            ax.text(s["x0"] - 0.05, s["acc"], label, ha="right", va="center", fontsize=FS_ANN,
                    color=INK)
        else:
            ax.text(label_x.get(k, mid), s["acc"] + 0.18, label, ha="center", va="bottom",
                    fontsize=FS_ANN, color=INK)
    acdc = agg.set_index("key").loc["acdc"]
    ax.scatter(acdc.peak_MiB, acdc.top1_mean, s=42, marker="s", color=COL["acdc"],
               edgecolors="white", linewidths=0.9, zorder=4)
    ax.annotate(f"AC/DC {acdc.top1_mean:.1f}%\n(needs {acdc.peak_MiB:.2f} MiB)",
                xy=(acdc.peak_MiB, acdc.top1_mean), xytext=(2.55, 92.1), ha="center", va="center",
                fontsize=FS_ANN, color=INK2, linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=2, shrinkB=4))
    ax.set_xlim(lo - 0.05, hi + 0.05)
    ax.set_ylim(90.0, 93.25)
    ax.set_xlabel("Peak parameter-memory budget (MiB)")
    ax.set_ylabel("Best feasible top-1 (%)")
    add_title(ax, "Best achievable accuracy under a memory budget")
    save(fig, "05a_iso_memory_best_accuracy", tight=False)


def fig_iso_winner(agg):
    lo, hi, segs = feasible_winners(agg)
    plot_h = 0.48
    h = ISO_BOTTOM + plot_h + 0.38 + TITLE_ROOM
    fig = plt.figure(figsize=(WIDE, h))
    ax = fig.add_axes([ISO_LEFT / WIDE, ISO_BOTTOM / h, ISO_PLOT_W / WIDE, plot_h / h])
    gap = 0.012
    for s in segs:
        c = COL.get(s["key"], OTHER)
        ax.add_patch(Rectangle((s["x0"] + gap, 0.0), s["x1"] - s["x0"] - 2 * gap, 1.0,
                               facecolor=c, linewidth=0))
        width_in = (s["x1"] - s["x0"]) / (hi - lo + 0.1) * ISO_PLOT_W
        name = NAME[s["key"]]
        if width_in > 1.0:
            txt_col = INK if s["key"] not in ("async_pp", "dense") else "white"
            ax.text((s["x0"] + s["x1"]) / 2, 0.5, name, ha="center", va="center", fontsize=FS_ANN,
                    color=txt_col)
        else:
            ax.annotate(name, xy=((s["x0"] + s["x1"]) / 2, 1.0), xytext=(s["x0"] - 0.1, 1.45),
                        ha="right", va="center", fontsize=FS_ANN, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=0.5),
                        annotation_clip=False)
    ax.set_xlim(lo - 0.05, hi + 0.05)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Peak parameter-memory budget (MiB)")
    add_title(ax, "Most accurate method that fits the budget")
    save(fig, "05b_iso_memory_winning_method", tight=False)


def shock_table(nb, runs):
    """Compression shocks of every seed, one row per (seed, event)."""
    spe = len(runs["acdc"][SEEDS[0]]["loss_curve"]) // EPOCHS      # steps per epoch
    frames = {}
    for k in ("acdc", "async_acdc", "async_pp"):
        parts = []
        for s in SEEDS:
            r = runs[k][s]
            res = types.SimpleNamespace(loss_curve=r["loss_curve"], events=r["events"])
            fr = nb["transition_shocks"](res, spe)
            fr = fr[fr["kind"] == "compress"]
            # A synchronous compression is ONE event, but it is logged once per block with an
            # identical loss window; keep one row per epoch so it is not counted three times.
            parts.append(fr.drop_duplicates(subset=["epoch"]).assign(seed=s))
        frames[k] = pd.concat(parts).reset_index(drop=True)
    return frames


def fig_shock_events(frames):
    fig, ax = plt.subplots(figsize=(WIDE, 2.9))
    ax.yaxis.grid(True)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    dodge = {"acdc": -1.0, "async_acdc": 0.0, "async_pp": 1.0}
    for k, fr in frames.items():
        g = fr.groupby("epoch")["shock_%"]
        ep = g.mean().index.to_numpy(dtype=float) + dodge[k]
        ax.vlines(ep, g.min().to_numpy(), g.max().to_numpy(), color=COL[k], lw=0.9, zorder=2)
        ax.scatter(ep, g.mean().to_numpy(), s=28, marker=MARK[k], color=COL[k],
                   edgecolors="white", linewidths=0.7, zorder=3)
    ax.set_xlim(0, 125)
    ax.set_xticks(range(0, 121, 10))
    ax.set_xlabel("Epoch of compression event")
    ax.set_ylabel("Training-loss shock (%)")
    legend_above(ax, [dot_key(COL[k], MARK[k], NAME[k]) for k in frames], ncol=3)
    add_title(ax, "Loss shock at each compression event")
    save(fig, "06a_transition_shock_events")


def fig_shock_distribution(frames):
    fig, ax = plt.subplots(figsize=(4.5, 2.9))
    ax.yaxis.grid(True)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    rng = np.random.default_rng(0)
    for i, (k, fr) in enumerate(frames.items()):
        v = fr["shock_%"].to_numpy()
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(Rectangle((i - 0.26, q1), 0.52, q3 - q1, facecolor="none", edgecolor=INK2,
                               linewidth=0.8, zorder=2))
        ax.plot([i - 0.26, i + 0.26], [med, med], color=INK, lw=1.6, solid_capstyle="butt", zorder=4)
        jitter = rng.uniform(-0.19, 0.19, size=len(v))
        ax.scatter(i + jitter, v, s=17, marker=MARK[k], color=COL[k], edgecolors="white",
                   linewidths=0.5, zorder=3)
    ax.set_xticks(range(len(frames)))
    ax.set_xticklabels([f"{NAME[k]}\n(n = {len(fr)})" for k, fr in frames.items()])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(frames) - 0.4)
    ax.set_ylabel("Training-loss shock (%)")
    legend_above(ax, [Line2D([], [], color=INK, lw=1.6, label="Median"),
                      Patch(facecolor="none", edgecolor=INK2, linewidth=0.8, label="Interquartile range")],
                 ncol=2)
    add_title(ax, "Distribution of compression shocks")
    save(fig, "06b_transition_shock_distribution")


def fig_blocks(nb, cfg, sm, tr_sync_peak, floor):
    from dataclasses import replace
    bs = list(range(1, 13))
    peaks = []
    for B in bs:
        cb = replace(cfg, n_blocks=B, acdc_cycle=max(cfg.acdc_cycle, B * cfg.acdc_dense_len))
        nb["assign_blocks"](sm, cb)
        plan = nb["plan_async_acdc"](cb) if B > 1 else nb["plan_sync_acdc"](cb)
        peaks.append(nb["simulate_memory_trace"](sm, cb, plan).max())
    peaks = np.array(peaks)
    ideal = np.array([floor + (tr_sync_peak - floor) / B for B in bs])

    fig, ax = plt.subplots(figsize=(WIDE, 3.2))
    ax.yaxis.grid(True)
    ax.axhline(tr_sync_peak, color=INK2, lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.axhline(floor, color=INK2, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
    ax.plot(bs, ideal, color=MUTED, lw=1.0, ls=(0, (5, 2, 1, 2)), zorder=2)
    ax.plot(bs, peaks, color=COL["async_acdc"], lw=1.6, marker="o", markersize=5.5,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    # Block counts 3 to 8 were all trained, so neither ring means "the one that was trained".
    # They mark the setting used for the main comparison and the leading candidate of the
    # exploratory block-count sweep, which still has to be confirmed.
    i3, i4 = bs.index(3), bs.index(4)
    for b, i in ((3, i3), (4, i4)):
        ax.scatter([b], [peaks[i]], s=95, facecolors="none", edgecolors=INK, linewidths=0.9,
                   zorder=4)
    ax.annotate(f"B = 3: {peaks[i3]:.2f} MiB, {MINUS}{100 * (1 - peaks[i3] / tr_sync_peak):.0f}%",
                xy=(3, peaks[i3]), xytext=(4.9, 2.55), ha="left", va="center", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=6))
    ax.annotate(f"B = 4 (leading candidate): {peaks[i4]:.2f} MiB, {MINUS}{100 * (1 - peaks[i4] / tr_sync_peak):.0f}%",
                xy=(4, peaks[i4]), xytext=(5.4, 1.78), ha="left", va="center", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=6))
    ax.text(12.3, tr_sync_peak + 0.07, f"AC/DC, synchronous ({tr_sync_peak:.2f} MiB)", ha="right",
            va="bottom", fontsize=FS_ANN, color=INK2)
    ax.text(12.3, floor - 0.07, f"Fully sparse ({floor:.2f} MiB)", ha="right", va="top",
            fontsize=FS_ANN, color=INK2)
    ax.set_xlim(0.6, 12.4)
    ax.set_ylim(0, 3.6)
    ax.set_xticks(bs)
    ax.set_xlabel("Number of blocks, B")
    ax.set_ylabel("Peak parameter-state memory (MiB)")
    legend_above(ax, [Line2D([], [], color=COL["async_acdc"], lw=1.6, marker="o", markersize=5.5,
                             markeredgecolor="white", markeredgewidth=0.8, label="Async-AC/DC peak"),
                      Line2D([], [], color=MUTED, lw=1.0, ls=(0, (5, 2, 1, 2)),
                             label="Ideal: sparse floor + dense excess / B")], ncol=2)
    add_title(ax, "Peak memory vs. number of blocks (ResNet-20)")
    save(fig, "07_ablation_blocks")
    return peaks


def fig_blocks_tradeoff(nb, cfg, sm):
    """What a block count costs the schedule, in the one unit both quantities share: epochs.

    Peak memory alone (fig 07) favours many blocks. The exploration budget is capped at
    E - F block-epochs whatever B is, so splitting further divides the same budget among
    more blocks and stretches the cycle. Both series below are epochs, so they belong on
    one axis; no second scale is introduced.
    """
    from dataclasses import replace
    bs = list(range(1, 13))
    lo, hi, gap = [], [], []
    for B in bs:
        cb = replace(cfg, n_blocks=B, acdc_cycle=max(cfg.acdc_cycle, B * cfg.acdc_dense_len))
        nb["assign_blocks"](sm, cb)
        plan = nb["plan_async_acdc"](cb) if B > 1 else nb["plan_sync_acdc"](cb)
        per = plan.dense_epochs_per_block()
        lo.append(int(np.min(per)))
        hi.append(int(np.max(per)))
        waits = []
        for b in range(B):
            sp = spans(plan.dense[:, b])
            waits += [sp[i + 1][0] - sp[i][1] for i in range(len(sp) - 1)]
        gap.append(int(max(waits)) if waits else 0)

    fig, ax = plt.subplots(figsize=(WIDE, 3.0))
    ax.yaxis.grid(True)
    # Solid tint, not alpha: EPS cannot store transparency.
    ax.fill_between(bs, lo, hi, facecolor=BAND_GREEN, edgecolor="none", zorder=1)
    ax.plot(bs, lo, color=COL["async_acdc"], lw=1.6, marker="o", markersize=5.5,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.plot(bs, gap, color=INK2, lw=1.4, marker="s", markersize=5, ls=(0, (5, 2)),
            markeredgecolor="white", markeredgewidth=0.8, zorder=3)

    # B = 3 sits above its marker, clear of the descending solid line; B = 4 sits to its
    # left, clear of the rising dashed line that passes just below and to the right of it.
    for b, note, off, ha, va in ((3, "B = 3", (0, 11), "center", "bottom"),
                                 (4, "B = 4 (candidate)", (-11, -3), "right", "center")):
        i = bs.index(b)
        ax.scatter([b], [lo[i]], s=95, facecolors="none", edgecolors=INK, linewidths=0.9, zorder=4)
        ax.annotate(note, xy=(b, lo[i]), xytext=off, textcoords="offset points",
                    ha=ha, va=va, fontsize=FS_ANN, color=INK)
    ax.set_xlim(0.6, 12.4)
    ax.set_ylim(0, 62)
    ax.set_xticks(bs)
    ax.set_xlabel("Number of blocks, B")
    ax.set_ylabel("Epochs")
    legend_above(ax, [
        Line2D([], [], color=COL["async_acdc"], lw=1.6, marker="o", markersize=5.5,
               markeredgecolor="white", markeredgewidth=0.8,
               label="Dense epochs for the least-served block"),
        Patch(facecolor=BAND_GREEN, label="Range across blocks"),
        Line2D([], [], color=INK2, lw=1.4, marker="s", markersize=5, ls=(0, (5, 2)),
               markeredgecolor="white", markeredgewidth=0.8,
               label="Longest sparse interval for a block")], ncol=3)
    add_title(ax, "Cost of splitting further (ResNet-20)")
    save(fig, "07b_blocks_schedule_cost")
    return lo, hi, gap


def fig_projection(proj, optimiser, name, title):
    sub = proj[proj["optimiser"] == optimiser]
    fig, ax = plt.subplots(figsize=(HALF, 2.5))
    ax.yaxis.grid(True)
    dense = sub["dense GiB"].iloc[0]
    ax.axhline(dense, color=INK2, lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.text(8.25, dense + 0.008, f"Dense = AC/DC ({dense:.2f} GiB)", ha="right", va="bottom",
            fontsize=FS_ANN, color=INK2)
    handles = []
    for sp, (colour, marker) in SPARSITY_RAMP.items():
        g = sub[sub["sparsity"] == sp].sort_values("B")
        ax.plot(g["B"], g["peak GiB"], color=colour, lw=1.5, marker=marker, markersize=5,
                markeredgecolor="white", markeredgewidth=0.7, zorder=3)
        handles.append(Line2D([], [], color=colour, lw=1.5, marker=marker, markersize=5,
                              markeredgecolor="white", markeredgewidth=0.7, label=f"{sp} sparsity"))
    ax.set_xticks(sorted(sub["B"].unique()))
    ax.set_xlim(0.6, 8.4)
    ax.set_ylim(0, 0.42)
    ax.set_xlabel("Number of blocks, B")
    ax.set_ylabel("Peak parameter state (GiB)")
    legend_above(ax, handles, ncol=1)               # one column keeps the 80/90/95 order readable
    add_title(ax, title)
    save(fig, name)


# ----------------------------------------------------------------------------- main
def main():
    print("loading results and notebook code ...")
    nb = load_notebook_code()
    runs = load_runs()
    agg = aggregate(runs)
    # The ResNet-50 projection is analytical (layer sizes, sparsity, optimiser), so the file
    # exported by any preset holds the same numbers.
    proj = pd.read_csv(sorted(glob.glob(os.path.join(RESULTS, "resnet50_projection_*.csv")))[0])

    Config = nb["Config"]
    cfg = Config(**runs["async_acdc"][SEEDS[0]]["cfg"])
    plan_sync, plan_async = nb["plan_sync_acdc"](cfg), nb["plan_async_acdc"](cfg)
    sm = nb["SparseModel"](nb["build_model"](cfg, 10), cfg)
    nb["assign_blocks"](sm, cfg)
    tr_sync = nb["simulate_memory_trace"](sm, cfg, plan_sync)
    tr_async = nb["simulate_memory_trace"](sm, cfg, plan_async)
    dense = nb["dense_reference_memory"](sm, cfg).total_mib
    floor = nb["fully_sparse_memory"](sm, cfg, 1 - cfg.target_sparsity).total_mib

    fig_schedule(nb, plan_sync, cfg, "acdc", "01a_schedule_acdc",
                 "AC/DC: synchronous decompression schedule")
    fig_schedule(nb, plan_async, cfg, "async_acdc", "01b_schedule_async_acdc",
                 "Async-AC/DC (proposed): staggered decompression schedule")
    fig_memory_analytical(tr_sync, tr_async, dense, floor)
    fig_pareto(agg)
    fig_accuracy(runs)
    fig_memory_measured(runs)
    fig_iso_accuracy(agg)
    fig_iso_winner(agg)
    frames = shock_table(nb, runs)
    fig_shock_events(frames)
    fig_shock_distribution(frames)
    peaks = fig_blocks(nb, cfg, sm, tr_sync.max(), floor)
    cost = fig_blocks_tradeoff(nb, cfg, sm)
    fig_projection(proj, "SGD-M", "08a_resnet50_projection_sgdm",
                   "ResNet-50 projection: SGD with momentum")
    fig_projection(proj, "Adam", "08b_resnet50_projection_adam", "ResNet-50 projection: Adam")
    fig_seed_dots(runs, agg)

    print("\ncross-checks against the 160-epoch sweep:")
    print("  per-block dense epochs  sync", plan_sync.dense_epochs_per_block().tolist(),
          " async", plan_async.dense_epochs_per_block().tolist())
    print(f"  analytical peaks        sync {tr_sync.max():.2f}  async {tr_async.max():.2f} MiB")
    print("  B-sweep peaks (MiB)    ", np.round(peaks, 2).tolist())
    for k, fr in frames.items():
        v = fr["shock_%"]
        print(f"  shocks {NAME[k]:<14} events {len(fr):2d}  mean {v.mean():6.2f}%  median {v.median():6.2f}%  "
              f"max {v.max():6.2f}%  positive {(v > 0).sum():2d}  positive sum per seed {v.clip(lower=0).sum() / len(SEEDS):6.1f}%")
    print("\n" + agg[["method", "top1_mean", "top1_std", "top5_mean", "peak_MiB", "mean_MiB",
                      "peak_vs_dense", "PFLOPs"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
