"""Re-analysis of the finished runs requested in the supervisors' review. No training.

Four analyses, all computed from results/*.json:

1. Paired seed-level comparisons at the reference schedule (B = 3). Every method uses the same
   seeds, and a seed fixes the initialization and the data order, so differences are taken
   within a seed rather than between independent groups.
2. The block-count sweep against AC/DC with paired two one-sided tests (TOST), Holm-corrected
   over the twelve comparisons, and the trend in B by repeated-measures ANOVA, per-seed slopes
   and a linear mixed model with a random intercept per seed.
3. A transition-disturbance metric anchored at the onset of each compression, so that every
   method is measured at the same steps relative to the event and the window covers the whole
   compression ramp of Async-AC/DC++ and one epoch of recovery.
4. The number of seeds a confirmatory paired TOST would need.

Usage:  python reanalysis_revision.py
Writes: results/revision_*.csv, figures/eps/06c_disturbance_aligned.eps and
        figures/eps/06d_disturbance_events.eps
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import make_thesis_figures as M  # noqa: E402  (house figure style)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EPOCHS = 160
SEEDS = (0, 1, 2)
MARGIN = 1.0            # equivalence margin in percentage points, fixed in the proposal
SPE = 390               # optimizer steps per epoch
RAMP = 130              # compression ramp of Async-AC/DC++ in steps (150 capped at a third of an epoch)
BASE_W = 50             # baseline window before the onset, in steps
RECOVERY = SPE          # steps after the block is fully sparse that the window still covers
NEXT_ONSET = 5 * SPE - RAMP   # the next staggered onset at B = 3: a window covering full recovery
SMOOTH = 25             # moving-average length for the peak excess


# ----------------------------------------------------------------------------- data
def load():
    """{(method, B, dense_len, seed): result} and a flat table of final accuracies."""
    runs, rows = {}, []
    for path in sorted(glob.glob(os.path.join("results", "*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        c = d["cfg"]
        if c["epochs"] != EPOCHS:
            continue
        key = (d["method_key"], c["n_blocks"], c["acdc_dense_len"], c["seed"])
        assert key not in runs, f"two results for {key}"
        runs[key] = d
        rows.append(dict(method=d["method_key"], B=c["n_blocks"], dl=c["acdc_dense_len"],
                         seed=c["seed"], top1=d["top1"]))
    return runs, pd.DataFrame(rows)


def acc(df, method, B=3, dl=5):
    """Final top-1 accuracy indexed by seed. Every baseline is stored with B = 3, dl = 5."""
    g = df[(df.method == method) & (df.B == B) & (df.dl == dl)]
    return g.set_index("seed").top1.sort_index()


# ----------------------------------------------------------------------------- statistics
def paired(a, b):
    """Paired t statistics and TOST on the seed-level differences a - b."""
    seeds = sorted(set(a.index) & set(b.index))
    d = (a.loc[seeds] - b.loc[seeds]).to_numpy()
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    t95, t90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    p_lower = stats.t.sf((mean + MARGIN) / se, n - 1)
    p_upper = stats.t.cdf((mean - MARGIN) / se, n - 1)
    return dict(n=n, diff=mean, sd_diff=sd,
                ci95_lo=mean - t95 * se, ci95_hi=mean + t95 * se,
                ci90_lo=mean - t90 * se, ci90_hi=mean + t90 * se,
                p_diff=2 * stats.t.sf(abs(mean / se), n - 1), p_tost=max(p_lower, p_upper))


def welch(a, b):
    """The unpaired analysis of the earlier draft, kept for comparison."""
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff = a.mean() - b.mean()
    p_tost = max(stats.t.sf((diff + MARGIN) / se, dof), stats.t.cdf((diff - MARGIN) / se, dof))
    return dict(welch_p=2 * stats.t.sf(abs(diff / se), dof), welch_tost=p_tost)


def holm(p):
    """Holm step-down adjustment."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main_comparisons(df):
    pairs = [("async_acdc", "acdc"), ("async_pp", "acdc"), ("async_pp", "async_acdc"),
             ("async_acdc", "set"), ("async_pp", "set"), ("gmp", "acdc"), ("dense", "acdc")]
    out = []
    for a, b in pairs:
        A, B_ = acc(df, a), acc(df, b)
        out.append(dict(comparison=f"{a} - {b}", **paired(A, B_), **welch(A, B_)))
    res = pd.DataFrame(out)
    # The two pre-specified primary comparisons form their own family of two.
    res["p_tost_holm_primary"] = np.nan
    res.loc[:1, "p_tost_holm_primary"] = holm(res.loc[:1, "p_tost"])
    return res


def block_sweep(df):
    base = acc(df, "acdc")
    out = []
    for m in ("async_acdc", "async_pp"):
        for B in range(3, 9):
            A = acc(df, m, B)
            out.append(dict(method=m, B=B, dl=5, mean=A.mean(), sd=A.std(ddof=1),
                            **paired(A, base), **welch(A, base)))
    res = pd.DataFrame(out)
    res["p_tost_holm"] = holm(res.p_tost)
    res["welch_tost_holm"] = holm(res.welch_tost)
    extra = []
    for m in ("async_acdc", "async_pp"):
        A = acc(df, m, 8, 2)
        extra.append(dict(method=m, B=8, dl=2, mean=A.mean(), sd=A.std(ddof=1),
                          **paired(A, base), **welch(A, base)))
    return pd.concat([res, pd.DataFrame(extra)], ignore_index=True)


def block_trend(df):
    """Trend in B, treating the seed as a repeated-measures unit."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import AnovaRM

    sw = df[df.method.isin(["async_acdc", "async_pp"]) & (df.dl == 5) & df.B.between(3, 8)].copy()
    sw["Bc"] = sw.B.astype(str)
    rows = []
    for m in ("async_acdc", "async_pp"):
        g = sw[sw.method == m]
        rm = AnovaRM(g, "top1", "seed", within=["Bc"]).fit().anova_table.loc["Bc"]
        # One slope per seed, then a t interval over seeds: a repeated-measures trend test.
        slopes = np.array([np.polyfit(gs.B, gs.top1, 1)[0] for _, gs in g.groupby("seed")])
        n = len(slopes)
        se = slopes.std(ddof=1) / np.sqrt(n)
        t = stats.t.ppf(0.975, n - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mm = smf.mixedlm("top1 ~ B", g, groups=g["seed"]).fit(reml=True)
        rows.append(dict(method=m, rm_F=rm["F Value"], rm_df1=rm["Num DF"], rm_df2=rm["Den DF"],
                         rm_p=rm["Pr > F"], slope=slopes.mean(), slope_ci_lo=slopes.mean() - t * se,
                         slope_ci_hi=slopes.mean() + t * se,
                         slope_p=2 * stats.t.sf(abs(slopes.mean() / se), n - 1),
                         mixed_slope=mm.params["B"], mixed_se=mm.bse["B"],
                         mixed_seed_var=float(mm.cov_re.iloc[0, 0]), mixed_resid_var=mm.scale))
    joint = AnovaRM(sw, "top1", "seed", within=["Bc", "method"]).fit().anova_table
    return pd.DataFrame(rows), joint


# ----------------------------------------------------------------------------- disturbance
def compression_epochs(d):
    """Epochs at which a compression takes full effect; a synchronous one counts once."""
    return sorted({e["epoch"] for e in d["events"] if e["kind"] == "compress" and e["epoch"] > 0})


def aligned_loss(L, t):
    """Loss from BASE_W steps before the onset to RECOVERY steps after t, relative to the
    baseline window, in percent. The onset is t - RAMP for every method: the start of the
    ramp for Async-AC/DC++, and RAMP steps of ordinary dense training for the others."""
    onset = t - RAMP
    base = L[onset - BASE_W:onset].mean()
    return 100 * (L[onset - BASE_W:t + RECOVERY] - base) / base


def disturbance(runs):
    rows, traces = [], {}
    for m in ("acdc", "async_acdc", "async_pp"):
        traces[m] = []
        for s in SEEDS:
            L = np.asarray(runs[(m, 3, 5, s)]["loss_curve"])
            for e in compression_epochs(runs[(m, 3, 5, s)]):
                t = e * SPE
                if t + RECOVERY > len(L):
                    continue
                rel = aligned_loss(L, t)
                win = rel[BASE_W:]                      # onset .. t + RECOVERY
                ramp, post = win[:RAMP], win[RAMP:]
                smooth = np.convolve(win, np.ones(SMOOTH) / SMOOTH, mode="valid")
                pre50 = L[t - 50:t].mean()
                # The one-epoch window ends before the staggered methods have recovered, so the
                # excess is also taken up to the next staggered onset, by which point it has.
                onset = t - RAMP
                base = L[onset - BASE_W:onset].mean()
                excess_next = 100 * (L[onset:t + NEXT_ONSET].mean() - base) / base
                end_next = 100 * (L[t + NEXT_ONSET - 50:t + NEXT_ONSET].mean() - base) / base
                rows.append(dict(method=m, seed=s, epoch=e,
                                 excess=win.mean(), excess_ramp=ramp.mean(), excess_post=post.mean(),
                                 peak=smooth.max(), excess_next=excess_next, end_next=end_next,
                                 shock50=100 * (L[t:t + 50].mean() - pre50) / pre50))
                traces[m].append(rel)
    return pd.DataFrame(rows), {k: np.array(v) for k, v in traces.items()}


def summarize_disturbance(sh):
    out = []
    for m in ("acdc", "async_acdc", "async_pp"):
        g = sh[sh.method == m]
        pos = g.excess.clip(lower=0).groupby(g.seed).sum().mean()
        out.append(dict(method=m, events=len(g), per_run=len(g) // len(SEEDS),
                        excess_mean=g.excess.mean(), excess_median=g.excess.median(),
                        excess_max=g.excess.max(), excess_ramp_mean=g.excess_ramp.mean(),
                        excess_post_mean=g.excess_post.mean(),
                        positive=int((g.excess > 0).sum()), positive_sum_per_run=pos,
                        peak_mean=g.peak.mean(), peak_median=g.peak.median(),
                        excess_next_mean=g.excess_next.mean(),
                        excess_next_median=g.excess_next.median(),
                        end_next_mean=g.end_next.mean(),
                        shock50_mean=g.shock50.mean(),
                        shock50_positive_sum_per_run=g.shock50.clip(lower=0).groupby(g.seed).sum().mean()))
    return pd.DataFrame(out)


def fig_aligned(traces):
    x = np.arange(-RAMP - BASE_W, RECOVERY)
    fig, ax = plt.subplots(figsize=(M.WIDE, 2.9))
    ax.yaxis.grid(True)
    ax.axvspan(-RAMP, 0, facecolor=M.BAND, edgecolor="none", zorder=0)
    ax.axhline(0, color=M.INK2, lw=0.6, zorder=1)
    ax.axvline(0, color=M.MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    k = np.ones(SMOOTH) / SMOOTH
    handles = []
    for m in ("acdc", "async_acdc", "async_pp"):
        med = np.median(traces[m], axis=0)
        sm = np.convolve(med, k, mode="same")
        sm[:SMOOTH // 2] = np.nan
        sm[-(SMOOTH // 2):] = np.nan
        ax.plot(x, sm, color=M.COL[m], lw=1.5, zorder=3)
        handles.append(M.line_key(m, marker=None))
    ax.text(-RAMP / 2, 0.98, "ramp of\nAsync-AC/DC++", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=M.FS_ANN, color=M.INK2)
    ax.text(6, 0.98, "block fully sparse", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=M.FS_ANN, color=M.INK2)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("Optimizer steps relative to the epoch boundary at which the block is fully sparse")
    ax.set_ylabel("Training loss vs. baseline (%)")
    handles.append(Patch(facecolor=M.BAND, label="Compression ramp"))
    M.legend_above(ax, handles, ncol=4)
    M.save(fig, "06c_disturbance_aligned")


def fig_events(sh):
    fig, ax = plt.subplots(figsize=(M.WIDE, 2.7))
    ax.yaxis.grid(True)
    ax.axhline(0, color=M.INK2, lw=0.6, zorder=1)
    dodge = {"acdc": -1.0, "async_acdc": 0.0, "async_pp": 1.0}
    for m in ("acdc", "async_acdc", "async_pp"):
        g = sh[sh.method == m].groupby("epoch")["excess"]
        ep = g.mean().index.to_numpy(dtype=float) + dodge[m]
        ax.vlines(ep, g.min().to_numpy(), g.max().to_numpy(), color=M.COL[m], lw=0.9, zorder=2)
        ax.scatter(ep, g.mean().to_numpy(), s=28, marker=M.MARK[m], color=M.COL[m],
                   edgecolors="white", linewidths=0.7, zorder=3)
    ax.set_xlim(0, 125)
    ax.set_xticks(range(0, 121, 10))
    ax.set_xlabel("Epoch at which the block is fully sparse")
    ax.set_ylabel("Mean excess loss (%)")
    M.legend_above(ax, [M.dot_key(M.COL[m], M.MARK[m], M.NAME[m])
                        for m in ("acdc", "async_acdc", "async_pp")], ncol=3)
    M.save(fig, "06d_disturbance_events")


# ----------------------------------------------------------------------------- power
def seeds_needed(sd_diff, true_diff, alpha=0.05, power=0.8, sims=40000, seed=0):
    """Smallest number of seeds at which a paired TOST reaches the requested power."""
    rng = np.random.default_rng(seed)
    for n in range(3, 61):
        x = rng.normal(true_diff, sd_diff, size=(sims, n))
        mean, se = x.mean(1), x.std(1, ddof=1) / np.sqrt(n)
        tc = stats.t.ppf(1 - alpha, n - 1)
        if (((mean + MARGIN) / se > tc) & ((mean - MARGIN) / se < -tc)).mean() >= power:
            return n
    return np.nan


def main():
    runs, df = load()
    print(f"{len(df)} runs at {EPOCHS} epochs\n")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    mc = main_comparisons(df)
    mc.to_csv(os.path.join("results", "revision_paired_main.csv"), index=False)
    print("Paired comparisons at B = 3\n", mc.round(3).to_string(index=False), "\n")

    bs = block_sweep(df)
    bs.to_csv(os.path.join("results", "revision_blocks_paired.csv"), index=False)
    print("Block sweep against AC/DC\n", bs.round(3).to_string(index=False), "\n")
    for B in range(3, 9):
        d = paired(acc(df, "async_pp", B), acc(df, "async_acdc", B))
        print(f"  Async-AC/DC++ - Async-AC/DC at B = {B}: {d['diff']:+.2f} "
              f"[{d['ci95_lo']:+.2f}, {d['ci95_hi']:+.2f}]  p = {d['p_diff']:.3f}")
    for m in ("async_acdc", "async_pp"):
        d = paired(acc(df, m, 8, 2), acc(df, m, 8, 5))
        print(f"  {m} B = 8, dense 2 vs 5 epochs: {d['diff']:+.2f} "
              f"[{d['ci95_lo']:+.2f}, {d['ci95_hi']:+.2f}]  p = {d['p_diff']:.3f}")

    trend, joint = block_trend(df)
    trend.to_csv(os.path.join("results", "revision_blocks_trend.csv"), index=False)
    print("\nTrend in B (repeated measures)\n", trend.round(4).to_string(index=False))
    print(joint, "\n")

    sh, traces = disturbance(runs)
    sh.to_csv(os.path.join("results", "revision_disturbance_events.csv"), index=False)
    summ = summarize_disturbance(sh)
    summ.to_csv(os.path.join("results", "revision_disturbance_summary.csv"), index=False)
    print("Disturbance anchored at the compression onset\n", summ.round(2).to_string(index=False))
    common = sorted(set.intersection(*[set(sh[sh.method == m].epoch) for m in traces]))
    print("  events common to all three methods:", common)
    print(sh[sh.epoch.isin(common)].groupby("method")[["excess", "peak", "shock50"]].mean().round(2))
    per_run = sh.groupby(["method", "seed"])[["excess", "peak", "excess_next"]].mean()
    for col in ("excess", "peak", "excess_next"):
        pr = per_run[col].unstack(0)
        for a, b in (("async_pp", "async_acdc"), ("async_pp", "acdc"), ("async_acdc", "acdc")):
            d = pr[a] - pr[b]
            print(f"  {col:<12} {a} - {b}: {d.mean():+.2f} (per seed {np.round(d.to_numpy(), 2)}) "
                  f"paired p = {stats.ttest_rel(pr[a], pr[b]).pvalue:.3f}")
    fig_aligned(traces)
    fig_events(sh)

    pooled = float(np.sqrt(np.mean(bs[bs.dl == 5].sd_diff ** 2)))
    rows = []
    for sd in (0.2, round(pooled, 2), 0.45, 0.6):
        for true_diff in (0.0, -0.3, -0.5):
            for alpha in (0.05, 0.025):
                rows.append(dict(sd_diff=sd, true_diff=true_diff, alpha=alpha,
                                 seeds=seeds_needed(sd, true_diff, alpha)))
    pw = pd.DataFrame(rows)
    pw.to_csv(os.path.join("results", "revision_power.csv"), index=False)
    print(f"\npooled SD of paired differences: {pooled:.3f}")
    print(pw.pivot_table(index=["sd_diff", "alpha"], columns="true_diff", values="seeds").to_string())


if __name__ == "__main__":
    main()
