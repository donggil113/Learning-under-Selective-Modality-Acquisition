"""Static figures for the report (reads results/summary/summary.json and runs)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .summarize import load_runs

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results/figures"
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "legend.frameon": False})

LABELS = {"full": "complete cases, no masking", "drop50": "dropout p=0.5", "dams_rate": "dropout at target rates (DAMS Alg. 1)",
          "freq": "target pattern frequencies", "freq_prior": "+ class balance (EM)", "freq_prior_plugin": "+ class balance (plug-in)",
          "freq_prior_oracle": "+ class balance (true prevalence)", "sel": "+ selection on base covariates (IW)",
          "pat": "per-pattern covariate IW (Γ=1)", "dm": "outcome-model plug-in (Γ=1)", "dr": "doubly robust (Γ=1)",
          "lab50": "50 target labels", "lab100": "100 target labels", "lab200": "200 target labels", "lab400": "400 target labels"}


def fig_tau(loss="brier"):
    order = list(LABELS)
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for k, (design, col, name) in enumerate((("cc", BLUE, "challenge sets, 6 orderings"),
                                             ("rs", "#1baf7a", "random re-partitions"),
                                             ("icu", ORANGE, "surgical complete cases → medical ICUs"))):
        runs = load_runs(design)
        if not runs:
            continue
        ys, m, s = [], [], []
        for i, e in enumerate(order):
            v = [r[loss]["agreement"][e]["kendall_tau"] for r in runs if e in r[loss]["agreement"]]
            if v:
                ys.append(i + (k - 1) * 0.25)
                m.append(np.mean(v))
                s.append(np.std(v))
        ax.errorbar(m, ys, xerr=s, fmt="o", color=col, ms=6, lw=1.5, capsize=0, label=f"{name} ({len(runs)} runs)",
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[e] for e in order])
    ax.invert_yaxis()
    ax.set_xlabel(f"Kendall τ vs the natural-missingness ranking ({loss}; mean ± sd over runs)")
    ax.set_xlim(-0.1, 1.0)
    ax.axvline(0, color=TEXT2, lw=0.8)
    ax.set_title("Which evaluation reproduces the deployment ranking\nof 29 candidate models?", loc="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
    fig.savefig(FIG / f"kendall_tau_{loss}.png", dpi=160)
    plt.close(fig)


def fig_or(summary):
    rows = summary["mnar_pooled"]
    rows = sorted(rows, key=lambda r: (r["n_missing_panels"], r["pattern(ABG,ALINE,LACT,LIVER)"]))
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    for i, r in enumerate(rows):
        b, se = r["log_or_ens"], r["se_ens"]
        lo, hi = np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
        ax.plot([lo, hi], [i, i], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot(np.exp(b), i, "o", color=BLUE, ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axvline(1, color=TEXT2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2])
    ax.set_xticklabels(["0.25", "0.5", "1", "2"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['pattern(ABG,ALINE,LACT,LIVER)']}  (n={r['n']:,})" for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("odds ratio of death vs complete-case model\ngiven the recorded features (95% CI, log scale)")
    ax.set_title("Death odds vs complete-case (LR+HGB) model, per recording pattern\n"
                 "bits = ABG, A-line, lactate, liver; 1111 = negative control", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "label_dependence_or.png", dpi=160)
    plt.close(fig)


def fig_aa(loss="brier"):
    runs = load_runs("cc")
    if not runs:
        return
    keys = list(runs[0][loss]["aa"])
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharex=True)
    metrics = (("decided_frac", "pairs decided (bound excludes 0)"), ("decided_correct", "correct among decided"),
               ("covers_truth", "bound covers the true Δ"))
    for d, col, name in (("two", BLUE, "two-sided Γ"), ("down", ORANGE, "one-sided Γ (missing ⇒ not riskier)")):
        ks = [k for k in keys if k.startswith(d + ":")]
        g = [float(k.split(":")[1]) for k in ks]
        for ax, (mkey, title) in zip(axes, metrics):
            v = [np.nanmean([r[loss]["aa"][k][mkey] for r in runs]) for k in ks]
            ax.plot(g, v, "-o", color=col, lw=2, ms=5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=name)
            ax.set_title(title, fontsize=9.5, loc="left")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Γ")
    # naive dropout with bootstrap CIs, for reference
    dec = np.mean([np.mean(np.array(r[loss]["pairs"]["drop50"]["sig_sign"]) != 0) for r in runs])
    cor = np.mean([np.mean((lambda es, ts: es[es != 0] == ts[es != 0])(np.array(r[loss]["pairs"]["drop50"]["sig_sign"]),
                                                                       np.sign(np.array(r[loss]["pairs"]["truth_delta"])))) for r in runs])
    axes[0].axhline(dec, color=TEXT2, lw=1)
    axes[0].text(4.0, dec + 0.02, "dropout p=0.5, 95% CI", ha="right", fontsize=8, color=TEXT2)
    axes[1].axhline(cor, color=TEXT2, lw=1)
    axes[1].text(4.0, 0.08, f"grey line: dropout p=0.5 with\nbootstrap 95% CI ({cor:.2f})", ha="right", fontsize=8, color=TEXT2)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Acquisition-aware bounds on pairwise risk differences ({loss}, complete-case design, {len(runs)} runs)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / f"aa_bounds_{loss}.png", dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT / "results/summary/summary.json").read_text())
    for loss in ("brier", "logloss"):
        fig_tau(loss)
        fig_aa(loss)
    fig_or(summary)


if __name__ == "__main__":
    main()
