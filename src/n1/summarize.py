"""Aggregate every run into tables (markdown + JSON) under results/summary/."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from .evaluators import Evaluation, loss01
from .features_c2012 import build
from .models import Prep, pattern_id, patterns
from .run_c2012 import make_spec

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/summary"
EVAL_ORDER = ["full", "drop50", "dams_rate", "freq", "freq_prior", "freq_prior_plugin", "freq_prior_oracle",
              "sel", "pat", "dm", "dr", "lab50", "lab100", "lab200", "lab400"]
LADDER = ["full", "drop50", "dams_rate", "freq", "freq_prior_plugin", "freq_prior_oracle", "sel", "pat", "dr",
          "gamma1_limit"]


def load_runs(design):
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(ROOT / f"results/c2012/run_{design}_*.json")))]


def ms(x):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.3f} ± {np.nanstd(x):.3f}"


def agreement_table(runs, loss):
    rows = []
    for e in EVAL_ORDER:
        vals = [r[loss]["agreement"][e] for r in runs if e in r[loss]["agreement"]]
        if not vals:
            continue
        rows.append({"evaluator": e, "kendall_tau": ms([v["kendall_tau"] for v in vals]),
                     "spearman": ms([v["spearman"] for v in vals]),
                     "top1_regret(x1e3)": ms([1e3 * v["top1_regret"] for v in vals]),
                     "pair_sign_agreement": ms([v["pair_sign_agreement"] for v in vals]),
                     "_tau": float(np.mean([v["kendall_tau"] for v in vals]))})
    return pd.DataFrame(rows)


def flips_table(runs, loss):
    rows = []
    for e in ["drop50", "dams_rate", "freq", "freq_prior_oracle", "sel", "pat", "dr"]:
        P = [r[loss]["pairs"][e] for r in runs]
        dec, cor = [], []
        for r in runs:
            es = np.array(r[loss]["pairs"][e]["sig_sign"])
            ts = np.sign(np.array(r[loss]["pairs"]["truth_delta"]))
            dec.append(np.mean(es != 0))
            cor.append(np.mean(es[es != 0] == ts[es != 0]) if (es != 0).any() else np.nan)
        rows.append({"evaluator": e,
                     "decided_frac(CI excl. 0)": ms(dec), "decided_correct(vs true sign)": ms(cor),
                     "truth_sig_pairs": ms([p["truth_sig_pairs"] for p in P]),
                     "eval_sig_pairs": ms([p["eval_sig_pairs"] for p in P]),
                     "sig_agree": ms([p["sig_agree"] for p in P]),
                     "sig_flips": ms([p["sig_flips"] for p in P]),
                     "runs_with_any_sig_flip": f"{sum(p['sig_flips'] > 0 for p in P)}/{len(P)}",
                     "CI_covers_true_delta": ms([p["ci_covers_truth"] for p in P])})
    return pd.DataFrame(rows)


def flip_explanation(runs, loss, base="freq"):
    """For pairs that `base` gets significantly wrong, which adjustment repairs the sign?"""
    tot = 0
    fixed = {k: 0 for k in ["freq_prior_oracle", "sel", "pat", "dr"]}
    for r in runs:
        P = r[loss]["pairs"]
        ts = np.array(P["truth_sig_sign"])
        bs = np.array(P[base]["sig_sign"])
        flip = (bs * ts) < 0
        tot += int(flip.sum())
        for k in fixed:
            d = np.sign(np.array(P[k]["delta"]))
            fixed[k] += int(np.sum(flip & (d == ts)))
    return {"base": base, "sig_flips_total": tot, **{f"repaired_by_{k}": v for k, v in fixed.items()}}


def ladder_table(runs, loss):
    rows = []
    for e in LADDER:
        vals = [r["ladder_T"][loss][e]["agreement"] for r in runs]
        rows.append({"step": e, "kendall_tau": ms([v["kendall_tau"] for v in vals]),
                     "pair_sign_agreement": ms([v["pair_sign_agreement"] for v in vals]),
                     "top1_regret(x1e3)": ms([1e3 * v["top1_regret"] for v in vals])})
    return pd.DataFrame(rows)


def strategy_ranks(runs, loss):
    names = runs[0]["names"]
    ranks = {e: [] for e in ["truth", "full", "drop50", "freq", "sel", "dr"]}
    for r in runs:
        for e in ranks:
            v = np.array(r[loss]["risks"][e])
            ranks[e].append(pd.Series(v).rank().values)
    df = pd.DataFrame({e: np.mean(v, 0) for e, v in ranks.items()}, index=names)
    df["truth_risk"] = np.mean([r[loss]["risks"]["truth"] for r in runs], 0)
    return df.sort_values("truth")


def aa_table(runs, loss):
    rows = []
    for key in runs[0][loss]["aa"]:
        A = [r[loss]["aa"][key] for r in runs]
        d, g = key.split(":")
        rows.append({"direction": d, "Gamma": float(g), "decided_frac": ms([a["decided_frac"] for a in A]),
                     "decided_correct": ms([a["decided_correct"] for a in A]),
                     "decided_wrong_where_truth_sig": ms([a["decided_wrong_sig"] for a in A]),
                     "covers_true_delta": ms([a["covers_truth"] for a in A]),
                     "robust_best_size": ms([len(a["robust_best"]) for a in A]),
                     "true_best_in_robust_set": f"{sum(r[loss]['true_best'] in r[loss]['aa'][key]['robust_best'] for r in runs)}/{len(runs)}"})
    return pd.DataFrame(rows)


def gamma_transfer(runs):
    """Per-pattern log-OR estimated on the labeled DEVELOPMENT set (D) vs on the
    deployment truth set (T): can Gamma be calibrated without target labels?"""
    rows = []
    for r in runs:
        d = {x["pattern"]: x for x in r["mnar_dev"]}
        t = {x["pattern"]: x for x in r["mnar"]}
        for k in set(d) & set(t):
            if k == "1111":
                continue
            rows.append({"run": r["order"] + str(r["seed"]), "pattern": k, "logor_dev": d[k]["log_or"],
                         "logor_T": t[k]["log_or"], "n_T": t[k]["n"]})
    df = pd.DataFrame(rows)
    out = {"pearson_r": float(df[["logor_dev", "logor_T"]].corr().iloc[0, 1]),
           "mean_abs_diff": float((df.logor_dev - df.logor_T).abs().mean()),
           "frac_T_within_dev_gamma": float(np.mean(np.abs(df.logor_T) <= np.abs(df.logor_dev).groupby(df.run).transform("max"))),
           "n_pairs": int(len(df))}
    return out


def saved_eval(npz):
    z = np.load(npz, allow_pickle=True)
    ev = Evaluation.__new__(Evaluation)
    for k in z.files:
        setattr(ev, k, z[k])
    ev.K = int(np.log2(ev.PV.shape[1]))
    ev.P = patterns(ev.K)
    ev.nP = len(ev.P)
    ev.full_id = ev.nP - 1
    ev.names = [str(n) for n in ev.names]
    return ev, bool(z["complete_exact"])


def aa_per_panel(design, loss, lams=(1.0, 1.1, 1.2, 1.3, 1.5, 2.0)):
    rows = []
    for f in sorted(glob.glob(str(ROOT / f"results/c2012/arrays_{design}_*.npz"))):
        ev, exact = saved_eval(f)
        tru = (np.where(ev.yT[None, :] == 1, *loss01(ev.PT, loss)[::-1])).mean(1)
        J = len(tru)
        iu = np.triu_indices(J, 1)
        dt = (tru[:, None] - tru[None, :])[iu]
        for direction in (None, "down"):
            for lam in lams:
                g = ev.gamma_per_panel(lam)
                b = np.array([ev.aa_pair(a, c, loss, g, direction, exact) for a, c in zip(*iu)])
                lo, up = b[:, 0], b[:, 2]
                dec = (lo > 0) | (up < 0)
                sgn = np.where(lo > 0, 1, np.where(up < 0, -1, 0))
                rows.append({"run": Path(f).stem, "direction": direction or "two", "lambda": lam,
                             "decided_frac": dec.mean(),
                             "decided_correct": np.mean(np.sign(dt[dec]) == sgn[dec]) if dec.any() else np.nan,
                             "covers": np.mean((lo <= dt) & (dt <= up))})
    df = pd.DataFrame(rows)
    return df.groupby(["direction", "lambda"])[["decided_frac", "decided_correct", "covers"]].agg(["mean", "std"]).round(3)


def label_part_analysis(design, loss, n_boot=300, seed=0):
    """Pairwise label-dependent part  D_ab = mean_T 1[group] (y - p_S)(g_a - g_b)
    with g = l1 - l0.  If Y is independent of M given X_o(m) and p_S is calibrated, E[D_ab] = 0.
    group = incomplete units (the MNAR term) and, as a negative control for the
    outcome model, complete units (exchangeable with V in the complete-case design)."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in sorted(glob.glob(str(ROOT / f"results/c2012/arrays_{design}_*.npz"))):
        ev, exact = saved_eval(f)
        l0, l1 = loss01(ev.PT, loss)
        g = (l1 - l0)                                     # J x nT
        res = (ev.yT - ev.pS_T)                           # nT
        tru_units = np.where(ev.yT[None, :] == 1, l1, l0)
        J, nT = g.shape
        iu = np.triu_indices(J, 1)
        inc = ev.pidT != ev.full_id
        B = rng.integers(0, nT, (n_boot, nT))
        for grp, mask in (("incomplete", inc), ("complete(control)", ~inc)):
            c = g * (res * mask)[None, :]                 # J x nT
            D = (c.mean(1)[:, None] - c.mean(1)[None, :])[iu]
            Db = np.stack([(lambda m: (m[:, None] - m[None, :])[iu])(c[:, b].mean(1)) for b in B])
            lo, hi = np.percentile(Db, [2.5, 97.5], axis=0)
            tr = tru_units.mean(1)
            dt = (tr[:, None] - tr[None, :])[iu]
            trb = np.stack([(lambda m: (m[:, None] - m[None, :])[iu])(tru_units[:, b].mean(1)) for b in B])
            tlo, thi = np.percentile(trb, [2.5, 97.5], axis=0)
            tsig = (tlo > 0) | (thi < 0)
            rows.append({"run": Path(f).stem, "group": grp,
                         "pairs_label_part_CI_excl_0": float(np.mean((lo > 0) | (hi < 0))),
                         "truth_sig_pairs_where_|label_part|>|true_delta|": float(np.mean(np.abs(D[tsig]) > np.abs(dt[tsig]))) if tsig.any() else np.nan,
                         "median_|label_part|/|true_delta|": float(np.median(np.abs(D) / np.maximum(np.abs(dt), 1e-12)))})
    df = pd.DataFrame(rows)
    return df.groupby("group")[[c for c in df.columns if c not in ("run", "group")]].agg(["mean", "std"]).round(3)


def mnar_pooled():
    """Per pattern: odds ratio of death vs the complete-case model given x_o(m),
    cross-fitted by challenge set (outcome models from the complete cases of the
    other two sets), pooled over all 12,000 stays.  Uses labels: evaluation only."""
    coh = build()
    spec = make_spec(coh)
    P = patterns(spec.K)
    pid = pattern_id(coh.M)
    off = np.full(len(pid), np.nan)
    for s in "abc":
        tr = (coh.set != s) & coh.M.all(1)
        te = coh.set == s
        prep = Prep(spec).fit(coh.X[tr])
        for m in range(len(P)):
            sel = te & (pid == m)
            if not sel.any():
                continue
            om = LogisticRegression(C=1.0, max_iter=5000).fit(prep.sub(coh.X[tr], P[m]), coh.y[tr])
            p = np.clip(om.predict_proba(prep.sub(coh.X[sel], P[m]))[:, 1], 1e-6, 1 - 1e-6)
            off[sel] = np.log(p / (1 - p))
    rows = []
    for m in range(len(P)):
        sel = pid == m
        y = coh.y[sel]
        fit = sm.GLM(y, np.ones((sel.sum(), 1)), family=sm.families.Binomial(), offset=off[sel]).fit()
        b, se = float(fit.params[0]), float(fit.bse[0])
        rows.append({"pattern(ABG,ALINE,LACT,LIVER)": "".join(str(int(v)) for v in P[m]),
                     "n_missing_panels": int(spec.K - P[m].sum()), "n": int(sel.sum()), "deaths": int(y.sum()),
                     "observed_mortality": round(float(y.mean()), 3),
                     "complete_case_model_mortality": round(float(np.mean(1 / (1 + np.exp(-off[sel])))), 3),
                     "odds_ratio": round(float(np.exp(b)), 3), "or_95ci": f"[{np.exp(b - 1.96 * se):.2f}, {np.exp(b + 1.96 * se):.2f}]",
                     "implied_Gamma": round(float(np.exp(abs(b))), 2)})
    df = pd.DataFrame(rows).sort_values(["n_missing_panels", "pattern(ABG,ALINE,LACT,LIVER)"])
    # per-panel log-OR slope: log OR ~ lambda * n_missing (inverse-variance weighted, incomplete patterns)
    return df


def semisynth():
    files = sorted(glob.glob(str(ROOT / "results/semisynth/semi_*.json")))
    if not files:
        return None, None
    R = [json.loads(Path(f).read_text()) for f in files]
    rows, aa = [], []
    for loss in ("brier", "logloss"):
        for mech in dict.fromkeys(r["mech"] for r in R):
            rr = [r for r in R if r["mech"] == mech]
            for e in rr[0][loss]["agreement"]:
                rows.append({"loss": loss, "mechanism": mech, "evaluator": e,
                             "kendall_tau": ms([r[loss]["agreement"][e]["kendall_tau"] for r in rr]),
                             "top1_regret(x1e3)": ms([1e3 * r[loss]["agreement"][e]["top1_regret"] for r in rr]),
                             "mean_abs_risk_error(x1e3)": ms([1e3 * r[loss]["bias_mean_abs"][e] for r in rr]),
                             "mean_abs_pair_error(x1e3)": ms([1e3 * r[loss]["pair_abs_err"][e] for r in rr])})
            for key in rr[0][loss]["aa"]:
                d, lam = key.split(":")
                A = [r[loss]["aa"][key] for r in rr]
                aa.append({"loss": loss, "mechanism": mech, "lambda_true": rr[0]["lambda_true"], "direction": d,
                           "lambda": float(lam), "covers": ms([a["covers"] for a in A]),
                           "decided_frac": ms([a["decided_frac"] for a in A]),
                           "decided_correct": ms([a["decided_correct"] for a in A])})
    return pd.DataFrame(rows), pd.DataFrame(aa)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# N1 results summary (generated by `python -m src.n1.summarize`)\n"]
    js = {}
    for design in ("cc", "icu"):
        runs = load_runs(design)
        if not runs:
            continue
        md.append(f"\n## Challenge 2012, design `{design}` ({len(runs)} runs)\n")
        md.append("Roles/sizes (first run): " + json.dumps(runs[0]["n"]) + "; prevalence: " +
                  json.dumps({k: round(v, 3) for k, v in runs[0]["prevalence"].items()}) +
                  f"; EM prior estimates: {ms([r['em_prior'] for r in runs])}\n")
        js[design] = {}
        for loss in ("brier", "logloss"):
            t = agreement_table(runs, loss)
            md.append(f"\n### {loss}: agreement with the natural-missingness truth\n\n" + t.drop(columns="_tau").to_markdown(index=False) + "\n")
            f = flips_table(runs, loss)
            md.append(f"\n### {loss}: bootstrap-significant pairwise disagreements (of {len(runs[0]['names']) * (len(runs[0]['names']) - 1) // 2} pairs)\n\n" + f.to_markdown(index=False) + "\n")
            fx = {b: flip_explanation(runs, loss, b) for b in ("drop50", "freq")}
            md.append(f"\n### {loss}: which adjustment repairs the significant flips\n\n" + pd.DataFrame(fx.values()).to_markdown(index=False) + "\n")
            L = ladder_table(runs, loss)
            md.append(f"\n### {loss}: within-truth-set ladder (V = complete cases of T, U = T)\n\n" + L.to_markdown(index=False) + "\n")
            S = strategy_ranks(runs, loss)
            md.append(f"\n### {loss}: mean rank of each candidate (1 = best)\n\n" + S.round(4).to_markdown() + "\n")
            A = aa_table(runs, loss)
            md.append(f"\n### {loss}: acquisition-aware bounds, scalar Gamma\n\n" + A.to_markdown(index=False) + "\n")
            try:
                LP = label_part_analysis(design, loss)
                md.append(f"\n### {loss}: label-dependent part of pairwise risk differences (truth set)\n\n" + LP.to_markdown() + "\n")
            except Exception as ex:
                md.append(f"\n(label-part analysis unavailable: {ex})\n")
            try:
                PP = aa_per_panel(design, loss)
                md.append(f"\n### {loss}: acquisition-aware bounds, per-panel Gamma = lambda^(#missing panels)\n\n" + PP.to_markdown() + "\n")
            except Exception as ex:  # arrays missing
                md.append(f"\n(per-panel bounds unavailable: {ex})\n")
            if loss == "brier":
                gt = gamma_transfer(runs)
                md.append("\n### Gamma calibration: per-pattern log-OR on labeled development data vs on deployment truth\n\n" + json.dumps(gt) + "\n")
                js[design]["gamma_transfer"] = gt
            js[design][loss] = {"agreement": t.to_dict("records"), "flips": f.to_dict("records"), "flip_explanation": fx,
                                "ladder": L.to_dict("records"), "aa": A.to_dict("records")}
    mn = mnar_pooled()
    md.append("\n## Label dependence of recording (pooled, cross-fitted; truth labels used for evaluation only)\n\n" + mn.to_markdown(index=False) + "\n")
    js["mnar_pooled"] = mn.to_dict("records")
    ss, ssa = semisynth()
    if ss is not None:
        md.append("\n## Semi-synthetic (known mechanism)\n\n" + ss.to_markdown(index=False) + "\n")
        md.append("\n### Semi-synthetic acquisition-aware bounds (per-panel lambda)\n\n" + ssa.to_markdown(index=False) + "\n")
        js["semisynth"] = ss.to_dict("records")
        js["semisynth_aa"] = ssa.to_dict("records")
    audit = ROOT / "results/audit/mimic_iv_demo_A_vs_M.json"
    if audit.exists():
        js["mimic_demo_audit"] = json.loads(audit.read_text())
    (OUT / "SUMMARY.md").write_text("\n".join(md))
    (OUT / "summary.json").write_text(json.dumps(js, indent=1, default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
