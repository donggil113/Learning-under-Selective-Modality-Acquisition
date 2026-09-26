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
    runs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(ROOT / f"results/c2012/run_{design}_*.json")))]
    return [r for r in runs if r.get("design") == design]


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
                     "sig_flips/run": ms([p["sig_flips"] for p in P]),
                     "runs_with_any_sig_flip": f"{sum(p['sig_flips'] > 0 for p in P)}/{len(P)}",
                     "CI_covers_true_delta": ms([p["ci_covers_truth"] for p in P]),
                     "expected_cover_if_unbiased": ms([p.get("expected_cover_if_unbiased", np.nan) for p in P])})
    return pd.DataFrame(rows)

def flip_explanation(runs, loss, base="drop50"):
    """Symmetric accounting on truth-significant pairs, by point sign: pairs the
    base evaluator gets wrong that the adjustment gets right (repaired) AND pairs
    the base gets right that the adjustment gets wrong (broken); plus each
    evaluator's own bootstrap-significant flips."""
    out = []
    for k in ["freq", "freq_prior_oracle", "sel", "pat", "dr"]:
        rep = brk = 0
        base_flips = adj_flips = 0
        for r in runs:
            P = r[loss]["pairs"]
            ts = np.array(P["truth_sig_sign"])
            m = ts != 0
            b = np.sign(np.array(P[base]["delta"]))
            a = np.sign(np.array(P[k]["delta"]))
            rep += int(np.sum(m & (b != ts) & (a == ts)))
            brk += int(np.sum(m & (b == ts) & (a != ts)))
            base_flips += P[base]["sig_flips"]
            adj_flips += P[k]["sig_flips"]
        out.append({"base": base, "adjustment": k, "repaired(point sign)": rep, "broken(point sign)": brk,
                    "net": rep - brk, "base_sig_flips_total": base_flips, "adjustment_sig_flips_total": adj_flips})
    return out

def ladder_table(runs, loss):
    rows = []
    for e in LADDER + ["gamma1_limit_lr", "gamma1_limit_hgb"]:
        vals = [r["ladder_T"][loss][e]["agreement"] for r in runs if e in r["ladder_T"][loss]]
        if not vals:
            continue
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
    deployment truth set (T), one run per distinct role assignment."""
    rows = []
    for r in runs:
        d = {x["pattern"]: x for x in r["mnar_dev"]}
        t = {x["pattern"]: x for x in r["mnar"]}
        for k in set(d) & set(t):
            if k == "1111":
                continue
            rows.append({"run": r["order"] + str(r["seed"]), "pattern": k, "logor_dev": d[k]["log_or"],
                         "logor_T": t[k]["log_or"], "n_T": t[k]["n"],
                         "same_direction": np.sign(d[k]["log_or"]) == np.sign(t[k]["log_or"])})
    df = pd.DataFrame(rows)
    return {"pearson_r": float(df[["logor_dev", "logor_T"]].corr().iloc[0, 1]),
            "mean_abs_diff": float((df.logor_dev - df.logor_T).abs().mean()),
            "frac_same_direction": float(df.same_direction.mean()), "n_pattern_runs": int(len(df))}

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


def label_part_analysis(design, loss, n_boot=300, seed=0, family=None):
    """Pairwise contribution of the non-identified term to the true risk
    difference, D_ab = (1/n_T) sum_{i in group} (y_i - p_S,i)(g_a,i - g_b,i),
    g = l1 - l0.  Reported: share of pairs whose D has a CI excluding 0, and the
    share of truth-significant pairs whose SIGN would change if D were removed
    (sign(Delta - D) != sign(Delta)).  Groups: incomplete units (the MNAR-given-
    recorded-features term) and complete units as a negative control, size-matched
    by rescaling the control to the incomplete group's size.  ``family``: outcome
    model used for p_S ('lr', 'hgb' or None = ensemble)."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in sorted(glob.glob(str(ROOT / f"results/c2012/arrays_{design}_*.npz"))):
        ev, exact = saved_eval(f)
        pS_T = ev.pS_T if family is None else getattr(ev, f"pS_T_{family}")
        l0, l1 = loss01(ev.PT, loss)
        g = (l1 - l0)
        res = (ev.yT - pS_T)
        tru_units = np.where(ev.yT[None, :] == 1, l1, l0)
        J, nT = g.shape
        iu = np.triu_indices(J, 1)
        inc = ev.pidT != ev.full_id
        B = rng.integers(0, nT, (n_boot, nT))
        tr = tru_units.mean(1)
        dt = (tr[:, None] - tr[None, :])[iu]
        trb = np.stack([(lambda m: (m[:, None] - m[None, :])[iu])(tru_units[:, b].mean(1)) for b in B])
        tlo, thi = np.percentile(trb, [2.5, 97.5], axis=0)
        tsig = (tlo > 0) | (thi < 0)
        for grp, mask in (("incomplete", inc), ("complete (control, size-matched)", ~inc)):
            scale = inc.sum() / max(mask.sum(), 1) if grp.startswith("complete") else 1.0
            c = g * (res * mask)[None, :] * scale
            D = (c.mean(1)[:, None] - c.mean(1)[None, :])[iu]
            Db = np.stack([(lambda m: (m[:, None] - m[None, :])[iu])(c[:, b].mean(1)) for b in B])
            lo, hi = np.percentile(Db, [2.5, 97.5], axis=0)
            rows.append({"run": Path(f).stem, "group": grp,
                         "pairs_with_CI_excl_0": float(np.mean((lo > 0) | (hi < 0))),
                         "truth_sig_pairs_sign_would_change": float(np.mean(np.sign(dt[tsig] - D[tsig]) != np.sign(dt[tsig]))) if tsig.any() else np.nan})
    df = pd.DataFrame(rows)
    return df.groupby("group")[[c for c in df.columns if c not in ("run", "group")]].agg(["mean", "std"]).round(3)

def aa_bootstrap(design, loss, settings=(("scalar", 1.0), ("scalar", 1.5), ("scalar", 2.0), ("scalar", 3.0),
                                         ("panel", 1.2), ("panel", 1.3), ("panel", 1.5)),
                 directions=(None, "down"), n_boot=200, seed=0):
    """Acquisition-aware selection WITH sampling uncertainty: a pair is decided only
    when the bootstrap 95% CI of the identified interval [L(Gamma), U(Gamma)]
    excludes 0 (evaluators.aa_bootstrap_ci).  Same information as the point
    evaluators: V labels + unlabeled U."""
    from .evaluators import aa_bootstrap_ci, unit_loss
    rng = np.random.default_rng(seed)
    rows = []
    for f in sorted(glob.glob(str(ROOT / f"results/c2012/arrays_{design}_*.npz"))):
        ev, exact = saved_eval(f)
        tru = unit_loss(ev.PT.astype(float), ev.yT[None, :], loss).mean(1)
        J = len(tru)
        iu = np.triu_indices(J, 1)
        dt = (tru[:, None] - tru[None, :])[iu]
        nmis = ev.K - ev.P[ev.pidU].sum(1)
        for kind, val in settings:
            gam = (val ** nmis) if kind == "panel" else np.full(len(nmis), val)
            for direction in directions:
                cl, cu, _ = aa_bootstrap_ci(ev, loss, gam, direction, exact, n_boot, rng)
                dec = (cl > 0) | (cu < 0)
                sgn = np.where(cl > 0, 1, -1)
                rows.append({"run": Path(f).stem, "gamma": f"{kind}:{val}", "direction": direction or "two",
                             "decided_frac": dec.mean(),
                             "decided_correct": np.mean(sgn[dec] == np.sign(dt[dec])) if dec.any() else np.nan,
                             "ci_covers_true_delta": np.mean((cl <= dt) & (dt <= cu))})
    df = pd.DataFrame(rows)
    return df.groupby(["direction", "gamma"])[["decided_frac", "decided_correct", "ci_covers_true_delta"]].agg(["mean", "std"]).round(3)


def dev_lambda(mnar_rows, conservative=False, min_n=100):
    """Per-panel sensitivity from DEVELOPMENT labels only.
    default: inverse-variance weighted slope (through the origin) of per-pattern
      log-OR on the number of unrecorded panels -- a pattern-AVERAGE, hence only a
      lower bound on the per-unit Gamma the bounds assume.
    conservative: max over patterns with n >= min_n of the per-panel upper 95%
      limit, exp((|log OR| + 1.96 se) / k).
    Returns (lambda, direction of the average slope)."""
    x, y, w, ub = [], [], [], []
    for r in mnar_rows:
        k = 4 - r["pattern"].count("1")
        if k == 0 or not np.isfinite(r.get("se", np.nan)) or r["se"] <= 0:
            continue
        x.append(k); y.append(r["log_or"]); w.append(1 / r["se"] ** 2)
        if r["n"] >= min_n:
            ub.append((abs(r["log_or"]) + 1.96 * r["se"]) / k)
    x, y, w = map(np.asarray, (x, y, w))
    b = float(np.sum(w * x * y) / np.sum(w * x * x))
    lam = float(np.exp(max(ub))) if conservative else float(np.exp(abs(b)))
    return lam, ("down" if b < 0 else "up")

def aa_dev_calibrated(design, loss, n_boot=200, seed=1):
    """Acquisition-aware selection with lambda AND direction chosen per run from
    the labeled development set D only (no deployment labels)."""
    from .evaluators import aa_bootstrap_ci, unit_loss
    rng = np.random.default_rng(seed)
    rows = []
    for f in sorted(glob.glob(str(ROOT / f"results/c2012/arrays_{design}_*.npz"))):
        run = json.loads(Path(f.replace("arrays_", "run_").replace(".npz", ".json")).read_text())
        lam_avg, d = dev_lambda(run["mnar_dev"])
        lam_cons, _ = dev_lambda(run["mnar_dev"], conservative=True)
        lam_T, d_T = dev_lambda(run["mnar"])
        ev, exact = saved_eval(f)
        tru = unit_loss(ev.PT.astype(float), ev.yT[None, :], loss).mean(1)
        iu = np.triu_indices(len(tru), 1)
        dt = (tru[:, None] - tru[None, :])[iu]
        for tag, lam in (("average", lam_avg), ("conservative", lam_cons)):
            for direction in (d, None):
                cl, cu, _ = aa_bootstrap_ci(ev, loss, ev.gamma_per_panel(lam), direction, exact, n_boot, rng)
                dec = (cl > 0) | (cu < 0)
                rows.append({"run": Path(f).stem, "lambda_kind": tag, "lambda": lam, "direction_dev": d,
                             "lambda_T_avg(oracle)": lam_T, "direction_T(oracle)": d_T,
                             "rule": f"{tag} lambda, {'one-sided (' + direction + ')' if direction else 'two-sided'}",
                             "decided_frac": dec.mean(),
                             "decided_correct": np.mean(np.where(cl > 0, 1, -1)[dec] == np.sign(dt[dec])) if dec.any() else np.nan,
                             "ci_covers_true_delta": np.mean((cl <= dt) & (dt <= cu))})
    return pd.DataFrame(rows)

def _run_for(npz):
    return json.loads(Path(npz.replace("arrays_", "run_").replace(".npz", ".json")).read_text())


def _roles_for(run):
    """Re-create the index sets of a run (incl. the random re-partition of design rs)."""
    from .run_c2012 import roles
    coh = build()
    if run["design"] == "rs":
        perm = np.random.default_rng(500 + run["seed"]).permutation(len(coh.y))
        coh.set = np.empty(len(coh.y), dtype="<U1")
        for i, s_ in enumerate("abc"):
            coh.set[perm[i * 4000:(i + 1) * 4000]] = s_
    return coh, roles(coh, tuple(run["order"]), run["design"])


def dm_dev(design, loss):
    """Same-population labeled baseline that the {V,U} regime deliberately ignores:
    pattern-wise outcome models P(Y | x_o(m), M=m) fitted on the labeled,
    naturally-missing development set D (LR+HGB ensemble; patterns with < 30 D
    units or one class fall back to D units whose pattern contains m), plugged into
    U.  Caveat: candidates were trained on the same D, so this is optimistic for
    candidates that fit D's noise the same way."""
    from .evaluators import agreement, outcome_model, unit_loss
    rows = []
    for f in sorted(glob.glob(str(ROOT / f"results/c2012/arrays_{design}_*.npz"))):
        run = _run_for(f)
        coh, (D, V, U, T, _) = _roles_for(run)
        spec = make_spec(coh, drop_icu=(design == "icu"))
        ev, _ = saved_eval(f)
        P = patterns(spec.K)
        MD, MU = coh.M[D], coh.M[U]
        pidD, pidU = pattern_id(MD), pattern_id(MU)
        Xn = coh.X[D].copy()
        for k, idx in enumerate(spec.panel_idx):
            Xn[np.ix_(~MD[:, k], idx)] = np.nan
        prep = Prep(spec).fit(Xn)
        p = np.zeros(len(U))
        for m in np.unique(pidU):
            tr = pidD == m
            if tr.sum() < 30 or len(np.unique(coh.y[D][tr])) < 2:
                tr = (MD | ~P[m][None, :]).all(1)
            Ztr, Zte = prep.sub(coh.X[D][tr], P[m]), prep.sub(coh.X[U][pidU == m], P[m])
            p[pidU == m] = np.mean([outcome_model(fam).fit(Ztr, coh.y[D][tr]).predict_proba(Zte)[:, 1] for fam in ("lr", "hgb")], 0)
        l0, l1 = loss01(ev.PU.astype(float), loss)
        est = (l0 * (1 - p) + l1 * p).mean(1)
        tru = unit_loss(ev.PT.astype(float), ev.yT[None, :], loss).mean(1)
        rows.append(agreement(est, tru))
    return {k: ms([r[k] for r in rows]) for k in ("kendall_tau", "top1_regret", "pair_sign_agreement")}


def flip_pairs_involvement(runs, loss, ev_name="drop50"):
    """Which candidates appear in the bootstrap-significant flips of an evaluator."""
    from collections import Counter
    cnt, tot, maskaware = Counter(), 0, 0
    for r in runs:
        names = r["names"]
        iu = np.triu_indices(len(names), 1)
        ts = np.array(r[loss]["pairs"]["truth_sig_sign"])
        es = np.array(r[loss]["pairs"][ev_name]["sig_sign"])
        for k in np.flatnonzero(es * ts < 0):
            a, b = names[iu[0][k]], names[iu[1][k]]
            cnt[a] += 1; cnt[b] += 1; tot += 1
            maskaware += any(x.endswith(("nat_mask", "nat_drop_mask")) for x in (a, b))
    return {"flips": tot, "pairs_with_natural_mask_aware_model": maskaware,
            "most_involved": cnt.most_common(6)}


def nat_cc_preserved(runs, loss):
    """Share of (natural-trained, complete-case-trained) candidate pairs in which an
    evaluator ranks the natural-trained model better (truth: see 'truth')."""
    out = {}
    for e in ["truth"] + EVAL_ORDER[:11]:
        fr = []
        for r in runs:
            v = np.array(r[loss]["risks"][e])
            nat = [i for i, n in enumerate(r["names"]) if ":nat_" in n]
            cc = [i for i, n in enumerate(r["names"]) if ":cc_" in n]
            fr.append(np.mean([v[i] < v[j] for i in nat for j in cc]))
        out[e] = round(float(np.mean(fr)), 3)
    return out


def aa_on_flips(design, loss, n_boot=200, seed=2):
    """For each bootstrap-significant drop50 flip: what does the acquisition-aware
    rule (lambda and direction from development labels only) decide?"""
    from .evaluators import aa_bootstrap_ci
    rng = np.random.default_rng(seed)
    tally = {"flips": 0, "aa_correct": 0, "aa_undecided": 0, "aa_wrong": 0, "dr_gamma1_correct": 0}
    for f in sorted(glob.glob(str(ROOT / f"results/c2012/arrays_{design}_*.npz"))):
        run = _run_for(f)
        ts = np.array(run[loss]["pairs"]["truth_sig_sign"])
        es = np.array(run[loss]["pairs"]["drop50"]["sig_sign"])
        flips = np.flatnonzero(es * ts < 0)
        if not len(flips):
            continue
        lam, d = dev_lambda(run["mnar_dev"])
        ev, exact = saved_eval(f)
        cl, cu, _ = aa_bootstrap_ci(ev, loss, ev.gamma_per_panel(lam), d, exact, n_boot, rng)
        c1, u1, _ = aa_bootstrap_ci(ev, loss, np.ones(len(ev.pidU)), None, exact, n_boot, rng)
        for k in flips:
            tally["flips"] += 1
            dec = 1 if cl[k] > 0 else (-1 if cu[k] < 0 else 0)
            tally["aa_correct" if dec == ts[k] else ("aa_undecided" if dec == 0 else "aa_wrong")] += 1
            tally["dr_gamma1_correct"] += int((1 if c1[k] > 0 else (-1 if u1[k] < 0 else 0)) == ts[k])
    return tally


def mnar_pooled():
    """Per pattern: odds of death in pattern m vs the complete-case law given
    x_o(m).  Cross-fitted by challenge set (models from the complete cases of the
    other two sets), pooled over all 12,000 stays.  Three estimators:
      offset-GLM with an LR, an HGB and an ensemble (mean) outcome model, each
      also reported relative to its own 1111 control; and
      IW: observed pattern-m odds vs complete-case odds reweighted to pattern m's
      x_o(m) distribution by a logistic domain classifier (no outcome model).
    Uses labels: evaluation only.  A departure from 1 means Y is not independent
    of M given X_o(m) (dependence on Y OR on unrecorded values) -- or misfit of
    the model used, which is why several are shown."""
    from .evaluators import outcome_model
    from .models import domain_weights, weights_from
    coh = build()
    spec = make_spec(coh)
    P = patterns(spec.K)
    pid = pattern_id(coh.M)
    fams = ("lr", "hgb")
    off = {f: np.full(len(pid), np.nan) for f in fams}
    iw_exp = np.full(len(pid), np.nan)
    for s_ in "abc":
        tr = (coh.set != s_) & coh.M.all(1)
        te = coh.set == s_
        prep = Prep(spec).fit(coh.X[tr])
        for m in range(len(P)):
            sel = te & (pid == m)
            if not sel.any():
                continue
            Ztr, Zte = prep.sub(coh.X[tr], P[m]), prep.sub(coh.X[sel], P[m])
            for f in fams:
                p = np.clip(outcome_model(f).fit(Ztr, coh.y[tr]).predict_proba(Zte)[:, 1], 1e-6, 1 - 1e-6)
                off[f][sel] = np.log(p / (1 - p))
            if m != len(P) - 1:
                clf = domain_weights(Ztr, Zte)
                w = weights_from(clf, Ztr, len(Ztr), len(Zte), clip_q=0.995)
                iw_exp[sel] = np.sum(w * coh.y[tr]) / np.sum(w)     # complete-case mortality reweighted to pattern m
    ens = np.log(1 / (1 + np.exp(-off["lr"])) / 2 + 1 / (1 + np.exp(-off["hgb"])) / 2)
    ens = np.log(np.exp(ens) / (1 - np.exp(ens)))
    offs = {"lr": off["lr"], "hgb": off["hgb"], "ens": ens}

    def glm(y, o):
        fit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=o).fit()
        return float(fit.params[0]), float(fit.bse[0])

    ctrl = {f: glm(coh.y[pid == len(P) - 1], offs[f][pid == len(P) - 1])[0] for f in offs}
    rows = []
    for m in range(len(P)):
        sel = pid == m
        y = coh.y[sel]
        row = {"pattern(ABG,ALINE,LACT,LIVER)": "".join(str(int(v)) for v in P[m]),
               "n_missing_panels": int(spec.K - P[m].sum()), "n": int(sel.sum()), "deaths": int(y.sum()),
               "observed_mortality": round(float(y.mean()), 3)}
        for f in ("ens", "lr", "hgb"):
            b, se = glm(y, offs[f][sel])
            row[f"OR_{f} [95% CI]"] = f"{np.exp(b):.2f} [{np.exp(b - 1.96 * se):.2f}, {np.exp(b + 1.96 * se):.2f}]"
            row[f"OR_{f}/control"] = round(float(np.exp(b - ctrl[f])), 2)
            if f == "ens":
                row["log_or_ens"], row["se_ens"] = b, se
        if m != len(P) - 1:
            pe = float(np.mean(iw_exp[sel]))
            po = float(y.mean())
            row["OR_IW"] = round((po / (1 - po)) / (pe / (1 - pe)), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_missing_panels", "pattern(ABG,ALINE,LACT,LIVER)"])

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
            for kind in ("aa", "aa_boot"):
                if kind not in rr[0][loss]:
                    continue
                for key in rr[0][loss][kind]:
                    d, lam = key.split(":")
                    A = [r[loss][kind][key] for r in rr]
                    aa.append({"loss": loss, "mechanism": mech, "lambda_true": rr[0]["lambda_true"],
                               "uncertainty": "bootstrap CI" if kind == "aa_boot" else "none (identified set only)",
                               "direction": d, "lambda": float(lam), "covers": ms([a["covers"] for a in A]),
                               "decided_frac": ms([a["decided_frac"] for a in A]),
                               "decided_correct": ms([a["decided_correct"] for a in A])})
            if "boot_ref" in rr[0][loss]:
                for e in rr[0][loss]["boot_ref"]:
                    A = [r[loss]["boot_ref"][e] for r in rr]
                    aa.append({"loss": loss, "mechanism": mech, "lambda_true": rr[0]["lambda_true"],
                               "uncertainty": f"reference: {e} + bootstrap CI", "direction": "-", "lambda": float("nan"),
                               "covers": ms([a["covers"] for a in A]), "decided_frac": ms([a["decided_frac"] for a in A]),
                               "decided_correct": ms([a["decided_correct"] for a in A])})
    return pd.DataFrame(rows), pd.DataFrame(aa)


def flat(df):
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [f"{a}_{b}" for a, b in d.columns]
    return d.reset_index().to_dict("records")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# N1 results summary (generated by `python -m src.n1.summarize`)\n",
          "Designs: `cc` = the challenge's own sets a/b/c in all 6 role orderings (only 3 distinct V/U samples and 3 "
          "distinct T samples; orderings sharing V/U are NOT independent); `rs` = 6 random re-partitions of the 12,000 "
          "stays (distinct V/U/T draws); `icu` = surgical complete cases -> medical ICUs, 6 orderings.\n"]
    js = {}
    for design in ("cc", "rs", "icu"):
        runs = load_runs(design)
        if not runs:
            continue
        md.append(f"\n## Challenge 2012, design `{design}` ({len(runs)} runs)\n")
        md.append("Roles/sizes (first run): " + json.dumps(runs[0]["n"]) + "; prevalence: " +
                  json.dumps({k: round(v, 3) for k, v in runs[0]["prevalence"].items()}) +
                  f"; EM prior estimates: {ms([r['em_prior'] for r in runs])}; raw weight mean on V (sel): "
                  f"{ms([r.get('w_sel_raw_mean', np.nan) for r in runs])}\n")
        js[design] = {"n_runs": len(runs)}
        for loss in ("brier", "logloss"):
            t = agreement_table(runs, loss)
            md.append(f"\n### {loss}: agreement with the natural-missingness truth\n\n" + t.drop(columns="_tau").to_markdown(index=False) + "\n")
            f = flips_table(runs, loss)
            npairs = len(runs[0]["names"]) * (len(runs[0]["names"]) - 1) // 2
            md.append(f"\n### {loss}: bootstrap-significant pairwise disagreements (of {npairs} pairs)\n\n" + f.to_markdown(index=False) + "\n")
            fx = flip_explanation(runs, loss, "drop50")
            md.append(f"\n### {loss}: repaired vs broken (truth-significant pairs, point sign) relative to dropout p=0.5\n\n" + pd.DataFrame(fx).to_markdown(index=False) + "\n")
            L = ladder_table(runs, loss)
            md.append(f"\n### {loss}: within-truth-set ladder (V = complete cases of T, U = T)\n\n" + L.to_markdown(index=False) + "\n")
            S = strategy_ranks(runs, loss)
            md.append(f"\n### {loss}: mean rank of each candidate (1 = best)\n\n" + S.round(4).to_markdown() + "\n")
            A = aa_table(runs, loss)
            md.append(f"\n### {loss}: acquisition-aware bounds without sampling uncertainty, scalar Gamma\n\n" + A.to_markdown(index=False) + "\n")
            out = {"agreement": t.to_dict("records"), "flips": f.to_dict("records"), "flip_explanation": fx,
                   "ladder": L.to_dict("records"), "aa": A.to_dict("records"),
                   "strategy_ranks": S.reset_index().rename(columns={"index": "model"}).to_dict("records")}
            try:
                AB = aa_bootstrap(design, loss)
                md.append(f"\n### {loss}: acquisition-aware selection with bootstrap 95% CI, lambda/Gamma grid\n\n" + AB.to_markdown() + "\n")
                out["aa_bootstrap"] = flat(AB)
            except Exception as ex:
                md.append(f"\n(aa bootstrap unavailable: {ex})\n")
            try:
                DC = aa_dev_calibrated(design, loss)
                agg = DC.groupby("rule")[["lambda", "decided_frac", "decided_correct", "ci_covers_true_delta"]].agg(["mean", "std"]).round(3)
                per = DC[["run", "lambda_kind", "lambda", "direction_dev", "lambda_T_avg(oracle)", "direction_T(oracle)"]].drop_duplicates().round(3)
                md.append(f"\n### {loss}: acquisition-aware selection, lambda and direction from DEVELOPMENT labels only\n\n"
                          + agg.to_markdown() + "\n\nper run:\n\n" + per.to_markdown(index=False) + "\n")
                out["aa_dev_calibrated"] = flat(agg)
                out["aa_dev_lambdas"] = per.to_dict("records")
            except Exception as ex:
                md.append(f"\n(dev-calibrated aa unavailable: {ex})\n")
            for fam in (None, "lr", "hgb"):
                try:
                    LP = label_part_analysis(design, loss, family=fam)
                    md.append(f"\n### {loss}: non-identified part of pairwise differences, outcome model = {fam or 'ensemble'}\n\n" + LP.to_markdown() + "\n")
                    out[f"label_part_{fam or 'ens'}"] = flat(LP)
                except Exception as ex:
                    md.append(f"\n(label-part analysis unavailable: {ex})\n")
            try:
                extra = {"dm_dev (D-labeled outcome model, not {V,U})": dm_dev(design, loss),
                         "drop50 flips: involvement": flip_pairs_involvement(runs, loss),
                         "nat-vs-cc pairs ranked nat-better": nat_cc_preserved(runs, loss),
                         "drop50 flips: dev-calibrated AA decisions": aa_on_flips(design, loss)}
                md.append(f"\n### {loss}: extra analyses\n\n```\n" + json.dumps(extra, indent=1, default=str) + "\n```\n")
                out["extra"] = extra
            except Exception as ex:
                md.append(f"\n(extra analyses unavailable: {ex})\n")
            if loss == "brier":
                gt = gamma_transfer(runs)
                md.append("\n### Per-pattern log-OR: development data vs deployment truth\n\n" + json.dumps(gt) + "\n")
                js[design]["gamma_transfer"] = gt
            js[design][loss] = out
    mn = mnar_pooled()
    md.append("\n## Recording vs outcome given the recorded features (pooled, cross-fitted; truth labels used for evaluation only)\n\n"
              + mn.drop(columns=["log_or_ens", "se_ens"]).to_markdown(index=False) + "\n")
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
