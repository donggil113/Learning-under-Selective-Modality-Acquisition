"""Real-data study on PhysioNet/CinC Challenge 2012 (open access, ODC-By 1.0).

Roles, rotated over the six orderings of the challenge's own sets (a, b, c):
  D  labeled development set, natural masks           -> fits every candidate
  V  complete cases of the second set, labeled         -> the "fully observed cohort"
  U  every unit of the second set, labels hidden       -> unlabeled deployment sample
  T  every unit of the third set, labeled              -> ground-truth deployment risk
Eligible population: all 12,000 stays; cutoff 48 h; label in-hospital death.

Information table (what each party can read):
                 inputs              mask            label
  D (develop)    X_obs               natural M       yes
  V (validate)   X (all panels)      M = 1           yes
  U (deploy)     X_obs               natural M       no  (except the explicit lab_n budget)
  T (truth)      X_obs               natural M       yes, evaluation only
Target labels exist for every deployment unit in this benchmark, so nothing
here is a non-identifiability claim; U hides them to emulate selection time.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from scipy.stats import norm

from .evaluators import Evaluation, agreement, loss01, outcome_model, unit_loss
from .features_c2012 import BASE_GROUPS, build
from .models import Prep, Spec, fit_candidates, pattern_id, patterns

ROOT = Path(__file__).resolve().parents[2]
EVALS = ["full", "drop50", "dams_rate", "freq", "freq_prior", "freq_prior_plugin", "freq_prior_oracle", "sel", "pat", "dm", "dr"]
BOOT_EVALS = ["drop50", "dams_rate", "freq", "freq_prior_oracle", "sel", "pat", "dr"]
GAMMAS = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]


def make_spec(coh, drop_icu=False):
    """``drop_icu``: remove the ICU-type dummies from x0.  Used for the selection
    rules' auxiliary models in the cross-ICU design, where ICU type IS the
    selection variable (V surgical vs U medical) and has no overlap."""
    groups = {g: [coh.cols.index(c) for c in coh.cols if c.split("_")[0] in vs] for g, vs in BASE_GROUPS.items()}
    base = [i for i in coh.base_idx if not (drop_icu and coh.cols[i].startswith("icu"))]
    return Spec(base_idx=base, panel_idx=[list(p) for p in coh.panel_idx], groups=groups)


def boot_estimates(ev: Evaluation, loss, idxV, idxU):
    """Re-evaluate the V/U-based estimators on a bootstrap resample.  V units and
    U units are resampled (pattern frequencies and marginal rates are recomputed
    from the U resample); the fitted weight, mask and outcome models are held
    fixed, so these CIs are conditional on the auxiliary fits."""
    LV = unit_loss(ev.PV[:, :, idxV], ev.yV[None, None, idxV], loss)
    freq = np.bincount(ev.pidU[idxU], minlength=ev.nP) / len(idxU)
    rate = ev.P[ev.pidU[idxU]].mean(0)
    q_rate = np.prod(np.where(ev.P, rate[None, :], 1 - rate[None, :]), axis=1)
    out = {}
    R = LV.mean(2)
    out["drop50"] = R.mean(1)
    out["dams_rate"] = R @ q_rate
    out["freq"] = R @ freq
    piS, pi = ev.yV[idxV].mean(), ev.yT.mean()
    wy = np.where(ev.yV[idxV] == 1, pi / piS, (1 - pi) / (1 - piS))
    out["freq_prior_oracle"] = ((LV * wy[None, None]).sum(2) / wy.sum()) @ freq
    ws, pm = ev.w_sel[idxV], ev.pm_sel[idxV]
    out["sel"] = np.einsum("jmi,im,i->j", LV, pm, ws) / ws.sum()
    wp = ev.w_pat[:, idxV]
    out["pat"] = ((LV * wp[None]).sum(2) / wp.sum(1)[None]) @ freq
    l0, l1 = loss01(ev.PU[:, idxU], loss)
    dm = (l0 * (1 - ev.pS_U[idxU]) + l1 * ev.pS_U[idxU]).mean(1)
    v0, v1 = loss01(ev.PV[:, :, idxV], loss)
    ps = ev.pS_V_cf[:, idxV][None]
    res = ((LV - (v0 * (1 - ps) + v1 * ps)) * wp[None]).sum(2) / wp.sum(1)[None]
    out["dr"] = dm + res @ freq
    return out


def mnar_diagnostic(coh, spec, D, Vset, T):
    """Odds ratio of death between each incomplete pattern and the complete cases
    given the pattern's observed features, estimated on the TRUTH set's labels
    (evaluation only).  Outcome models are fitted on the complete cases of D and V."""
    idx_cc = np.r_[D, Vset][coh.M[np.r_[D, Vset]].all(1)]
    prep = Prep(spec).fit(coh.X[idx_cc])
    P = patterns(spec.K)
    pidT = pattern_id(coh.M[T])
    rows = []
    for m in range(len(P)):
        selT = T[pidT == m]
        if len(selT) < 30:
            continue
        Ztr, Zte = prep.sub(coh.X[idx_cc], P[m]), prep.sub(coh.X[selT], P[m])
        pf = {f: outcome_model(f).fit(Ztr, coh.y[idx_cc]).predict_proba(Zte)[:, 1] for f in ("lr", "hgb")}
        p = np.clip((pf["lr"] + pf["hgb"]) / 2, 1e-6, 1 - 1e-6)   # ensemble outcome model
        off = np.log(p / (1 - p))
        y = coh.y[selT]
        fam = {}
        for f, pp in pf.items():
            pp = np.clip(pp, 1e-6, 1 - 1e-6)
            try:
                fam[f] = float(sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                                      offset=np.log(pp / (1 - pp))).fit().params[0])
            except Exception:
                fam[f] = float("nan")
        try:
            fit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=off).fit()
            b, se = float(fit.params[0]), float(fit.bse[0])
            fit2 = sm.GLM(y, sm.add_constant(off), family=sm.families.Binomial()).fit()
            slope = float(fit2.params[1])
        except Exception:
            b, se, slope = np.nan, np.nan, np.nan
        rows.append({"pattern": "".join(str(int(v)) for v in P[m]), "n": int(len(selT)),
                     "deaths": int(y.sum()), "observed_rate": float(y.mean()), "complete_case_model_rate": float(p.mean()),
                     "log_or": b, "se": se, "or": float(np.exp(b)), "or_lo": float(np.exp(b - 1.96 * se)),
                     "or_hi": float(np.exp(b + 1.96 * se)), "calibration_slope": slope,
                     "log_or_lr": fam["lr"], "log_or_hgb": fam["hgb"]})
    return rows


def roles(coh, order, design):
    """Index sets for one run.  design 'cc': V = complete cases of the second set,
    U = all of it (V inside U).  design 'icu': V = complete cases of the second set
    in the surgical ICUs (CSRU, SICU), U and T restricted to the medical ICUs (CCU,
    MICU) -- population shift on top of completeness selection."""
    sD, sV, sT = order
    D = np.where(coh.set == sD)[0]
    if design in ("cc", "rs"):
        Uidx = np.where(coh.set == sV)[0]
        Vidx = Uidx[coh.M[Uidx].all(1)]
        T = np.where(coh.set == sT)[0]
        v_in_u = np.where(coh.M[Uidx].all(1))[0]
    else:
        med, sur = np.isin(coh.icu, [1, 3]), np.isin(coh.icu, [2, 4])
        Uidx = np.where((coh.set == sV) & med)[0]
        Vidx = np.where((coh.set == sV) & sur & coh.M.all(1))[0]
        T = np.where((coh.set == sT) & med)[0]
        v_in_u = None
    return D, Vidx, Uidx, T, v_in_u


def within_truth_ladder(cands, spec, coh, T, seed):
    """Same estimators, but V = complete cases of T and U = T itself: removes the
    V-vs-T sampling noise and isolates what each adjustment explains."""
    Tc = T[coh.M[T].all(1)]
    ev = Evaluation(cands, spec, coh.X[Tc], coh.y[Tc], coh.X[T], coh.M[T], coh.X[T], coh.M[T], coh.y[T],
                    seed=seed, v_in_u=np.where(coh.M[T].all(1))[0])
    out = {}
    for loss in ("brier", "logloss"):
        est, tru = ev.estimates(loss), ev.truth(loss)
        out[loss] = {e: {"agreement": agreement(est[e], tru), "risk": est[e].tolist()}
                     for e in ["full", "drop50", "dams_rate", "freq", "freq_prior_plugin", "freq_prior_oracle", "sel", "pat", "dr"]}
        # Gamma = 1 limit: replace each incomplete unit's label by the complete-case
        # outcome model's probability P(Y=1 | x_o(m), M=1).  Its ranking is what a
        # perfect covariate adjustment converges to; its disagreement with the truth
        # is caused only by label-dependent recording (plus outcome-model error).
        # Reported for the ensemble outcome model and, as a sensitivity check, for
        # each family alone (an LR reference structurally favours LR candidates).
        l0, l1 = loss01(ev.PT, loss)
        inc = ev.pidT != ev.full_id
        for tag, pS in (("ens", ev.pS_T), ("lr", ev.fam_pS_T["lr"]), ("hgb", ev.fam_pS_T["hgb"])):
            lab = ((ev.yT - pS)[None, :] * (l1 - l0) * inc[None, :]).mean(1)
            key = "gamma1_limit" if tag == "ens" else f"gamma1_limit_{tag}"
            out[loss][key] = {"agreement": agreement(tru - lab, tru), "risk": (tru - lab).tolist(), "label_part": lab.tolist()}
    return out


def one_run(order, seed, n_boot, out_dir, design="cc"):
    t0 = time.time()
    coh = build()
    if design == "rs":
        # independent replicate: random re-partition of the 12,000 stays into three
        # sets of 4,000 (the challenge's own sets are used by designs cc and icu)
        perm = np.random.default_rng(500 + seed).permutation(len(coh.y))
        coh.set = np.empty(len(coh.y), dtype="<U1")
        for i, s_ in enumerate("abc"):
            coh.set[perm[i * 4000:(i + 1) * 4000]] = s_
    spec = make_spec(coh)
    spec_eval = make_spec(coh, drop_icu=(design == "icu"))
    D, Vidx, Uidx, T, v_in_u = roles(coh, order, design)
    X = coh.X.copy()
    cands = fit_candidates(spec, X[D], coh.M[D], coh.y[D], X[Uidx], coh.M[Uidx], seed=seed)
    ev = Evaluation(cands, spec_eval, X[Vidx], coh.y[Vidx], X[Uidx], coh.M[Uidx], X[T], coh.M[T], coh.y[T],
                    yU=coh.y[Uidx], seed=seed, v_in_u=v_in_u)
    complete_exact = design in ("cc", "rs")
    rng = np.random.default_rng(1000 + seed)
    res = {"design": design, "order": "".join(order), "seed": seed, "names": ev.names, "n": {"D": len(D), "V": len(Vidx),
           "U": len(Uidx), "T": len(T)}, "prevalence": {"D": float(coh.y[D].mean()), "V": float(coh.y[Vidx].mean()),
           "U": float(coh.y[Uidx].mean()), "T": float(coh.y[T].mean())}, "em_prior": ev.em_prior(),
           "freq": ev.freq.tolist(), "q_rate": ev.q_rate.tolist(),
           "w_sel_ess": float(ev.w_sel.sum() ** 2 / (ev.w_sel ** 2).sum()),
           "w_pat_ess": (ev.w_pat.sum(1) ** 2 / (ev.w_pat ** 2).sum(1)).tolist(),
           "w_sel_raw_mean": ev.w_sel_raw_mean, "w_pat_raw_mean": ev.w_pat_raw_mean.tolist()}
    J = len(cands)
    iu = np.triu_indices(J, 1)
    for loss in ("brier", "logloss"):
        est = ev.estimates(loss)
        tru = ev.truth(loss)
        R = {"truth": tru.tolist()}
        agr = {}
        for e in EVALS:
            R[e] = est[e].tolist()
            agr[e] = agreement(est[e], tru)
        for n in (50, 100, 200, 400):
            a = [agreement(d, tru) for d in est[f"lab{n}"]]
            agr[f"lab{n}"] = {k: float(np.mean([x[k] for x in a])) for k in ("kendall_tau", "spearman", "top1_regret", "pair_sign_agreement")}
        # bootstrap: pairwise significance of disagreements
        dt = (tru[:, None] - tru[None, :])[iu]
        LT = unit_loss(ev.PT, ev.yT[None, :], loss)
        bt = np.empty((n_boot, len(dt)))
        be = {e: np.empty((n_boot, len(dt))) for e in BOOT_EVALS}
        nV, nU, nT = len(Vidx), len(Uidx), len(T)
        for b in range(n_boot):
            it = rng.integers(0, nT, nT)
            r = LT[:, it].mean(1)
            bt[b] = (r[:, None] - r[None, :])[iu]
            bb = boot_estimates(ev, loss, rng.integers(0, nV, nV), rng.integers(0, nU, nU))
            for e in BOOT_EVALS:
                be[e][b] = (bb[e][:, None] - bb[e][None, :])[iu]
        tlo, thi = np.percentile(bt, [2.5, 97.5], axis=0)
        tsig = np.where(tlo > 0, 1, np.where(thi < 0, -1, 0))
        pair = {"truth_sig_sign": tsig.tolist(), "truth_delta": dt.tolist()}
        for e in BOOT_EVALS:
            elo, ehi = np.percentile(be[e], [2.5, 97.5], axis=0)
            esig = np.where(elo > 0, 1, np.where(ehi < 0, -1, 0))
            de = (est[e][:, None] - est[e][None, :])[iu]
            se_e, se_t = be[e].std(0), bt.std(0)
            exp_cov = 2 * norm.cdf(1.96 * se_e / np.sqrt(se_e ** 2 + se_t ** 2 + 1e-30)) - 1
            pair[e] = {"sig_sign": esig.tolist(), "delta": de.tolist(),
                       "expected_cover_if_unbiased": float(np.mean(exp_cov)),
                       "sig_flips": int(np.sum((esig * tsig) < 0)),
                       "sig_agree": int(np.sum((esig * tsig) > 0)),
                       "truth_sig_pairs": int(np.sum(tsig != 0)), "eval_sig_pairs": int(np.sum(esig != 0)),
                       "ci_covers_truth": float(np.mean((elo <= dt) & (dt <= ehi)))}
        # acquisition-aware bounds over a Gamma grid (two-sided and one-sided "down")
        aa = {}
        for direction in (None, "down"):
            for g in GAMMAS:
                lo = np.empty(len(dt)); up = np.empty(len(dt)); pt = np.empty(len(dt))
                for k, (a, b2) in enumerate(zip(*iu)):
                    lo[k], pt[k], up[k] = ev.aa_pair(a, b2, loss, g, direction, complete_exact)
                decided = (lo > 0) | (up < 0)
                sgn = np.where(lo > 0, 1, np.where(up < 0, -1, 0))
                aa[f"{direction or 'two'}:{g}"] = {
                    "decided_frac": float(decided.mean()),
                    "decided_correct": float(np.mean(np.sign(dt[decided]) == sgn[decided])) if decided.any() else float("nan"),
                    "decided_wrong_sig": int(np.sum((sgn * tsig) < 0)),
                    "covers_truth": float(np.mean((lo <= dt) & (dt <= up))),
                    "robust_best": [ev.names[j] for j in range(J)
                                    if all((ev.aa_pair(j, k2, loss, g, direction, complete_exact)[0] <= 0) for k2 in range(J) if k2 != j)],
                }
        # oracle split of each candidate's IW-estimate error into the label part
        l0, l1 = loss01(ev.PT, loss)
        inc = ev.pidT != ev.full_id
        label_part = ((ev.yT - ev.pS_T)[None, :] * (l1 - l0) * inc[None, :]).mean(1)
        res[loss] = {"risks": R, "agreement": agr, "pairs": pair, "aa": aa,
                     "oracle_label_part": label_part.tolist(),
                     "true_best": ev.names[int(np.argmin(tru))]}
    res["mnar"] = mnar_diagnostic(coh, spec_eval, D, Vidx, T)
    res["mnar_dev"] = mnar_diagnostic(coh, spec_eval, Vidx, np.array([], dtype=int), D)  # Gamma calibration from labeled development data
    res["ladder_T"] = within_truth_ladder(cands, spec_eval, coh, T, seed)
    res["seconds"] = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    # arrays for post-hoc analyses (no raw patient features are stored)
    np.savez_compressed(out_dir / f"arrays_{design}_{''.join(order)}_s{seed}.npz",
                        PV=ev.PV.astype(np.float32), PU=ev.PU.astype(np.float32), PT=ev.PT.astype(np.float32),
                        yV=ev.yV, yU=ev.yU, yT=ev.yT, pidU=ev.pidU, pidT=ev.pidT, pS_U=ev.pS_U, pS_T=ev.pS_T,
                        pS_V_cf=ev.pS_V_cf, w_sel=ev.w_sel, pm_sel=ev.pm_sel, w_pat=ev.w_pat, freq=ev.freq,
                        q_rate=ev.q_rate, names=np.array(ev.names), complete_exact=complete_exact,
                        pS_U_lr=ev.fam_pS_U["lr"], pS_U_hgb=ev.fam_pS_U["hgb"], pS_T_lr=ev.fam_pS_T["lr"],
                        pS_T_hgb=ev.fam_pS_T["hgb"], pS_V_cf_lr=ev.fam_pS_V_cf["lr"], pS_V_cf_hgb=ev.fam_pS_V_cf["hgb"])
    (out_dir / f"run_{design}_{''.join(order)}_s{seed}.json").write_text(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--orders", default="all")
    ap.add_argument("--design", default="cc", choices=["cc", "icu", "rs"])
    ap.add_argument("--out", default=str(ROOT / "results/c2012"))
    a = ap.parse_args()
    orders = list(itertools.permutations("abc")) if a.orders == "all" else [tuple(o) for o in a.orders.split(",")]
    if a.design == "rs":
        orders = [tuple("abc")]  # each seed is its own random re-partition
    jobs = [(o, s) for o in orders for s in range(a.seeds)]
    Parallel(n_jobs=a.jobs)(delayed(one_run)(o, s, a.boot, Path(a.out), a.design) for o, s in jobs)


if __name__ == "__main__":
    main()
