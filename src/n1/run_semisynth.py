"""Semi-synthetic validation: real features and labels, KNOWN acquisition mechanism.

Only the 2,677 fully recorded stays are used, so every panel value exists and
masks can be imposed.  A fixed severity score s(x0) (base-only logistic model
fitted on the other 9,323 stays) drives covariate-dependent recording; the
label drives label-dependent recording:
    logit P(M_k = 1 | x0, y) = alpha_k + beta * s(x0) + delta * y,  independent over k,
with alpha_k set so that marginal availability matches ``RATES``.
Then, exactly, for pattern m with r unrecorded panels
    odds P(Y=1 | x_o(m), M=m) / odds P(Y=1 | x_o(m), M=1) = exp(-delta * r),
so the true per-panel sensitivity parameter is lambda* = exp(delta) (direction
"down" for delta > 0).

Roles (random split of the fully recorded stays per replicate):
  D 35 %: masks imposed, labeled         -> candidates
  pool 40 %: masks imposed; V = its units that stayed complete (labeled),
             U = all of it (labels hidden)
  T 25 %: masks imposed, labeled         -> truth, computed as the EXPECTED risk
             over the known mask law (exact given the T units)
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

from .evaluators import Evaluation, agreement, loss01, unit_loss
from .features_c2012 import build
from .models import Prep, fit_candidates, pattern_id, patterns
from .run_c2012 import make_spec

ROOT = Path(__file__).resolve().parents[2]
RATES = np.array([0.85, 0.80, 0.70, 0.60])
MECHS = {"MCAR": (0.0, 0.0), "MAR": (0.75, 0.0), "MNAR_0.25": (0.75, 0.25), "MNAR_0.5": (0.75, 0.5),
         "MNAR_1.0": (0.75, 1.0)}
EVALS = ["full", "drop50", "dams_rate", "freq", "freq_prior_plugin", "freq_prior_oracle", "sel", "pat", "dm", "dr"]


def severity(coh, spec):
    cc = coh.M.all(1)
    prep = Prep(spec).fit(coh.X[~cc])
    m = LogisticRegression(C=1.0, max_iter=5000).fit(prep.x0(coh.X[~cc]), coh.y[~cc])
    s = m.decision_function(prep.x0(coh.X[cc]))
    return (s - s.mean()) / s.std()


def calibrate_alpha(s, y, beta, delta):
    return np.array([brentq(lambda a: np.mean(1 / (1 + np.exp(-(a + beta * s + delta * y)))) - r, -20, 20)
                     for r in RATES])


def mask_probs(s, y, alpha, beta, delta):
    return 1 / (1 + np.exp(-(alpha[None, :] + beta * s[:, None] + delta * y[:, None])))


def one(mech, rep, out_dir):
    beta, delta = MECHS[mech]
    coh = build()
    spec = make_spec(coh)
    cc = np.where(coh.M.all(1))[0]
    s = severity(coh, spec)
    X, y = coh.X[cc], coh.y[cc]
    alpha = calibrate_alpha(s, y, beta, delta)
    rng = np.random.default_rng(10_000 * rep + hash(mech) % 997)
    pi = mask_probs(s, y, alpha, beta, delta)
    M = rng.random(pi.shape) < pi
    perm = rng.permutation(len(cc))
    nD, nP = int(0.35 * len(cc)), int(0.40 * len(cc))
    D, pool, T = perm[:nD], perm[nD:nD + nP], perm[nD + nP:]
    V = pool[M[pool].all(1)]
    Xm = X.copy()
    for k, idx in enumerate(spec.panel_idx):
        Xm[np.ix_(~M[:, k], idx)] = np.nan
    cands = fit_candidates(spec, Xm[D], M[D], y[D], Xm[pool], M[pool], seed=rep)
    ev = Evaluation(cands, spec, X[V], y[V], Xm[pool], M[pool], Xm[T], M[T], y[T], yU=y[pool], seed=rep,
                    v_in_u=np.where(M[pool].all(1))[0])
    # exact expected truth over the known mask law, per T unit
    P = patterns(spec.K)
    pm = np.prod(np.where(P[None, :, :], pi[T][:, None, :], 1 - pi[T][:, None, :]), axis=2)  # nT x 16
    J = len(cands)
    PTm = np.empty((J, len(P), len(T)))
    for m in range(len(P)):
        Mm = np.repeat(P[m][None, :], len(T), 0)
        for j, c in enumerate(cands):
            PTm[j, m] = c.predict(X[T], Mm)
    res = {"mech": mech, "rep": rep, "beta": beta, "delta": delta, "lambda_true": float(np.exp(delta)),
           "n": {"D": len(D), "V": len(V), "U": len(pool), "T": len(T)},
           "prev": {"V": float(y[V].mean()), "U": float(y[pool].mean())}, "names": ev.names}
    iu = np.triu_indices(J, 1)
    for loss in ("brier", "logloss"):
        L = unit_loss(PTm, y[T][None, None, :], loss)
        tru = np.einsum("jmi,im->j", L, pm) / len(T)
        est = ev.estimates(loss)
        dt = (tru[:, None] - tru[None, :])[iu]
        out = {"truth": tru.tolist(), "agreement": {e: agreement(est[e], tru) for e in EVALS}}
        for e in EVALS:
            out.setdefault("bias_mean_abs", {})[e] = float(np.mean(np.abs(est[e] - tru)))
            de = (est[e][:, None] - est[e][None, :])[iu]
            out.setdefault("pair_abs_err", {})[e] = float(np.mean(np.abs(de - dt)))
        aa = {}
        lam_grid = sorted(set([1.0, float(np.exp(0.25)), float(np.exp(0.5)), float(np.exp(1.0)), float(np.exp(1.5))]))
        for direction in (None, "down"):
            for lam in lam_grid:
                g = ev.gamma_per_panel(lam)
                b = np.array([ev.aa_pair(a, c, loss, g, direction) for a, c in zip(*iu)])
                lo, up = b[:, 0], b[:, 2]
                decided = (lo > 0) | (up < 0)
                sgn = np.where(lo > 0, 1, np.where(up < 0, -1, 0))
                aa[f"{direction or 'two'}:{lam:.3f}"] = {
                    "covers": float(np.mean((lo <= dt) & (dt <= up))),
                    "decided_frac": float(decided.mean()),
                    "decided_correct": float(np.mean(np.sign(dt[decided]) == sgn[decided])) if decided.any() else float("nan"),
                    "width": float(np.mean(up - lo))}
        out["aa"] = aa
        res[loss] = out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"semi_{mech}_r{rep}.json").write_text(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results/semisynth"))
    a = ap.parse_args()
    jobs = [(m, r) for m in MECHS for r in range(a.reps)]
    Parallel(n_jobs=a.jobs)(delayed(one)(m, r, Path(a.out)) for m, r in jobs)


if __name__ == "__main__":
    main()
