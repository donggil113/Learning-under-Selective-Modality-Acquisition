r"""Exact statements behind N1, each checked numerically by ``verify()``.

Notation (binary label, one prediction per unit):
  * ``S``  population indicator (which cohort a unit belongs to).  In the
    complete-case design S=src is *defined* by M=1 inside the development split,
    so S is a function of M there; in a cross-site design S is the site.
  * ``M``  record-availability mask at the prediction cutoff (what the deployed
    model sees).  Never identified with acquisition ``A`` (test performed).
  * ``f(x_o(m), m)`` a predictor applied to the observed part of pattern m.

For a loss that is affine in the label, l(f, y) = l0 + y (l1 - l0), the risk of a
pattern-m prediction only depends on P(Y=1 | X_o(m), M=m).  Everything below is
a consequence of that affinity.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np
from scipy.optimize import linprog


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------

def brier(f, y):
    return (np.asarray(y) - np.asarray(f)) ** 2


def logloss(f, y, eps=1e-12):
    f = np.clip(np.asarray(f, dtype=float), eps, 1 - eps)
    y = np.asarray(y)
    return -(y * np.log(f) + (1 - y) * np.log(1 - f))


def loss_pair(f, loss="brier"):
    """(l(f,0), l(f,1)) for each prediction."""
    if loss == "brier":
        f = np.asarray(f, dtype=float)
        return f ** 2, (1 - f) ** 2
    if loss == "logloss":
        return logloss(f, 0), logloss(f, 1)
    raise ValueError(loss)


# --------------------------------------------------------------------------
# 1. Minimal counterexample: rate-matched dropout + invariant complete-data law
#    + unlabeled target data cannot rank two models.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class World:
    name: str
    r: dict  # r[(x, y)] = P_T(M=1 | X=x, Y=y)


def ce1():
    """One binary modality X, binary Y, no always-observed covariate.

    Complete-data law P(X, Y) is the SAME in source and target (the
    missingness-shift invariance).  Source is fully observed.  Target mask:
      W1: MCAR, P(M=1)=1/2.
      W2: label-dependent, chosen so that P_T(M, X*M) equals W1 exactly.
    Model A and B agree whenever X is observed (both output P(Y|X)); when X is
    missing, A outputs 1/2 (what a dropout- or rate-matched-trained model
    learns: E_S[Y]) and B outputs 3/8.
    """
    P = {(1, 1): F(3, 8), (1, 0): F(1, 8), (0, 1): F(1, 8), (0, 0): F(3, 8)}
    W1 = World("MCAR", {k: F(1, 2) for k in P})
    W2 = World("label-dependent", {(1, 1): F(2, 3), (1, 0): F(0), (0, 1): F(1, 2), (0, 0): F(1, 2)})
    pyx = {x: P[(x, 1)] / (P[(x, 1)] + P[(x, 0)]) for x in (0, 1)}  # source P(Y=1|X=x)
    cA, cB = F(1, 2), F(3, 8)

    def observed_law(W):
        return {("obs", x): sum(P[(x, y)] * W.r[(x, y)] for y in (0, 1)) for x in (0, 1)} | {
            ("mis",): sum(P[k] * (1 - W.r[k]) for k in P)}

    def target_brier(W, c_missing):
        risk = F(0)
        for (x, y), pxy in P.items():
            risk += pxy * W.r[(x, y)] * (y - pyx[x]) ** 2          # observed pattern
            risk += pxy * (1 - W.r[(x, y)]) * (y - c_missing) ** 2  # missing pattern
        return risk

    def dropout_brier(q_missing, c_missing):
        # artificial mask on the fully observed source, independent of (X, Y)
        risk = F(0)
        for (x, y), pxy in P.items():
            risk += pxy * (1 - q_missing) * (y - pyx[x]) ** 2 + pxy * q_missing * (y - c_missing) ** 2
        return risk

    out = {}
    for W in (W1, W2):
        pm = observed_law(W)[("mis",)]
        out[W.name] = {
            "observed_target_law": {str(k): v for k, v in observed_law(W).items()},
            "P_T(Y=1|M=0)": sum(P[k] * (1 - W.r[k]) for k in P if k[1] == 1) / pm,
            "R_T(A)": target_brier(W, cA),
            "R_T(B)": target_brier(W, cB),
            "R_drop_rate_matched(A)": dropout_brier(pm, cA),
            "R_drop_rate_matched(B)": dropout_brier(pm, cB),
        }
    return out


def ce2():
    """Population selection S (complete-case source) vs mask M under MAR.

    Always-observed X0 in {0,1}, modality X1 in {0,1}, Y.  In the deployment
    population P(X0=1)=1/2 and the modality is recorded with P(M=1|X0) =
    (1/10, 9/10) -- MAR given X0, no label dependence.  The development cohort
    is the complete cases {M=1}, so P_S(X0=1) = 9/10: S is a function of M.
    Y depends on X0 and X1.  Both models use P(Y|X0,X1) when X1 is recorded
    and a constant fallback when it is not: B falls back to the complete-case
    marginal P_S(Y=1)=7/10 (what a dropout-trained constant fallback learns on
    the complete cases), A to 3/10.  Rate-matched dropout on the complete cases
    ranks the fallbacks on the S-population (90% X0=1), the deployment ranks
    them on the missing-pattern population (90% X0=0); reweighting source units
    by P_T(x0)/P_S(x0) and masking them with P_T(M|x0) restores the target
    risk exactly, because no label dependence was assumed.
    """
    pT0 = {0: F(1, 2), 1: F(1, 2)}
    r = {0: F(1, 10), 1: F(9, 10)}                   # P(M=1 | X0)
    p1 = {0: F(1, 2), 1: F(1, 2)}                    # P(X1=1 | X0)
    py = {(0, 0): F(1, 20), (0, 1): F(9, 20), (1, 0): F(11, 20), (1, 1): F(19, 20)}  # P(Y=1|X0,X1)
    # complete-case source
    zS = sum(pT0[a] * r[a] for a in (0, 1))
    pS0 = {a: pT0[a] * r[a] / zS for a in (0, 1)}
    pY_x0 = {a: p1[a] * py[(a, 1)] + (1 - p1[a]) * py[(a, 0)] for a in (0, 1)}
    pS_y = sum(pS0[a] * pY_x0[a] for a in (0, 1))
    full = {(a, b): py[(a, b)] for a in (0, 1) for b in (0, 1)}

    def pred(model, a, b, observed):
        if observed:
            return full[(a, b)]
        return F(3, 10) if model == "A" else pS_y

    def unit_brier(p, f):
        return p * (1 - f) ** 2 + (1 - p) * f ** 2

    def risk(model, pop0, mprob):
        """mprob(a) = probability that X1 is recorded for a unit with X0=a."""
        tot = F(0)
        for a in (0, 1):
            for b in (0, 1):
                w = pop0[a] * (p1[a] if b else 1 - p1[a])
                p = py[(a, b)]
                tot += w * mprob(a) * unit_brier(p, pred(model, a, b, True))
                tot += w * (1 - mprob(a)) * unit_brier(p, pred(model, a, b, False))
        return tot

    pmis_T = sum(pT0[a] * (1 - r[a]) for a in (0, 1))
    assert pS_y == F(7, 10)
    res = {
        "P_T(Y=1|M=0)": sum(pT0[a] * (1 - r[a]) * pY_x0[a] for a in (0, 1)) / pmis_T,
        "P_S(X0=1)": pS0[1], "P_T(X0=1)": pT0[1], "P_T(M=0)": pmis_T,
        "R_T(A)": risk("A", pT0, lambda a: r[a]), "R_T(B)": risk("B", pT0, lambda a: r[a]),
        "R_drop_rate_matched(A)": risk("A", pS0, lambda a: 1 - pmis_T),
        "R_drop_rate_matched(B)": risk("B", pS0, lambda a: 1 - pmis_T),
    }
    # identified correction under MAR: reweight source units by P_T(x0)/P_S(x0),
    # and mask them with P_T(M|x0)
    res["R_reweighted(A)"] = risk("A", pT0, lambda a: r[a])
    res["R_reweighted(B)"] = risk("B", pT0, lambda a: r[a])
    return res


# --------------------------------------------------------------------------
# 2. Decomposition of the dropout-vs-deployment gap
# --------------------------------------------------------------------------

def gap_decomposition(pS, pT_m, qm, pT_given, lossA):
    r"""Exact three-term decomposition on a finite space.

    Arguments (all numpy arrays over a finite covariate grid x of the observed
    part of each pattern):
      pS[m]      : source covariate law of X_o(m), shape (n_x,)
      pT_m[m]    : target covariate law of X_o(m) given M=m
      qm[m]      : (artificial q(m), target P_T(M=m))
      pT_given[m]: (p_S(x)=P_S(Y=1|x), p_T(x)=P_T(Y=1|x,M=m))
      lossA[m]   : (l(f,0)(x), l(f,1)(x)) for the model on pattern m

    R_T - R_D = sum_m (P_T(m) - q(m)) E_S[l_m]                          (weights)
              + sum_m P_T(m) sum_x (pT_m - pS)(x) lbar_S(x)                (covariates)
              + sum_m P_T(m) sum_x pT_m(x) (p_T - p_S)(x) (l1 - l0)(x)     (labels)
    """
    RT = RD = w = c = lab = 0.0
    for m in pS:
        q, pt = qm[m]
        ps, ptx = pT_given[m]
        l0, l1 = lossA[m]
        lbarS = l0 + ps * (l1 - l0)
        lbarT = l0 + ptx * (l1 - l0)
        ES = float(np.sum(pS[m] * lbarS))
        RD += q * ES
        RT += pt * float(np.sum(pT_m[m] * lbarT))
        w += (pt - q) * ES
        c += pt * float(np.sum((pT_m[m] - pS[m]) * lbarS))
        lab += pt * float(np.sum(pT_m[m] * (ptx - ps) * (l1 - l0)))
    return {"R_T": RT, "R_D": RD, "weights": w, "covariates": c, "labels": lab}


# --------------------------------------------------------------------------
# 3. Sensitivity bounds on a risk DIFFERENCE
# --------------------------------------------------------------------------

def or_interval(p, gamma):
    """{q : odds(q)/odds(p) in [1/gamma, gamma]} for p in (0,1)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    o = p / (1 - p)
    lo = o / gamma / (1 + o / gamma)
    hi = o * gamma / (1 + o * gamma)
    return lo, hi


def diff_bounds(a0, a1, b0, b1, ps, gamma, weights=None, direction=None):
    r"""Sharp bounds on  E[ l_A - l_B ]  when each unit's label probability q_i
    may lie anywhere in the odds-ratio band around ps_i (optionally one-sided).

    d_i(q) = (a0 - b0)_i + q_i * g_i,   g_i = (a1 - a0) - (b1 - b0).
    Linear in q_i, separable across units, so the extreme is attained at an
    endpoint of each unit's interval: sharp without further constraints.

    direction: None (two-sided), "down" (q <= ps: missing-pattern units are no
    riskier than complete cases with the same observed features), "up".
    """
    a0, a1, b0, b1, ps = map(lambda v: np.asarray(v, dtype=float), (a0, a1, b0, b1, ps))
    w = np.full(len(ps), 1.0 / len(ps)) if weights is None else np.asarray(weights, float) / np.sum(weights)
    lo, hi = or_interval(ps, gamma)
    if direction == "down":
        hi = np.minimum(hi, ps)
        lo = np.minimum(lo, ps)
    elif direction == "up":
        lo = np.maximum(lo, ps)
        hi = np.maximum(hi, ps)
    c = a0 - b0
    g = (a1 - a0) - (b1 - b0)
    point = float(np.sum(w * (c + ps * g)))
    upper = float(np.sum(w * (c + np.where(g > 0, hi, lo) * g)))
    lower = float(np.sum(w * (c + np.where(g > 0, lo, hi) * g)))
    return lower, point, upper


def diff_bounds_prevalence(a0, a1, b0, b1, ps, gamma, prevalence, weights=None):
    r"""Sharp bounds with the extra linear constraint  sum_i w_i q_i = prevalence.

    Maximise sum_i w_i g_i q_i over the box, one equality: a continuous
    knapsack.  Start every q_i at its lower end, then raise the units with the
    largest g_i first until the prevalence budget is spent (and symmetrically for
    the minimum).  Returns (lower, upper) or (nan, nan) if infeasible.
    """
    a0, a1, b0, b1, ps = map(lambda v: np.asarray(v, dtype=float), (a0, a1, b0, b1, ps))
    n = len(ps)
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, float) / np.sum(weights)
    lo, hi = or_interval(ps, gamma)
    c = float(np.sum(w * (a0 - b0)))
    g = (a1 - a0) - (b1 - b0)
    if not (np.sum(w * lo) - 1e-12 <= prevalence <= np.sum(w * hi) + 1e-12):
        return float("nan"), float("nan")

    def extreme(sign):
        q = lo.copy()
        budget = prevalence - float(np.sum(w * lo))
        for i in np.argsort(-sign * g):
            if budget <= 0:
                break
            room = (hi[i] - lo[i]) * w[i]
            take = min(room, budget)
            q[i] += take / w[i]
            budget -= take
        return c + float(np.sum(w * g * q))

    return extreme(-1.0), extreme(+1.0)


def breakdown_gamma(a0, a1, b0, b1, ps, weights=None, direction=None, gmax=50.0):
    """Smallest gamma at which the bound interval on E[l_A - l_B] contains 0."""
    lo1, pt, up1 = diff_bounds(a0, a1, b0, b1, ps, 1.0, weights, direction)
    if lo1 <= 0 <= up1:
        return 1.0
    f = (lambda gm: diff_bounds(a0, a1, b0, b1, ps, gm, weights, direction)[2]) if pt < 0 else \
        (lambda gm: -diff_bounds(a0, a1, b0, b1, ps, gm, weights, direction)[0])
    if f(gmax) < 0:
        return float("inf")
    a, b = 1.0, gmax
    for _ in range(60):
        mid = np.sqrt(a * b)
        if f(mid) < 0:
            a = mid
        else:
            b = mid
    return float(b)


# --------------------------------------------------------------------------
# numerical certification
# --------------------------------------------------------------------------

def verify(seed: int = 0, n_rand: int = 300) -> dict:
    rng = np.random.default_rng(seed)
    report = {}

    # CE1
    c1 = ce1()
    w1, w2 = c1["MCAR"], c1["label-dependent"]
    report["ce1_same_observed_law"] = w1["observed_target_law"] == w2["observed_target_law"]
    report["ce1_same_dropout_estimate"] = (w1["R_drop_rate_matched(A)"] == w2["R_drop_rate_matched(A)"]
                                           and w1["R_drop_rate_matched(B)"] == w2["R_drop_rate_matched(B)"])
    report["ce1_dropout_prefers_A"] = w1["R_drop_rate_matched(A)"] < w1["R_drop_rate_matched(B)"]
    report["ce1_W1_prefers_A"] = w1["R_T(A)"] < w1["R_T(B)"]
    report["ce1_W2_prefers_B"] = w2["R_T(B)"] < w2["R_T(A)"]
    report["ce1_values"] = {k: {kk: str(vv) for kk, vv in v.items() if kk != "observed_target_law"} for k, v in c1.items()}

    # CE2
    c2 = ce2()
    report["ce2_dropout_prefers_B"] = c2["R_drop_rate_matched(B)"] < c2["R_drop_rate_matched(A)"]
    report["ce2_target_prefers_A"] = c2["R_T(A)"] < c2["R_T(B)"]
    report["ce2_reweighting_recovers"] = (c2["R_reweighted(A)"] == c2["R_T(A)"] and c2["R_reweighted(B)"] == c2["R_T(B)"])
    report["ce2_values"] = {k: str(v) for k, v in c2.items()}

    # decomposition identity on random finite instances
    worst = 0.0
    for _ in range(n_rand):
        nx = rng.integers(2, 6)
        pats = range(rng.integers(1, 4))
        pS = {m: rng.dirichlet(np.ones(nx)) for m in pats}
        pT = {m: rng.dirichlet(np.ones(nx)) for m in pats}
        qs = rng.dirichlet(np.ones(len(pS)))
        ts = rng.dirichlet(np.ones(len(pS)))
        qm = {m: (qs[i], ts[i]) for i, m in enumerate(pats)}
        pg = {m: (rng.uniform(.05, .95, nx), rng.uniform(.05, .95, nx)) for m in pats}
        la = {m: loss_pair(rng.uniform(.05, .95, nx), "logloss") for m in pats}
        d = gap_decomposition(pS, pT, qm, pg, la)
        worst = max(worst, abs((d["R_T"] - d["R_D"]) - (d["weights"] + d["covariates"] + d["labels"])))
    report["decomposition_max_abs_error"] = worst

    # bounds: sharpness & validity against LP and random feasible points
    viol = 0
    lp_gap = 0.0
    prev_gap = 0.0
    for _ in range(n_rand):
        n = int(rng.integers(3, 12))
        ps = rng.uniform(.02, .98, n)
        fa, fb = rng.uniform(.02, .98, n), rng.uniform(.02, .98, n)
        loss = "brier" if rng.random() < .5 else "logloss"
        a0, a1 = loss_pair(fa, loss)
        b0, b1 = loss_pair(fb, loss)
        gam = float(rng.uniform(1, 5))
        w = rng.dirichlet(np.ones(n))
        lo, pt, up = diff_bounds(a0, a1, b0, b1, ps, gam, w)
        ql, qh = or_interval(ps, gam)
        for _k in range(50):
            q = rng.uniform(ql, qh)
            v = float(np.sum(w * ((a0 - b0) + q * ((a1 - a0) - (b1 - b0)))))
            viol += not (lo - 1e-10 <= v <= up + 1e-10)
        cvec = w * ((a1 - a0) - (b1 - b0))
        const = float(np.sum(w * (a0 - b0)))
        for sgn, ref in ((1, lo), (-1, up)):
            r = linprog(sgn * cvec, bounds=list(zip(ql, qh)), method="highs")
            lp_gap = max(lp_gap, abs(sgn * r.fun + const - ref))
        prev = float(np.sum(w * rng.uniform(ql, qh)))
        plo, pup = diff_bounds_prevalence(a0, a1, b0, b1, ps, gam, prev, w)
        for sgn, ref in ((1, plo), (-1, pup)):
            r = linprog(sgn * cvec, A_eq=w[None, :], b_eq=[prev], bounds=list(zip(ql, qh)), method="highs")
            prev_gap = max(prev_gap, abs(sgn * r.fun + const - ref))
    report["bounds_violations_random_feasible"] = int(viol)
    report["bounds_max_gap_vs_LP"] = lp_gap
    report["prevalence_bounds_max_gap_vs_LP"] = prev_gap

    # gamma = 1 collapses to the point estimate; bounds are nested in gamma
    ps = rng.uniform(.05, .95, 20)
    a0, a1 = loss_pair(rng.uniform(.05, .95, 20))
    b0, b1 = loss_pair(rng.uniform(.05, .95, 20))
    l1, p1, u1 = diff_bounds(a0, a1, b0, b1, ps, 1.0)
    report["gamma1_collapses"] = abs(l1 - p1) < 1e-12 and abs(u1 - p1) < 1e-12
    nested = True
    prev = (l1, u1)
    for gm in (1.2, 1.5, 2, 3, 5, 10):
        l, _, u = diff_bounds(a0, a1, b0, b1, ps, gm)
        nested &= l <= prev[0] + 1e-12 and u >= prev[1] - 1e-12
        prev = (l, u)
    report["bounds_nested_in_gamma"] = bool(nested)

    report["all_ok"] = bool(
        report["ce1_same_observed_law"] and report["ce1_same_dropout_estimate"] and report["ce1_dropout_prefers_A"]
        and report["ce1_W1_prefers_A"] and report["ce1_W2_prefers_B"] and report["ce2_dropout_prefers_B"]
        and report["ce2_target_prefers_A"] and report["ce2_reweighting_recovers"]
        and report["decomposition_max_abs_error"] < 1e-12 and report["bounds_violations_random_feasible"] == 0
        and report["bounds_max_gap_vs_LP"] < 1e-7 and report["prevalence_bounds_max_gap_vs_LP"] < 1e-7
        and report["gamma1_collapses"] and report["bounds_nested_in_gamma"])
    return report


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    rep = verify()
    out = Path(__file__).resolve().parents[2] / "results/theory"
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    sys.exit(0 if rep["all_ok"] else 1)
