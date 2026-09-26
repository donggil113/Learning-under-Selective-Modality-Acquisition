"""Pipeline tests on a small synthetic cohort (no real data needed)."""
import numpy as np

from src.n1.evaluators import Evaluation, agreement
from src.n1.models import Spec, fit_candidates, pattern_id, patterns


def toy(n, rng, delta=0.0):
    K = 2
    x0 = rng.normal(size=(n, 3))
    x1 = x0[:, :1] + rng.normal(size=(n, 2))
    x2 = rng.normal(size=(n, 2))
    lin = x0[:, 0] + 0.8 * x1[:, 0] + 0.8 * x2[:, 0] - 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    X = np.hstack([x0, x1, x2])
    pi = 1 / (1 + np.exp(-(0.5 + 0.8 * x0[:, [0]] + delta * y[:, None])))
    M = rng.random((n, K)) < pi
    spec = Spec(base_idx=[0, 1, 2], panel_idx=[[3, 4], [5, 6]], groups={"g": [0]})
    return X, M, y, spec


def masked(X, M, spec):
    Xm = X.copy()
    for k, idx in enumerate(spec.panel_idx):
        Xm[np.ix_(~M[:, k], idx)] = np.nan
    return Xm


def test_candidates_and_evaluators_run_and_rank_sensibly():
    rng = np.random.default_rng(0)
    X, M, y, spec = toy(3000, rng)
    D, P, Tt = np.arange(0, 1200), np.arange(1200, 2200), np.arange(2200, 3000)
    Xm = masked(X, M, spec)
    cands = fit_candidates(spec, Xm[D], M[D], y[D], Xm[P], M[P], seed=0, learners=("lr",))
    assert len(cands) == 14
    V = P[M[P].all(1)]
    ev = Evaluation(cands, spec, X[V], y[V], Xm[P], M[P], Xm[Tt], M[Tt], y[Tt], yU=y[P], seed=0,
                    v_in_u=np.where(M[P].all(1))[0])
    est = ev.estimates("brier")
    tru = ev.truth("brier")
    for e in ("full", "drop50", "freq", "sel", "pat", "dr"):
        assert np.all(np.isfinite(est[e]))
        a = agreement(est[e], tru)
        assert -1 <= a["kendall_tau"] <= 1
    lo, pt, up = ev.aa_pair(0, 1, "brier", 2.0)
    assert lo <= pt <= up
    lo1, pt1, up1 = ev.aa_pair(0, 1, "brier", 1.0)
    assert abs(lo1 - up1) < 1e-12


def test_replicated_fit_matches_unreplicated_regularisation():
    """Stacking N_REP identical copies with weight 1/N_REP must equal a single fit (LR)."""
    from src.n1.models import N_REP, _fit
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(300, 4))
    y = (rng.random(300) < 1 / (1 + np.exp(-Z[:, 0]))).astype(int)
    a = _fit("lr", Z, y, 0).coef_
    b = _fit("lr", np.tile(Z, (N_REP, 1)), np.tile(y, N_REP), 0, np.full(300 * N_REP, 1 / N_REP), rep=N_REP).coef_
    assert np.allclose(a, b, atol=1e-4)


def test_predictions_use_only_observed_panels():
    rng = np.random.default_rng(1)
    X, M, y, spec = toy(1500, rng)
    Xm = masked(X, M, spec)
    cands = fit_candidates(spec, Xm[:800], M[:800], y[:800], Xm[800:], M[800:], seed=0, learners=("lr",))
    Xp = X[800:900].copy()
    Mp = np.zeros((100, 2), bool)
    garbage = Xp.copy()
    garbage[:, 3:] = 1e6  # values in unobserved panels must be ignored
    for c in cands:
        assert np.allclose(c.predict(Xp, Mp), c.predict(garbage, Mp)), c.name
