"""Candidate predictors: every missing-data strategy on the same data and budget.

All candidates of one run are fitted on the same labeled development set
``D`` (natural masks) and may read the same unlabeled deployment sample ``U``
(X_obs, M; no labels).  Complete-case (``cc_*``) strategies restrict ``D`` to
its fully observed units, i.e. the typical "paired multimodal cohort" model;
natural (``nat_*``) strategies use every unit of ``D`` with its natural mask.
Each strategy is one fixed configuration per learner -- no strategy gets
tuning the others do not get.

Strategies (per learner: logistic regression ``lr`` and gradient boosting ``hgb``):
  cc_imp_mean   full-feature model; missing panels filled with training means
  cc_imp_reg    full-feature model; missing panels imputed by their linear
                conditional mean given the observed columns (fitted on D_cc)
  cc_obs        observed-only pattern submodels, one per mask pattern
  cc_drop       modality dropout training (each panel dropped w.p. 1/2), mean fill
  cc_drop_mask  same, plus the mask as input (mask-aware)
  cc_dams       Zhou-Balakrishnan-Lipton (AISTATS 2023) Algorithm 1 with a fully
                observed source: re-mask each panel independently at the target's
                marginal missing rate estimated from U, mean fill, no indicators
  cc_vmar       re-mask from P_U(M | x0) (v-MAR, their Prop. 1 setting) + mask input
  cc_vmar_rw    cc_vmar trained with importance weights P_U(x0)/P_Dcc(x0)
  nat_imp_mean  natural D, mean fill, no indicators (missingness-unconditional)
  nat_imp_ice   natural D, iterative (chained-equation) imputation, no indicators
  nat_mask      natural D, mean fill + mask input (missingness-conditional)
  nat_obs       observed-only pattern submodels, each fitted on units whose
                pattern contains it (shared-pattern / reduced-model approach)
  nat_nan       (hgb only) natural D, native NaN routing in the trees
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression

N_REP = 5  # dropout / re-masking replicates of the training set


def patterns(K):
    return np.array([[(m >> k) & 1 for k in range(K)] for m in range(1 << K)], dtype=bool)


def pattern_id(M):
    M = np.asarray(M, dtype=int)
    return (M * (1 << np.arange(M.shape[1]))).sum(1)


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

@dataclass
class Spec:
    base_idx: list
    panel_idx: list
    groups: dict  # name -> base column indices for base-missing indicators

    @property
    def K(self):
        return len(self.panel_idx)

    def obs_cols(self, m_bits):
        cols = list(self.base_idx)
        for k, on in enumerate(m_bits):
            if on:
                cols += list(self.panel_idx[k])
        return cols


class Prep:
    """Standardise with training statistics; base NaN -> training median + group
    indicators; masked panel columns -> 0 (the training mean)."""

    def __init__(self, spec: Spec):
        self.spec = spec

    def fit(self, X):
        self.med = np.nanmedian(X, axis=0)
        self.mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        self.sd = np.where(sd > 1e-9, sd, 1.0)
        return self

    def base_block(self, X):
        B = X[:, self.spec.base_idx].copy()
        miss = np.isnan(B)
        B = np.where(miss, self.med[self.spec.base_idx], B)
        B = (B - self.mu[self.spec.base_idx]) / self.sd[self.spec.base_idx]
        pos = {c: j for j, c in enumerate(self.spec.base_idx)}
        ind = np.stack([np.isnan(X[:, g]).any(1) for g in self.spec.groups.values()], 1).astype(float)
        return np.hstack([B, ind])

    def full(self, X, M, indicators=False):
        """All columns; masked panels set to 0 after standardisation."""
        Z = [self.base_block(X)]
        for k, idx in enumerate(self.spec.panel_idx):
            P = (X[:, idx] - self.mu[idx]) / self.sd[idx]
            P = np.where(M[:, [k]] & ~np.isnan(P), P, 0.0)
            Z.append(P)
        if indicators:
            Z.append(M.astype(float))
        return np.hstack(Z)

    def sub(self, X, m_bits):
        Z = [self.base_block(X)]
        for k, idx in enumerate(self.spec.panel_idx):
            if m_bits[k]:
                Z.append(np.nan_to_num((X[:, idx] - self.mu[idx]) / self.sd[idx]))
        return np.hstack(Z)

    def x0(self, X):
        return self.base_block(X)


def make_learner(kind, seed):
    if kind == "lr":
        return LogisticRegression(C=1.0, max_iter=5000)
    if kind == "hgb":
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                              min_samples_leaf=30, l2_regularization=1.0,
                                              early_stopping=False, random_state=seed)
    raise ValueError(kind)


def _fit(kind, Z, y, seed, w=None):
    mdl = make_learner(kind, seed)
    if len(np.unique(y)) < 2:
        return ("const", float(np.mean(y)))
    mdl.fit(Z, y, sample_weight=w)
    return mdl


def _pred(mdl, Z):
    if isinstance(mdl, tuple):
        return np.full(len(Z), mdl[1])
    return mdl.predict_proba(Z)[:, 1]


# --------------------------------------------------------------------------
# auxiliary models fitted on the unlabeled deployment sample
# --------------------------------------------------------------------------

def mask_model(prep, XU, MU):
    """P_U(pattern | x0): multinomial logistic regression over observed patterns."""
    pid = pattern_id(MU)
    Z = prep.x0(XU)
    mdl = LogisticRegression(C=1.0, max_iter=5000)
    mdl.fit(Z, pid)
    return mdl


def sample_masks(rng, K, n, how, prep=None, X=None, mm=None, rate=None, freq=None):
    P = patterns(K)
    if how == "half":
        return rng.random((n, K)) < 0.5
    if how == "rate":  # independent per panel at target marginal availability
        return rng.random((n, K)) < rate[None, :]
    if how == "freq":
        return P[rng.choice(len(P), size=n, p=freq)]
    if how == "vmar":
        pr = mm.predict_proba(prep.x0(X))
        full = np.zeros((n, len(P)))
        full[:, mm.classes_] = pr
        cum = full.cumsum(1)
        u = rng.random((n, 1))
        return P[np.minimum((u > cum).sum(1), len(P) - 1)]
    raise ValueError(how)


def domain_weights(ZS, ZT, clip_q=0.99):
    """w(z) proportional to P_T(z)/P_S(z) on the S sample, via a logistic domain classifier."""
    Z = np.vstack([ZS, ZT])
    d = np.r_[np.zeros(len(ZS)), np.ones(len(ZT))]
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(Z, d)
    return clf


def weights_from(clf, Z, nS, nT, clip_q=0.99):
    p = np.clip(clf.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
    w = p / (1 - p) * nS / nT
    w = np.minimum(w, np.quantile(w, clip_q))
    return w / w.mean()


# --------------------------------------------------------------------------
# candidate strategies
# --------------------------------------------------------------------------

class Candidate:
    def __init__(self, name, learner, strategy, prep, K):
        self.name, self.learner, self.strategy, self.prep, self.K = name, learner, strategy, prep, K

    # fitted parts are attached by fit_candidates
    def predict(self, X, M):
        s = self.strategy
        M = np.asarray(M, dtype=bool)
        if s in ("cc_obs", "nat_obs"):
            out = np.empty(len(X))
            pid = pattern_id(M)
            for m in np.unique(pid):
                bits = patterns(self.K)[m]
                sel = pid == m
                out[sel] = _pred(self.sub[m], self.prep.sub(X[sel], bits))
            return out
        if s == "cc_imp_reg":
            Xi = self.impute_reg(X, M)
            return _pred(self.mdl, self.prep.full(Xi, np.ones_like(M)))
        if s == "nat_imp_ice":
            Xm = X.copy()
            for k, idx in enumerate(self.prep.spec.panel_idx):
                Xm[np.ix_(~M[:, k], idx)] = np.nan
            Xi = self.ice.transform(Xm)
            return _pred(self.mdl, self.prep.full(Xi, np.ones_like(M)))
        if s == "nat_nan":
            Xm = X.copy()
            for k, idx in enumerate(self.prep.spec.panel_idx):
                Xm[np.ix_(~M[:, k], idx)] = np.nan
            return _pred(self.mdl, Xm)
        ind = s in ("cc_drop_mask", "cc_vmar", "cc_vmar_rw", "nat_mask")
        return _pred(self.mdl, self.prep.full(X, M, indicators=ind))

    def impute_reg(self, X, M):
        Xi = X.copy()
        pid = pattern_id(M)
        for m in np.unique(pid):
            bits = patterns(self.K)[m]
            if bits.all():
                continue
            sel = pid == m
            obs = self.prep.spec.obs_cols(bits)
            mis = [c for k in range(self.K) if not bits[k] for c in self.prep.spec.panel_idx[k]]
            reg = self.reg[m]
            Zo = np.where(np.isnan(X[np.ix_(sel, obs)]), self.prep.med[obs], X[np.ix_(sel, obs)])
            Xi[np.ix_(sel, mis)] = reg.predict(Zo)
        return Xi


def fit_candidates(spec, XD, MD, yD, XU, MU, seed=0, learners=("lr", "hgb")):
    """Fit every strategy for every learner.  Returns list[Candidate]."""
    rng = np.random.default_rng(seed)
    K = spec.K
    P = patterns(K)
    cc = MD.all(1)
    Xc, yc = XD[cc], yD[cc]
    prep_cc = Prep(spec).fit(Xc)
    Xn = XD.copy()
    for k, idx in enumerate(spec.panel_idx):
        Xn[np.ix_(~MD[:, k], idx)] = np.nan
    prep_nat = Prep(spec).fit(Xn)

    rate = MU.mean(0)
    pidU = pattern_id(MU)
    freq = np.bincount(pidU, minlength=len(P)) / len(pidU)
    mm = mask_model(prep_cc, XU, MU)
    dclf = domain_weights(prep_cc.x0(Xc), prep_cc.x0(XU))
    w_cc = weights_from(dclf, prep_cc.x0(Xc), len(Xc), len(XU))

    # regression imputers on complete cases, one per pattern
    regs = {}
    for m in range(len(P)):
        bits = P[m]
        if bits.all():
            continue
        obs = spec.obs_cols(bits)
        mis = [c for k in range(K) if not bits[k] for c in spec.panel_idx[k]]
        regs[m] = LinearRegression().fit(np.where(np.isnan(Xc[:, obs]), prep_cc.med[obs], Xc[:, obs]), Xc[:, mis])

    ice = IterativeImputer(max_iter=5, random_state=seed, sample_posterior=False).fit(Xn)

    def rep(n):
        return np.tile(np.arange(n), N_REP)

    out = []
    for L in learners:
        def add(strategy, **parts):
            c = Candidate(f"{L}:{strategy}", L, strategy, parts.pop("prep"), K)
            for k, v in parts.items():
                setattr(c, k, v)
            out.append(c)

        ones = np.ones_like(MD[cc])
        add("cc_imp_mean", prep=prep_cc, mdl=_fit(L, prep_cc.full(Xc, ones), yc, seed))
        add("cc_imp_reg", prep=prep_cc, mdl=_fit(L, prep_cc.full(Xc, ones), yc, seed), reg=regs)
        add("cc_obs", prep=prep_cc, sub={m: _fit(L, prep_cc.sub(Xc, P[m]), yc, seed) for m in range(len(P))})
        r = rep(len(Xc))
        for strategy, how, ind, w in (("cc_drop", "half", False, None), ("cc_drop_mask", "half", True, None),
                                      ("cc_dams", "rate", False, None), ("cc_vmar", "vmar", True, None),
                                      ("cc_vmar_rw", "vmar", True, w_cc)):
            Ma = sample_masks(rng, K, len(r), how, prep=prep_cc, X=Xc[r], mm=mm, rate=rate, freq=freq)
            Z = prep_cc.full(Xc[r], Ma, indicators=ind)
            add(strategy, prep=prep_cc, mdl=_fit(L, Z, yc[r], seed, None if w is None else w[r]))
        # natural development data
        add("nat_imp_mean", prep=prep_nat, mdl=_fit(L, prep_nat.full(Xn, MD), yD, seed))
        add("nat_imp_ice", prep=prep_nat, ice=ice,
            mdl=_fit(L, prep_nat.full(ice.transform(Xn), np.ones_like(MD)), yD, seed))
        add("nat_mask", prep=prep_nat, mdl=_fit(L, prep_nat.full(Xn, MD, indicators=True), yD, seed))
        sub = {}
        for m in range(len(P)):
            sup = (MD | ~P[m][None, :]).all(1)  # units whose pattern contains m
            sub[m] = _fit(L, prep_nat.sub(Xn[sup], P[m]), yD[sup], seed)
        add("nat_obs", prep=prep_nat, sub=sub)
        if L == "hgb":
            add("nat_nan", prep=prep_nat, mdl=_fit(L, Xn, yD, seed))
    return out
