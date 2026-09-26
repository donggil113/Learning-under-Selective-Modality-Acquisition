"""Selection rules: estimate every candidate's deployment risk from
  V : a labeled, FULLY OBSERVED validation cohort (the complete cases), and
  U : an unlabeled deployment sample (X_obs, M) with natural masks,
and score them against the truth T (labeled deployment units, natural masks).

No rule reads a label outside V except ``lab_n`` (an explicit n-label budget
drawn from U), which is reported as a different information regime.

  full         V, nothing masked
  drop50       V, each panel dropped independently w.p. 1/2 (standard robustness eval)
  dams_rate    V, each panel dropped independently at U's marginal missing rate
               (Zhou et al. 2023, Alg. 1, used as an evaluation distribution)
  freq         V, patterns drawn from U's joint pattern frequencies
  freq_prior   freq + class-balance weights to a target prevalence estimated by
               EM prior adjustment (Saerens et al. 2002) on U (assumes label shift)
  freq_prior_plugin  same, prevalence = mean complete-case-model prediction on U
               (assumes Y independent of M given X_o(m))
  freq_prior_oracle  freq + class-balance weights to the TRUE deployment
               prevalence (upper bound on what class balance can explain)
  sel          V weighted by P_U(x0)/P_V(x0), masks from P_U(M | x0)
               (sample-selection adjustment on always-recorded covariates;
               the v-MAR importance-weighting of Zhou et al. Prop. 1)
  pat          per-pattern covariate-shift IW on the pattern's observed features
               x_o(m): identified iff Y is independent of M given X_o(m) (Gamma = 1)
  dr           doubly robust version of pat with a pattern outcome model fitted on V
  aa(Gamma)    acquisition-aware: sharp bounds on each pairwise risk DIFFERENCE when
               P_T(Y | x_o(m), M=m) may differ from the complete-case P(Y | x_o(m))
               by an odds ratio in [1/Gamma, Gamma]
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import HistGradientBoostingClassifier

from .models import Prep, domain_weights, pattern_id, patterns, weights_from


def outcome_model(family, seed=0):
    if family == "lr":
        return LogisticRegression(C=1.0, max_iter=5000)
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=40,
                                          l2_regularization=1.0, early_stopping=False, random_state=seed)


def raw_weight_mean(clf, Z, nS, nT):
    """Mean of the UNclipped, UNnormalised density-ratio estimate on the source
    sample; about 1 under overlap, near 0 when the target has no source support."""
    p = np.clip(clf.predict_proba(Z)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.mean(p / (1 - p) * nS / nT))
from .theory import diff_bounds, or_interval


def loss01(p, loss):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if loss == "brier":
        return p ** 2, (1 - p) ** 2
    return -np.log(1 - p), -np.log(p)


def unit_loss(p, y, loss):
    l0, l1 = loss01(p, loss)
    return np.where(y == 1, l1, l0)


class Evaluation:
    """Precomputes predictions of every candidate on V (all patterns), U, T."""

    def __init__(self, cands, spec, XV, yV, XU, MU, XT, MT, yT, yU=None, seed=0, v_in_u=None):
        """``v_in_u``: positions in U of the V units (V is a subset of U in the
        complete-case design); their outcome-model predictions are then taken
        from the cross-fitted values so no V label leaks into U-side estimates."""
        self.cands, self.spec = cands, spec
        self.v_in_u = v_in_u
        self.K = spec.K
        self.P = patterns(self.K)
        self.nP = len(self.P)
        self.XV, self.yV, self.XU, self.MU, self.XT, self.MT, self.yT, self.yU = XV, yV, XU, MU, XT, MT, yT, yU
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        J = len(cands)
        self.names = [c.name for c in cands]
        nV = len(XV)
        self.PV = np.empty((J, self.nP, nV))
        for m in range(self.nP):
            Mm = np.repeat(self.P[m][None, :], nV, 0)
            for j, c in enumerate(cands):
                self.PV[j, m] = c.predict(XV, Mm)
        self.PU = np.stack([c.predict(XU, MU) for c in cands])
        self.PT = np.stack([c.predict(XT, MT) for c in cands])
        self.pidU = pattern_id(MU)
        self.pidT = pattern_id(MT)
        self.full_id = self.nP - 1
        self.freq = np.bincount(self.pidU, minlength=self.nP) / len(self.pidU)
        rate = MU.mean(0)
        self.q_rate = np.prod(np.where(self.P, rate[None, :], 1 - rate[None, :]), axis=1)
        self.prep = Prep(spec).fit(XV)
        self._fit_aux()

    # ------------------------------------------------------------------
    def _fit_aux(self):
        prep, XV, XU = self.prep, self.XV, self.XU
        # sample selection on always-recorded covariates
        Z0V, Z0U = prep.x0(XV), prep.x0(XU)
        clf = domain_weights(Z0V, Z0U)
        self.w_sel = weights_from(clf, Z0V, len(Z0V), len(Z0U))
        self.w_sel_raw_mean = raw_weight_mean(clf, Z0V, len(Z0V), len(Z0U))
        mm = LogisticRegression(C=1.0, max_iter=5000).fit(Z0U, self.pidU)
        pr = np.zeros((len(XV), self.nP))
        pr[:, mm.classes_] = mm.predict_proba(Z0V)
        self.pm_sel = pr
        # per-pattern covariate weights and outcome models on x_o(m).
        # Outcome model = average of an LR and an HGB fit (not the function class
        # of any single candidate); per-family predictions are kept for sensitivity.
        self.w_pat = np.ones((self.nP, len(XV)))
        self.w_pat_raw_mean = np.ones(self.nP)
        fam = ("lr", "hgb")
        self.fam_pS_V_cf = {f: np.zeros((self.nP, len(XV))) for f in fam}
        self.fam_pS_U = {f: np.zeros(len(XU)) for f in fam}
        self.fam_pS_T = {f: np.zeros(len(self.XT)) for f in fam}
        folds = self.rng.integers(0, 5, len(XV))
        for m in range(self.nP):
            bits = self.P[m]
            ZV = prep.sub(XV, bits)
            selU = self.pidU == m
            if selU.sum() >= 5:
                ZU = prep.sub(XU[selU], bits)
                c = domain_weights(ZV, ZU)
                self.w_pat[m] = weights_from(c, ZV, len(ZV), len(ZU))
                self.w_pat_raw_mean[m] = raw_weight_mean(c, ZV, len(ZV), len(ZU))
            selT = self.pidT == m
            for f in fam:
                om = outcome_model(f, self.seed).fit(ZV, self.yV)
                for k in range(5):
                    tr, te = folds != k, folds == k
                    self.fam_pS_V_cf[f][m, te] = outcome_model(f, self.seed).fit(ZV[tr], self.yV[tr]).predict_proba(ZV[te])[:, 1]
                if selU.any():
                    self.fam_pS_U[f][selU] = om.predict_proba(prep.sub(XU[selU], bits))[:, 1]
                if selT.any():
                    self.fam_pS_T[f][selT] = om.predict_proba(prep.sub(self.XT[selT], bits))[:, 1]
            if self.v_in_u is not None and m == self.full_id:
                for f in fam:
                    self.fam_pS_U[f][self.v_in_u] = self.fam_pS_V_cf[f][self.full_id]
        self.pS_V_cf = np.mean([self.fam_pS_V_cf[f] for f in fam], 0)
        self.pS_U = np.mean([self.fam_pS_U[f] for f in fam], 0)
        self.pS_T = np.mean([self.fam_pS_T[f] for f in fam], 0)

    # ------------------------------------------------------------------
    def _pattern_risk(self, loss, wunit=None):
        """R[j, m] = weighted mean over V of loss of model j under pattern m."""
        LV = unit_loss(self.PV, self.yV[None, None, :], loss)
        if wunit is None:
            return LV.mean(2)
        w = np.asarray(wunit)
        if w.ndim == 1:
            return (LV * w[None, None, :]).sum(2) / w.sum()
        return (LV * w[None, :, :]).sum(2) / w.sum(1)[None, :]

    def em_prior(self, tol=1e-8):
        piS = self.yV.mean()
        p = np.clip(self.pS_U, 1e-6, 1 - 1e-6)
        pi = piS
        for _ in range(1000):
            a = pi / piS * p
            b = (1 - pi) / (1 - piS) * (1 - p)
            new = float(np.mean(a / (a + b)))
            if abs(new - pi) < tol:
                break
            pi = new
        return pi

    def estimates(self, loss):
        est = {}
        R = self._pattern_risk(loss)
        est["full"] = R[:, self.full_id]
        est["drop50"] = R.mean(1)
        est["dams_rate"] = R @ self.q_rate
        est["freq"] = R @ self.freq
        piS = self.yV.mean()
        for tag, pi in (("freq_prior", self.em_prior()), ("freq_prior_plugin", float(self.pS_U.mean())),
                        ("freq_prior_oracle", self.yT.mean())):
            wy = np.where(self.yV == 1, pi / piS, (1 - pi) / (1 - piS))
            est[tag] = self._pattern_risk(loss, wy) @ self.freq
        # selection on x0 with MAR-on-x0 masks
        LV = unit_loss(self.PV, self.yV[None, None, :], loss)            # J, P, nV
        est["sel"] = np.einsum("jmi,im,i->j", LV, self.pm_sel, self.w_sel) / self.w_sel.sum()
        # per-pattern IW
        Rpat = self._pattern_risk(loss, self.w_pat)
        est["pat"] = Rpat @ self.freq
        # doubly robust
        est["dr"] = self._dm(loss) + self._dr_correction(loss)
        est["dm"] = self._dm(loss)
        self.lab_budget = {}
        if self.yU is not None:
            for n in (50, 100, 200, 400):
                draws = []
                for _ in range(50):
                    idx = self.rng.choice(len(self.yU), n, replace=False)
                    draws.append(unit_loss(self.PU[:, idx], self.yU[None, idx], loss).mean(1))
                est[f"lab{n}"] = np.array(draws)  # (50, J)
        return est

    def _dm(self, loss):
        l0, l1 = loss01(self.PU, loss)
        return (l0 * (1 - self.pS_U) + l1 * self.pS_U).mean(1)

    def _dr_correction(self, loss):
        cache = self.__dict__.setdefault("_drc", {})
        if loss not in cache:
            cache[loss] = self._dr_correction_uncached(loss)
        return cache[loss]

    def _dr_correction_uncached(self, loss):
        LV = unit_loss(self.PV, self.yV[None, None, :], loss)
        l0, l1 = loss01(self.PV, loss)
        fit = l0 * (1 - self.pS_V_cf[None]) + l1 * self.pS_V_cf[None]
        res = ((LV - fit) * self.w_pat[None]).sum(2) / self.w_pat.sum(1)[None]
        return res @ self.freq

    def truth(self, loss):
        return unit_loss(self.PT, self.yT[None, :], loss).mean(1)

    # ------------------------------------------------------------------
    def gamma_per_panel(self, lam):
        """Per-unit Gamma = lam ** (number of unrecorded panels): the more
        modalities a unit lacks, the further its label law may sit from the
        complete cases'."""
        nmis = self.K - self.P[self.pidU].sum(1)
        return lam ** nmis

    def aa_pair(self, a, b, loss, gamma, direction=None, complete_exact=True):
        """Bounds on R_T(a) - R_T(b) using U, with the DR correction for the
        source-side outcome model.  ``gamma`` is a scalar or one value per U unit.  Complete-pattern units keep Gamma = 1 when
        ``complete_exact`` (the deployment's complete cases are exchangeable
        with V in the complete-case design)."""
        a0, a1 = loss01(self.PU[a], loss)
        b0, b1 = loss01(self.PU[b], loss)
        ps = np.clip(self.pS_U, 1e-6, 1 - 1e-6)
        inc = self.pidU != self.full_id if complete_exact else np.ones(len(ps), bool)
        corr = self._dr_correction(loss)
        shift = corr[a] - corr[b]
        g = gamma[inc] if np.ndim(gamma) else gamma  # per-unit Gamma allowed
        lo_i, pt_i, up_i = diff_bounds(a0[inc], a1[inc], b0[inc], b1[inc], ps[inc], g, direction=direction)
        c = inc.mean()
        base = 0.0
        if (~inc).any():
            base = float(np.mean((a0[~inc] - b0[~inc]) + ps[~inc] * ((a1[~inc] - a0[~inc]) - (b1[~inc] - b0[~inc]))))
        k = (~inc).mean()
        return c * lo_i + k * base + shift, c * pt_i + k * base + shift, c * up_i + k * base + shift

    def aa_breakdown(self, a, b, loss, direction=None):
        a0, a1 = loss01(self.PU[a], loss)
        b0, b1 = loss01(self.PU[b], loss)
        ps = np.clip(self.pS_U, 1e-6, 1 - 1e-6)
        inc = self.pidU != self.full_id
        corr = self._dr_correction(loss)
        # fold the Gamma-free part into a constant by appending pseudo-units is messy;
        # search directly on the pair bound instead.
        lo, pt, up = self.aa_pair(a, b, loss, 1.0, direction)
        if lo <= 0 <= up:
            return 1.0
        f = (lambda g: self.aa_pair(a, b, loss, g, direction)[2]) if pt < 0 else \
            (lambda g: -self.aa_pair(a, b, loss, g, direction)[0])
        if f(50.0) < 0:
            return float("inf")
        x, y = 1.0, 50.0
        for _ in range(40):
            mid = np.sqrt(x * y)
            if f(mid) < 0:
                x = mid
            else:
                y = mid
        return float(y)


def aa_bootstrap_ci(ev, loss, gamma_unit, direction=None, exact=True, n_boot=200, rng=None):
    """Bootstrap 95% CI of the identified interval for every pairwise risk
    difference: (2.5 % quantile of L*, 97.5 % quantile of U*), resampling U
    units (bound) and V units (DR shift).  ``gamma_unit``: one Gamma per U unit
    (complete-pattern units are held at Gamma = 1 when ``exact``).  Returns
    (ci_lo, ci_hi, pair_index)."""
    rng = np.random.default_rng(0) if rng is None else rng
    J = ev.PU.shape[0]
    iu = np.triu_indices(J, 1)
    l0, l1 = loss01(np.asarray(ev.PU, float), loss)
    c = (l0[:, None, :] - l0[None, :, :])[iu]
    g = ((l1 - l0)[:, None, :] - (l1 - l0)[None, :, :])[iu]
    ps = np.clip(ev.pS_U, 1e-6, 1 - 1e-6)
    inc = (ev.pidU != ev.full_id) if exact else np.ones(len(ps), bool)
    gam = np.where(inc, gamma_unit, 1.0)
    lo, hi = or_interval(ps, gam)
    if direction == "down":
        hi, lo = np.minimum(hi, ps), np.minimum(lo, ps)
    elif direction == "up":
        hi, lo = np.maximum(hi, ps), np.maximum(lo, ps)
    ulo = c + np.where(g > 0, lo, hi) * g
    uup = c + np.where(g > 0, hi, lo) * g
    PV = np.asarray(ev.PV, float)
    LV = unit_loss(PV, ev.yV[None, None, :], loss)
    v0, v1 = loss01(PV, loss)
    R = LV - (v0 * (1 - ev.pS_V_cf[None]) + v1 * ev.pS_V_cf[None])
    nV, nU = len(ev.yV), len(ps)
    WV = rng.multinomial(nV, np.full(nV, 1 / nV), size=n_boot).astype(float)
    WU = rng.multinomial(nU, np.full(nU, 1 / nU), size=n_boot).astype(float) / nU
    corr = np.einsum("jmi,mi,bi->bjm", R, ev.w_pat, WV) / np.einsum("mi,bi->bm", ev.w_pat, WV)[:, None, :]
    corr = corr @ ev.freq
    shift = (corr[:, :, None] - corr[:, None, :])[:, iu[0], iu[1]]
    Lb, Ub = WU @ ulo.T + shift, WU @ uup.T + shift
    return np.percentile(Lb, 2.5, axis=0), np.percentile(Ub, 97.5, axis=0), iu


# --------------------------------------------------------------------------
# agreement metrics
# --------------------------------------------------------------------------

def agreement(est, truth):
    est, truth = np.asarray(est), np.asarray(truth)
    tau = kendalltau(est, truth).statistic
    rho = spearmanr(est, truth).statistic
    tied_best = np.flatnonzero(est == est.min())
    regret = float(truth[tied_best].mean() - truth.min())          # ties: expected regret of a random pick
    J = len(truth)
    iu = np.triu_indices(J, 1)
    de = (est[:, None] - est[None, :])[iu]
    dt = (truth[:, None] - truth[None, :])[iu]
    sign_agree = float(np.mean(np.where(de == 0, 0.5, np.sign(de) == np.sign(dt))))  # ties score 1/2
    return {"kendall_tau": float(tau), "spearman": float(rho), "top1_regret": regret,
            "pair_sign_agreement": sign_agree, "selected": int(tied_best[0]), "best": int(np.argmin(truth))}
