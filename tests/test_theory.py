import numpy as np
import pytest

from src.n1 import theory as T


def test_verify_all_ok():
    rep = T.verify(seed=1, n_rand=100)
    assert rep["all_ok"], rep


def test_ce1_is_a_ranking_reversal_with_identical_observables():
    c = T.ce1()
    w1, w2 = c["MCAR"], c["label-dependent"]
    assert w1["observed_target_law"] == w2["observed_target_law"]
    assert w1["R_drop_rate_matched(A)"] == w2["R_drop_rate_matched(A)"]
    assert (w1["R_T(A)"] < w1["R_T(B)"]) and (w2["R_T(B)"] < w2["R_T(A)"])


def test_ce2_selection_flip_is_repaired_by_reweighting():
    c = T.ce2()
    assert c["R_drop_rate_matched(B)"] < c["R_drop_rate_matched(A)"]
    assert c["R_T(A)"] < c["R_T(B)"]
    assert c["R_reweighted(A)"] == c["R_T(A)"] and c["R_reweighted(B)"] == c["R_T(B)"]


@pytest.mark.parametrize("loss", ["brier", "logloss"])
def test_difference_bounds_are_tighter_than_differenced_single_bounds(loss):
    rng = np.random.default_rng(0)
    n = 200
    ps = rng.uniform(.05, .6, n)
    fa = np.clip(ps + rng.normal(0, .05, n), .01, .99)
    fb = np.clip(ps + rng.normal(0, .05, n), .01, .99)
    a0, a1 = T.loss_pair(fa, loss)
    b0, b1 = T.loss_pair(fb, loss)
    z = np.zeros(n)
    lo, _, up = T.diff_bounds(a0, a1, b0, b1, ps, 2.0)
    # single-model bounds: compare each model against a zero-loss reference
    la, _, ua = T.diff_bounds(a0, a1, z, z, ps, 2.0)
    lb, _, ub = T.diff_bounds(b0, b1, z, z, ps, 2.0)
    assert up - lo <= (ua - lb) - (la - ub) + 1e-12


def test_per_unit_gamma_array_matches_scalar():
    rng = np.random.default_rng(3)
    n = 50
    ps = rng.uniform(.05, .95, n)
    a0, a1 = T.loss_pair(rng.uniform(.05, .95, n))
    b0, b1 = T.loss_pair(rng.uniform(.05, .95, n))
    s = T.diff_bounds(a0, a1, b0, b1, ps, 1.7)
    v = T.diff_bounds(a0, a1, b0, b1, ps, np.full(n, 1.7))
    assert np.allclose(s, v)


def test_one_sided_down_is_inside_two_sided():
    rng = np.random.default_rng(4)
    n = 80
    ps = rng.uniform(.05, .95, n)
    a0, a1 = T.loss_pair(rng.uniform(.05, .95, n), "logloss")
    b0, b1 = T.loss_pair(rng.uniform(.05, .95, n), "logloss")
    lo2, _, up2 = T.diff_bounds(a0, a1, b0, b1, ps, 3.0)
    lo1, _, up1 = T.diff_bounds(a0, a1, b0, b1, ps, 3.0, direction="down")
    assert lo2 - 1e-12 <= lo1 <= up1 <= up2 + 1e-12


def test_ce1b_complete_case_source_no_label_dependence_still_reverses():
    c = T.ce1b()
    assert c["W1"]["complete_case_law"] == c["W2"]["complete_case_law"]
    assert c["W1"]["unlabeled_law"] == c["W2"]["unlabeled_law"]
    assert c["W1"]["R_T(A)"] < c["W1"]["R_T(B)"] and c["W2"]["R_T(B)"] < c["W2"]["R_T(A)"]


def test_ce3_selection_only_flip_repaired_by_reweighting():
    c = T.ce3()
    assert c["R_drop_rate_matched(B)"] < c["R_drop_rate_matched(A)"] and c["R_T(A)"] < c["R_T(B)"]
    assert c["R_reweighted(A)"] == c["R_T(A)"] and c["R_reweighted(B)"] == c["R_T(B)"]
