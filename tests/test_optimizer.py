"""Optimizer correctness against independent solutions and invariances."""

from __future__ import annotations

import numpy as np
import pytest

from portfolio_engine.constraints import FeasibleSet, InfeasibleConstraintError
from portfolio_engine.estimates import estimate_moments, psd_factor
from portfolio_engine.optimizer import Optimizer, mv_objective
from portfolio_engine.strategies import STRATEGIES, AllocationParams, propose

OPT = Optimizer()


def test_diagonal_minvar_matches_inverse_variance():
    var = np.array([1.0, 1.5, 2.0, 2.5, 3.0]) * 1e-4
    f = psd_factor(np.diag(var))
    res = OPT.min_variance(f.factor, max_weight=1.0)
    expected = (1 / var) / (1 / var).sum()
    assert expected.max() < 1.0  # bounds do not bind
    np.testing.assert_allclose(res.weights, expected, atol=1e-6)


def test_two_asset_constrained_mv_matches_dense_grid():
    sigma = np.array([[0.0004, 0.0001], [0.0001, 0.0001]])
    mu = np.array([0.0012, 0.0002])
    gamma, h, cap = 5.0, 21, 0.7
    res = OPT.mean_variance(psd_factor(sigma).factor, mu, gamma, h, cap)
    grid = np.linspace(1 - cap, cap, 200_001)  # w1 in [0.3, 0.7], w2 = 1 - w1
    obj = [
        mv_objective(np.array([a, 1 - a]), sigma, mu, gamma, h, 0.0, np.zeros(2))
        for a in grid[::100]
    ]
    a_star = grid[::100][int(np.argmin(obj))]
    # Analytic interior optimum of the 1-D quadratic, clipped to the box:
    d = np.array([1.0, -1.0])
    w_base = np.array([0.0, 1.0])
    a_analytic = (h * mu @ d - gamma * h * w_base @ sigma @ d) / (gamma * h * d @ sigma @ d)
    a_analytic = float(np.clip(a_analytic, 1 - cap, cap))
    assert res.weights[0] == pytest.approx(a_analytic, abs=1e-5)
    assert res.weights[0] == pytest.approx(a_star, abs=2e-3)


def test_cost_penalty_creates_no_trade_region():
    sigma = np.diag([1e-4, 1e-4])
    f = psd_factor(sigma).factor
    mu = np.array([0.00011, 0.0001])
    w_sig = np.array([0.5, 0.5])
    no_cost = OPT.mean_variance(f, mu, 5.0, 21, 1.0, 0.0, w_sig)
    big_cost = OPT.mean_variance(f, mu, 5.0, 21, 1.0, 0.05, w_sig)
    assert no_cost.weights[0] > 0.5 + 1e-4
    np.testing.assert_allclose(big_cost.weights, w_sig, atol=1e-6)


def test_targets_respect_constraints_and_caps():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((252, 30)) * 0.01 + rng.normal(0, 0.002, 30)
    est = estimate_moments(x)
    p = AllocationParams(5.0, 21, 0.75, 0.10, 0.001)
    feas = FeasibleSet(30, 0.10)
    for sid, spec in STRATEGIES.items():
        r = propose(spec, est, np.zeros(30), p, OPT)
        assert r.ok, sid
        assert feas.residuals(r.weights)["max"] <= 1e-9
        assert r.weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_infeasible_caps_rejected():
    with pytest.raises(InfeasibleConstraintError):
        FeasibleSet(30, 0.03)
    with pytest.raises(InfeasibleConstraintError):
        OPT.min_variance(np.eye(5) * 0.01, max_weight=0.1)


def test_rho_one_reduces_to_minvar():
    rng = np.random.default_rng(5)
    est = estimate_moments(
        rng.standard_normal((252, 12)) * np.linspace(0.006, 0.02, 12)
        + 0.0005 * rng.standard_normal(12)
    )
    p = AllocationParams(5.0, 21, 1.0, 0.2, 0.0)
    a = propose(STRATEGIES["MEAN_COV_SHRINK_MV"], est, np.zeros(12), p, OPT).weights
    b = propose(STRATEGIES["LW_MINVAR"], est, np.zeros(12), p, OPT).weights
    np.testing.assert_allclose(a, b, atol=2e-5)


def test_zero_cost_cost_aware_equals_combined_shrinkage():
    rng = np.random.default_rng(6)
    est = estimate_moments(rng.standard_normal((252, 10)) * 0.01 + 0.0003)
    w_sig = rng.dirichlet(np.ones(10))
    p = AllocationParams(5.0, 21, 0.75, 0.25, 0.0)
    a = propose(STRATEGIES["COST_AWARE_REG_MV"], est, w_sig, p, OPT).weights
    b = propose(STRATEGIES["MEAN_COV_SHRINK_MV"], est, w_sig, p, OPT).weights
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_column_permutation_invariance():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((252, 8)) * np.linspace(0.006, 0.02, 8) + 0.0004 * rng.standard_normal(
        8
    )
    perm = rng.permutation(8)
    p = AllocationParams(5.0, 21, 0.75, 0.3, 0.001)
    w_sig = rng.dirichlet(np.ones(8))
    for sid in ("SAMPLE_MV", "COV_SHRINK_MV", "COST_AWARE_REG_MV", "LW_MINVAR"):
        a = propose(STRATEGIES[sid], estimate_moments(x), w_sig, p, OPT)
        b = propose(STRATEGIES[sid], estimate_moments(x[:, perm]), w_sig[perm], p, OPT)
        # Compare objective values (robust to non-unique optima) and weights where unique.
        assert a.objective == pytest.approx(b.objective, rel=1e-6, abs=1e-10)
        back = np.empty(8)
        back[perm] = b.weights
        np.testing.assert_allclose(a.weights, back, atol=1e-4)


def test_solver_failure_is_reported_not_replaced(monkeypatch):
    opt = Optimizer(solvers=("CLARABEL",))
    prob = opt.problem(4, 0.5, "minvar")

    def boom(*a, **k):
        raise __import__("cvxpy").error.SolverError("forced failure")

    monkeypatch.setattr(prob.problem, "solve", boom)
    r = prob.solve(np.eye(4) * 0.01)
    assert not r.ok and r.status == "failed" and r.weights is None
    assert "forced failure" in r.attempts[0]["outcome"]
