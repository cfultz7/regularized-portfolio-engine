"""Estimation: normalization, Ledoit-Wolf identity, mean shrinkage, PSD handling."""

from __future__ import annotations

import numpy as np
import pytest

from portfolio_engine.estimates import (
    CovarianceError,
    estimate_moments,
    psd_factor,
    risk_contributions,
    shrink_mean,
)


def _window(t=252, n=10, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) * 0.004
    return rng.standard_normal((t, n)) @ a + 0.0004


def test_empirical_covariance_uses_divisor_T_and_lw_identity():
    x = _window()
    est = estimate_moments(x)
    xc = x - x.mean(0)
    s_manual = xc.T @ xc / x.shape[0]
    np.testing.assert_allclose(est.sample.matrix, s_manual, rtol=1e-12, atol=1e-18)
    n = x.shape[1]
    target = np.trace(s_manual) / n * np.eye(n)
    expect = (1 - est.lw_alpha) * s_manual + est.lw_alpha * target
    np.testing.assert_allclose(est.lw.matrix, expect, rtol=1e-10, atol=1e-18)
    assert 0.0 <= est.lw_alpha <= 1.0


def test_estimates_symmetric_psd_and_better_conditioned():
    est = estimate_moments(_window(t=40, n=30))  # T close to N: sample badly conditioned
    for c in (est.sample, est.lw):
        np.testing.assert_allclose(c.matrix, c.matrix.T, atol=0)
        assert c.eigenvalues.min() >= 0
        np.testing.assert_allclose(c.factor @ c.factor.T, c.matrix, rtol=1e-9, atol=1e-14)
    assert est.lw.condition_number < est.sample.condition_number


def test_singular_sample_reports_infinite_condition():
    est = estimate_moments(_window(t=20, n=30))
    assert est.sample.condition_number == np.inf or est.sample.condition_number > 1e12
    assert np.isfinite(est.lw.condition_number)
    assert any("singular" in n for n in est.notes)


def test_shrink_mean_formula_and_bounds():
    mu = np.array([0.001, -0.002, 0.0005])
    np.testing.assert_allclose(shrink_mean(mu, 0.0), mu)
    np.testing.assert_allclose(shrink_mean(mu, 1.0), np.full(3, mu.mean()))
    np.testing.assert_allclose(shrink_mean(mu, 0.75), 0.25 * mu + 0.75 * mu.mean())
    with pytest.raises(ValueError):
        shrink_mean(mu, 1.5)


def test_psd_repair_only_for_roundoff():
    m = np.diag([1.0, 0.5, 0.0])
    m[2, 2] = -1e-14  # roundoff-level negative
    f = psd_factor(m)
    assert f.repaired and f.eigenvalues.min() == 0.0 and f.repair_magnitude == pytest.approx(1e-14)
    with pytest.raises(CovarianceError, match="materially negative"):
        psd_factor(np.diag([1.0, -0.1]))
    with pytest.raises(CovarianceError):
        psd_factor(np.array([[1.0, np.nan], [np.nan, 1.0]]))


def test_invalid_windows_rejected():
    with pytest.raises(ValueError):
        estimate_moments(np.array([[0.01, 0.02]]))
    x = _window(t=30, n=3)
    x[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        estimate_moments(x)


def test_risk_contributions_sum_to_one_and_keep_negatives():
    sigma = np.array([[0.04, -0.03], [-0.03, 0.09]])
    w = np.array([0.8, 0.2])
    rc = risk_contributions(w, sigma)
    assert rc.sum() == pytest.approx(1.0)
    w2 = np.array([0.1, 0.9])
    rc2 = risk_contributions(w2, sigma)
    assert rc2.min() < 0  # negative contribution not clipped
    assert rc2.sum() == pytest.approx(1.0)
