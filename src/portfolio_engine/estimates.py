"""Moment estimation: sample moments, Ledoit-Wolf covariance shrinkage, fixed mean shrinkage.

Conventions
-----------
* Means are arithmetic daily means of decimal simple returns.
* The empirical covariance is ``S = Xc.T @ Xc / T`` with ``Xc`` the column-centered window
  (divisor ``T``, the maximum-likelihood normalization). This is the matrix that
  ``sklearn.covariance.LedoitWolf(assume_centered=False)`` shrinks, so sample and shrunk
  estimates differ only through shrinkage, never through normalization.
* Ledoit-Wolf (2004) target: scaled identity ``(trace(S)/N) I``;
  ``Sigma_LW = (1 - alpha) S + alpha (trace(S)/N) I``. ``alpha`` is the covariance shrinkage
  intensity estimated by scikit-learn at every fit.
* Mean shrinkage: ``mu_shrunk = (1 - rho) mu + rho mean(mu) 1``, a fixed, transparent rule with
  constant ``rho`` (not a James-Stein or Bayesian posterior estimator).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.covariance import LedoitWolf

log = logging.getLogger(__name__)


class CovarianceError(ValueError):
    """Raised for materially invalid covariance matrices."""


@dataclass(frozen=True)
class CovarianceFactor:
    """A symmetric PSD matrix with a factor ``F`` such that ``matrix == F @ F.T``."""

    matrix: np.ndarray
    factor: np.ndarray
    eigenvalues: np.ndarray  # ascending, after any numerical repair
    raw_min_eigenvalue: float
    repaired: bool
    repair_magnitude: float
    condition_number: float  # inf when singular


@dataclass(frozen=True)
class MomentEstimates:
    n_obs: int
    mu_sample: np.ndarray
    sample: CovarianceFactor
    lw: CovarianceFactor
    lw_alpha: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    def mean(self, model: str, rho: float) -> np.ndarray:
        if model == "sample":
            return self.mu_sample
        if model == "shrunk":
            return shrink_mean(self.mu_sample, rho)
        raise ValueError(model)

    def cov(self, model: str) -> CovarianceFactor:
        if model == "sample":
            return self.sample
        if model == "lw":
            return self.lw
        raise ValueError(model)


def shrink_mean(mu: np.ndarray, rho: float) -> np.ndarray:
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    mu = np.asarray(mu, dtype=float)
    return (1.0 - rho) * mu + rho * mu.mean() * np.ones_like(mu)


def empirical_covariance(x: np.ndarray) -> np.ndarray:
    """Centered empirical covariance with divisor T (matches scikit-learn)."""
    x = np.asarray(x, dtype=float)
    xc = x - x.mean(axis=0)
    return xc.T @ xc / x.shape[0]


def psd_factor(
    matrix: np.ndarray, rel_tol: float = 1e-10, label: str = "covariance"
) -> CovarianceFactor:
    """Symmetrize, check PSD, and factor a covariance matrix.

    Floating-point repair: eigenvalues in ``[-rel_tol * max_eig, 0)`` are set to zero and the
    repair is recorded. Anything more negative is a materially invalid matrix and raises.
    This is numerical repair, not statistical regularization: no ridge is ever added.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise CovarianceError(f"{label}: matrix must be square")
    if not np.all(np.isfinite(m)):
        raise CovarianceError(f"{label}: non-finite entries")
    sym = 0.5 * (m + m.T)
    evals, evecs = np.linalg.eigh(sym)
    top = float(max(evals[-1], 0.0))
    if top <= 0.0:
        raise CovarianceError(f"{label}: no positive eigenvalue")
    raw_min = float(evals[0])
    floor = -rel_tol * top
    if raw_min < floor:
        raise CovarianceError(
            f"{label}: materially negative eigenvalue {raw_min:.3e} (max {top:.3e})"
        )
    repaired = bool(raw_min < 0.0)
    clipped = np.clip(evals, 0.0, None)
    repair_mag = float(np.max(clipped - evals))
    if repaired:
        log.debug("%s: clipped eigenvalues >= %.3e to zero (roundoff repair)", label, raw_min)
        sym = (evecs * clipped) @ evecs.T
        sym = 0.5 * (sym + sym.T)
    factor = evecs * np.sqrt(clipped)
    cond = float(clipped[-1] / clipped[0]) if clipped[0] > 0 else float("inf")
    return CovarianceFactor(sym, factor, clipped, raw_min, repaired, repair_mag, cond)


def estimate_moments(window: np.ndarray, rel_tol: float = 1e-10) -> MomentEstimates:
    """Estimate sample and Ledoit-Wolf moments from a T x N window of decimal returns."""
    x = np.asarray(window, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("window must be a 2-D array with at least two observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("estimation window contains non-finite values")
    mu = x.mean(axis=0)
    s = empirical_covariance(x)
    lw = LedoitWolf(assume_centered=False).fit(x)
    notes = []
    if x.shape[0] <= x.shape[1]:
        notes.append(f"T={x.shape[0]} <= N={x.shape[1]}: sample covariance is singular")
    return MomentEstimates(
        n_obs=x.shape[0],
        mu_sample=mu,
        sample=psd_factor(s, rel_tol, "sample covariance"),
        lw=psd_factor(lw.covariance_, rel_tol, "Ledoit-Wolf covariance"),
        lw_alpha=float(lw.shrinkage_),
        notes=tuple(notes),
    )


def risk_contributions(w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Fractional contributions to forecast variance; sum to one when variance > 0.

    Negative contributions are possible (negative covariances) and are not clipped.
    """
    w = np.asarray(w, dtype=float)
    mv = sigma @ w
    var = float(w @ mv)
    if var <= 0:
        return np.full_like(w, np.nan)
    return w * mv / var
