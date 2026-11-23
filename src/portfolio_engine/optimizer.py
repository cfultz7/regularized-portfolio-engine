"""Constrained mean-variance and minimum-variance solvers (CVXPY, DPP-cached).

Objective (mean-variance methods), in holding-period units with H trading days::

    minimize  (gamma/2) * H * w' Sigma_daily w  -  H * mu_daily' w  +  kappa * ||w - w_signal||_1
    s.t.      sum(w) == 1,  0 <= w <= max_weight

``kappa`` is the one-way cost rate (decimal) for the cost-aware method and zero otherwise.
Minimum variance minimizes ``w' Sigma_daily w`` over the same set.

The quadratic form is written as ``||F' w||^2`` with ``Sigma = F F'`` (an eigen factor), which
is PSD by construction and keeps the problem DPP-compliant so it is canonicalized once and
re-solved with new parameter values. Scaling constants are folded into parameters so the
argmin is unchanged; reported objective values are recomputed from the formulas above.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Literal

import cvxpy as cp
import numpy as np
import pandas as pd

from .constraints import FeasibleSet

Kind = Literal["mv", "minvar"]
#: Positive constant multiplying the whole objective to improve solver scaling (argmin unchanged).
OBJECTIVE_SCALE = 100.0


@dataclass
class SolveResult:
    weights: np.ndarray | None
    status: str  # "optimal" | "accepted_inaccurate" | "failed" | "analytic"
    solver: str | None
    solver_status: str | None
    objective: float | None
    solve_time: float
    residuals: dict[str, float]
    cleanup: float = 0.0
    attempts: list[dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.weights is not None


def mv_objective(
    w: np.ndarray,
    sigma: np.ndarray,
    mu: np.ndarray,
    gamma: float,
    horizon: int,
    kappa: float,
    w_signal: np.ndarray,
) -> float:
    return float(
        0.5 * gamma * horizon * w @ sigma @ w
        - horizon * mu @ w
        + kappa * np.abs(w - w_signal).sum()
    )


class AllocationProblem:
    """A cached parametrized problem for one (N, cap, kind)."""

    def __init__(self, feasible: FeasibleSet, kind: Kind, solvers: tuple[str, ...]):
        self.feasible = feasible
        self.kind = kind
        n = feasible.n_assets
        self.solvers = tuple(s for s in solvers if s in cp.installed_solvers())
        if not self.solvers:
            raise RuntimeError(
                f"none of the configured solvers {solvers} is installed; installed: {cp.installed_solvers()}"
            )
        self.w = cp.Variable(n, name="w")
        self.F = cp.Parameter((n, n), name="F_scaled")
        cons = [cp.sum(self.w) == 1, self.w >= 0, self.w <= feasible.max_weight]
        risk = cp.sum_squares(self.F.T @ self.w)
        if kind == "mv":
            self.mu = cp.Parameter(n, name="mu_scaled")
            self.k = cp.Parameter(nonneg=True, name="kappa_scaled")
            self.kw0 = cp.Parameter(n, name="kappa_w_signal_scaled")
            obj = risk - self.mu @ self.w + cp.norm1(self.k * self.w - self.kw0)
        else:
            obj = risk
        self.problem = cp.Problem(cp.Minimize(obj), cons)

    def solve(
        self,
        factor: np.ndarray,
        mu: np.ndarray | None = None,
        gamma: float = 1.0,
        horizon: int = 1,
        kappa: float = 0.0,
        w_signal: np.ndarray | None = None,
    ) -> SolveResult:
        n = self.feasible.n_assets
        sigma = factor @ factor.T
        if self.kind == "mv":
            assert mu is not None
            w0 = np.zeros(n) if w_signal is None else np.asarray(w_signal, dtype=float)
            self.F.value = factor * math.sqrt(OBJECTIVE_SCALE * 0.5 * gamma * horizon)
            self.mu.value = OBJECTIVE_SCALE * horizon * np.asarray(mu, dtype=float)
            self.k.value = OBJECTIVE_SCALE * kappa
            self.kw0.value = OBJECTIVE_SCALE * kappa * w0
        else:
            # Minimum variance: scale so the objective is O(1) for daily equity-like variances.
            scale = OBJECTIVE_SCALE * 0.5 * 5.0 * 21.0
            self.F.value = factor * math.sqrt(scale)
        attempts: list[dict[str, str]] = []
        t0 = time.perf_counter()
        for solver in self.solvers:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.problem.solve(solver=solver, warm_start=True)
            except (
                cp.error.SolverError,
                ArithmeticError,
                ValueError,
            ) as exc:  # pragma: no cover - solver specific
                attempts.append(
                    {"solver": solver, "outcome": f"error: {type(exc).__name__}: {exc}"[:200]}
                )
                continue
            status = str(self.problem.status)
            raw = self.w.value
            if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or raw is None:
                attempts.append({"solver": solver, "outcome": f"status {status}"})
                continue
            raw = np.asarray(raw, dtype=float).ravel()
            res = self.feasible.residuals(raw)
            if not np.all(np.isfinite(raw)) or res["max"] > self.feasible.tol:
                attempts.append(
                    {
                        "solver": solver,
                        "outcome": f"status {status}; residual {res['max']:.2e} exceeds tolerance",
                    }
                )
                continue
            clean, adj = self.feasible.clean(raw)
            attempts.append({"solver": solver, "outcome": f"accepted ({status})"})
            if self.kind == "mv":
                assert mu is not None
                obj = mv_objective(
                    clean,
                    sigma,
                    np.asarray(mu),
                    gamma,
                    horizon,
                    kappa,
                    np.zeros(n) if w_signal is None else np.asarray(w_signal),
                )
            else:
                obj = float(clean @ sigma @ clean)
            return SolveResult(
                weights=clean,
                status="optimal" if status == cp.OPTIMAL else "accepted_inaccurate",
                solver=solver,
                solver_status=status,
                objective=obj,
                solve_time=time.perf_counter() - t0,
                residuals=res,
                cleanup=adj,
                attempts=attempts,
            )
        return SolveResult(
            None,
            "failed",
            None,
            None,
            None,
            time.perf_counter() - t0,
            {"budget": np.nan, "lower": np.nan, "upper": np.nan, "max": np.nan},
            0.0,
            attempts,
        )


class Optimizer:
    """Holds cached problems keyed by (N, cap, kind)."""

    def __init__(self, solvers: tuple[str, ...] = ("CLARABEL", "OSQP", "SCS"), tol: float = 1e-6):
        self.solvers = solvers
        self.tol = tol
        self._cache: dict[tuple[int, float, str], AllocationProblem] = {}

    def problem(self, n: int, max_weight: float, kind: Kind) -> AllocationProblem:
        key = (n, float(max_weight), kind)
        if key not in self._cache:
            self._cache[key] = AllocationProblem(
                FeasibleSet(n, max_weight, self.tol), kind, self.solvers
            )
        return self._cache[key]

    def mean_variance(
        self,
        factor: np.ndarray,
        mu: np.ndarray,
        gamma: float,
        horizon: int,
        max_weight: float,
        kappa: float = 0.0,
        w_signal: np.ndarray | None = None,
    ) -> SolveResult:
        return self.problem(len(mu), max_weight, "mv").solve(
            factor, mu, gamma, horizon, kappa, w_signal
        )

    def min_variance(self, factor: np.ndarray, max_weight: float) -> SolveResult:
        return self.problem(factor.shape[0], max_weight, "minvar").solve(factor)


def optimize_user_panel(
    returns: pd.DataFrame,
    as_of: str | pd.Timestamp,
    current_weights: pd.Series | dict[str, float] | None = None,
    method: str = "COST_AWARE_REG_MV",
    lookback: int = 252,
    gamma: float = 5.0,
    horizon: int = 21,
    rho: float = 0.75,
    max_weight: float = 0.10,
    cost_bps: float = 10.0,
) -> tuple[pd.Series, dict]:
    """Public API: propose a target for a user return panel as of a date.

    ``returns`` is a DataFrame of decimal simple daily returns indexed by date. Only rows on or
    before ``as_of`` are used. ``current_weights`` (Series or mapping, risky weights at the
    as-of close) default to all cash. Returns ``(target_weights: Series, info: dict)``.
    """
    from .estimates import estimate_moments
    from .strategies import STRATEGIES, AllocationParams, propose

    df = pd.DataFrame(returns).sort_index()
    if df.index.has_duplicates:
        raise ValueError("duplicate dates in return panel")
    hist = df.loc[: pd.Timestamp(as_of)]
    if len(hist) < lookback:
        raise ValueError(f"need {lookback} observations through {as_of}; have {len(hist)}")
    window = hist.iloc[-lookback:]
    if window.isna().any().any():
        raise ValueError("estimation window contains missing values")
    cols = list(df.columns)
    if current_weights is None:
        w0 = np.zeros(len(cols))
    else:
        w0 = pd.Series(current_weights, dtype=float).reindex(cols).fillna(0.0).to_numpy()
    FeasibleSet(len(cols), max_weight)
    est = estimate_moments(window.to_numpy())
    params = AllocationParams(
        gamma=gamma, horizon=horizon, rho=rho, max_weight=max_weight, cost_rate=cost_bps / 1e4
    )
    res = propose(STRATEGIES[method], est, w0, params, Optimizer())
    if res.weights is None:
        raise RuntimeError(f"all solvers failed: {res.attempts}")
    info = {
        "method": method,
        "as_of": str(window.index[-1].date()),
        "window_start": str(window.index[0].date()),
        "n_obs": len(window),
        "solver": res.solver,
        "status": res.status,
        "objective": res.objective,
        "lw_alpha": est.lw_alpha,
        "planned_trade_fraction": float(np.abs(res.weights - w0).sum()),
        "planned_cost_fraction": float(cost_bps / 1e4 * np.abs(res.weights - w0).sum()),
    }
    return pd.Series(res.weights, index=cols, name="target_weight"), info
