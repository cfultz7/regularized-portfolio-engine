"""Exact self-financing accounting for a linear (proportional) trading-cost model.

At an execution with pretrade NAV ``V_pre``, pretrade risky weights ``b = holdings / V_pre``,
desired post-cost target weights ``w`` and one-way cost rate ``c``, find ``p`` in [0, 1] with

    p + c * sum_i |p * w_i - b_i| = 1.

Then ``V_post = p V_pre``, posttrade holdings ``V_post * w``, traded notional
``V_post * w - V_pre * b`` and realized fee ``c * sum|traded| = V_pre - V_post``.
``f(p) = p + c * sum|p w - b| - 1`` is continuous with slope >= 1 - c > 0, f(0) <= 0 and
f(1) >= 0 (for c * sum(b) <= 1), so the root is unique and bracketed by [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

RECONCILE_TOL = 1e-8  # relative to NAV


class AccountingError(RuntimeError):
    """Raised when a ledger identity fails."""


@dataclass(frozen=True)
class TradeResult:
    p: float
    v_pre: float
    v_post: float
    pre_holdings: np.ndarray
    post_holdings: np.ndarray
    traded: np.ndarray  # signed dollars
    fee: float
    gross_traded: float  # sum |traded| dollars
    cash_pre: float
    cash_post: float

    @property
    def turnover_gross(self) -> float:
        return self.gross_traded / self.v_pre

    @property
    def turnover_half(self) -> float:
        return 0.5 * self.turnover_gross


def solve_cost_fraction(b: np.ndarray, w: np.ndarray, c: float) -> float:
    b = np.asarray(b, dtype=float)
    w = np.asarray(w, dtype=float)
    if c < 0 or c >= 1:
        raise ValueError("cost rate must lie in [0, 1)")
    if c == 0.0:
        return 1.0

    def f(p: float) -> float:
        return p + c * np.abs(p * w - b).sum() - 1.0

    f1 = f(1.0)
    if f1 <= 0.0:  # no trade needed (w == b) up to roundoff
        return 1.0
    f0 = f(0.0)
    if f0 > 0.0:
        raise AccountingError("costs exceed NAV: no self-financing solution in [0, 1]")
    return float(brentq(f, 0.0, 1.0, xtol=1e-16, rtol=4 * np.finfo(float).eps, maxiter=200))


def execute_trade(holdings: np.ndarray, cash: float, target: np.ndarray, c: float) -> TradeResult:
    """Trade from current dollar ``holdings`` and ``cash`` to target risky weights ``target``."""
    holdings = np.asarray(holdings, dtype=float)
    target = np.asarray(target, dtype=float)
    v_pre = float(holdings.sum() + cash)
    if v_pre <= 0:
        raise AccountingError("nonpositive NAV at execution")
    if abs(target.sum() - 1.0) > 1e-9 or target.min() < -1e-12:
        raise AccountingError("targets must be fully invested and long only")
    b = holdings / v_pre
    p = solve_cost_fraction(b, target, c)
    v_post = p * v_pre
    post = v_post * target
    traded = post - holdings
    gross = float(np.abs(traded).sum())
    fee = c * gross
    cash_post = float(v_post - post.sum())
    # Identity checks: fee equals the NAV reduction and value is conserved.
    if abs((v_pre - v_post) - fee) > RECONCILE_TOL * v_pre:
        raise AccountingError(f"self-financing identity failed: {v_pre - v_post} vs fee {fee}")
    if abs(post.sum() + cash_post + fee - v_pre) > RECONCILE_TOL * v_pre:
        raise AccountingError("holdings + cash + fee != pretrade NAV")
    return TradeResult(
        p, v_pre, v_post, holdings.copy(), post, traded, fee, gross, float(cash), cash_post
    )
