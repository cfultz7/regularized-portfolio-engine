"""Self-financing ledger, hand-calculated drift, and NAV scaling (SYNTHETIC data)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from portfolio_engine.accounting import execute_trade, solve_cost_fraction
from portfolio_engine.backtest import RunSpec, run_backtest
from portfolio_engine.optimizer import Optimizer
from portfolio_engine.strategies import AllocationParams
from tests.conftest import gaussian_panel, make_panel


def test_zero_cost_is_identity_fraction():
    assert solve_cost_fraction(np.array([0.2, 0.3]), np.array([0.5, 0.5]), 0.0) == 1.0


def test_initial_purchase_from_cash():
    c = 0.001
    tr = execute_trade(np.zeros(3), 1_000_000.0, np.array([0.5, 0.3, 0.2]), c)
    assert tr.p == pytest.approx(1 / (1 + c), rel=1e-14)
    assert tr.fee == pytest.approx(1_000_000.0 * c / (1 + c), rel=1e-12)
    assert tr.cash_post == pytest.approx(0.0, abs=1e-6)
    assert tr.turnover_gross == pytest.approx(1 / (1 + c), rel=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_self_financing_identities_random(seed):
    rng = np.random.default_rng(seed)
    n = 6
    hold = rng.uniform(0, 1, n) * 1e5
    cash = rng.uniform(0, 1) * 1e4 * (seed % 2)
    w = rng.dirichlet(np.ones(n))
    c = rng.choice([0.0, 0.0005, 0.001, 0.0025, 0.01])
    tr = execute_trade(hold, cash, w, c)
    v_pre = hold.sum() + cash
    b = hold / v_pre
    assert tr.p + c * np.abs(tr.p * w - b).sum() == pytest.approx(1.0, abs=1e-14)
    assert v_pre - tr.v_post == pytest.approx(c * np.abs(tr.traded).sum(), rel=1e-10, abs=1e-8)
    np.testing.assert_allclose(tr.traded, tr.v_post * w - v_pre * b, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(tr.post_holdings / tr.v_post, w, rtol=1e-12)


def _two_asset_panel():
    dates = pd.bdate_range("2000-12-01", "2001-02-28", name="date")
    dates = dates[dates != pd.Timestamp("2001-01-01")]  # holiday: no trading row
    r = pd.DataFrame({"A": 0.01, "B": -0.01}, index=dates)
    return make_panel(r, 0.0)


def test_hand_calculated_drifting_portfolio():
    panel = _two_asset_panel()
    spec = RunSpec(
        "t", ("EQW",), AllocationParams(5, 21, 0.75, 0.5, 0.0), 2, 1_000_000.0, dt.date(2001, 1, 1)
    )
    res = run_backtest(panel, spec, Optimizer())
    d = res.daily.set_index("date")
    # Anchor at the Dec-29 signal; Jan-2 execution buys 50/50 at Jan-2 close (no return on cash, rf = 0).
    assert d.loc["2001-01-02", "nav_post"] == pytest.approx(1_000_000.0)
    # Through Jan 31 (21 trading days after Jan 2), holdings drift with no rebalancing.
    k = len(pd.bdate_range("2001-01-03", "2001-01-31"))
    expected = 500_000 * 1.01**k + 500_000 * 0.99**k
    assert d.loc["2001-01-31", "nav_post"] == pytest.approx(expected, rel=1e-12)
    wA = 500_000 * 1.01**k / expected
    w = res.weights
    sig = w[
        (w["signal_date"] == pd.Timestamp("2001-01-31")) & (w["weight_type"] == "signal_holding")
    ].set_index("asset")["value"]
    assert sig["A"] == pytest.approx(wA, rel=1e-12)
    # Feb 1 execution: pretrade weights drifted one more day, then rebalance to 50/50.
    pre = w[
        (w["date"] == pd.Timestamp("2001-02-01")) & (w["weight_type"] == "exec_pretrade")
    ].set_index("asset")["value"]
    a1, b1 = 500_000 * 1.01 ** (k + 1), 500_000 * 0.99 ** (k + 1)
    assert pre["A"] == pytest.approx(a1 / (a1 + b1), rel=1e-12)
    assert d.loc["2001-02-01", "nav_post"] == pytest.approx(a1 + b1, rel=1e-12)


def test_doubling_nav_scales_dollars_only():
    panel = gaussian_panel(n=5)
    out = []
    for nav in (1e6, 2e6):
        spec = RunSpec(
            "t",
            ("EQW", "COST_AWARE_REG_MV", "LW_MINVAR"),
            AllocationParams(5, 21, 0.75, 0.4, 0.001),
            63,
            nav,
            dt.date(2000, 7, 1),
        )
        out.append(run_backtest(panel, spec, Optimizer()))
    a, b = out
    np.testing.assert_allclose(b.daily["nav_post"], 2 * a.daily["nav_post"], rtol=1e-11)
    np.testing.assert_allclose(b.daily["fee"], 2 * a.daily["fee"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        b.daily["net_return"].fillna(0), a.daily["net_return"].fillna(0), atol=1e-13
    )
    np.testing.assert_allclose(b.weights["value"], a.weights["value"], atol=1e-9)
    np.testing.assert_allclose(
        b.reallocations["change_dollars"],
        2 * a.reallocations["change_dollars"],
        rtol=1e-8,
        atol=1e-6,
    )


def test_ledger_reconciles_every_day():
    panel = gaussian_panel(n=5)
    spec = RunSpec(
        "t",
        ("SAMPLE_MV",),
        AllocationParams(5, 21, 0.75, 0.4, 0.0025),
        63,
        1e6,
        dt.date(2000, 7, 1),
    )
    res = run_backtest(panel, spec, Optimizer())
    d = res.daily
    # nav_post = nav_pre - fee on every date; before-fee NAV dominates net NAV.
    np.testing.assert_allclose(d["nav_post"], d["nav_pre"] - d["fee"], rtol=1e-12)
    assert (d["nav_before_fees"] >= d["nav_post"] - 1e-6).all()
    ex = res.events[res.events["status"] == "executed"]
    np.testing.assert_allclose(
        ex["fee"], 0.0025 * ex["realized_trade_fraction"] * ex["nav_pre_exec"], rtol=1e-9
    )
    assert (d["nav_post"] > 0).all()
