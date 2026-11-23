"""Metric formulas against independent hand calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_engine.experiments import block_indices, paired_block_bootstrap
from portfolio_engine.metrics import SplitWindow, cagr, sharpe_ratio, split_metrics


def _daily(navs, dates, rf=0.0, fees=None, gross=None, execs=None, initial=None):
    navs = np.asarray(navs, dtype=float)
    net = np.r_[np.nan, navs[1:] / navs[:-1] - 1]
    n = len(navs)
    fees = np.zeros(n) if fees is None else np.asarray(fees, float)
    nav_pre = navs + fees
    g = np.r_[np.nan, nav_pre[1:] / navs[:-1] - 1] if gross is None else gross
    bf = navs[0] * np.r_[1.0, np.cumprod(1 + g[1:])]
    execs = np.zeros(n, bool) if execs is None else np.asarray(execs)
    initial = np.zeros(n, bool) if initial is None else np.asarray(initial)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "strategy": "X",
            "scenario": "s",
            "nav_post": navs,
            "nav_pre": nav_pre,
            "nav_before_fees": bf,
            "net_return": net,
            "gross_return": g,
            "rf": rf,
            "fee": fees,
            "turnover_gross": np.where(execs, 0.4, 0.0),
            "turnover_half": np.where(execs, 0.2, 0.0),
            "hhi": 0.5,
            "effective_n": 2.0,
            "max_weight": 0.5,
            "drift_cap_exceedances": 0,
            "is_execution": execs,
            "is_initial_deployment": initial,
        }
    )


def test_sharpe_hand_calculation_and_undefined_cases():
    net = np.array([0.01, -0.005, 0.002, 0.004])
    rf = np.array([0.0001] * 4)
    ex = net - rf
    expect = np.sqrt(252) * ex.mean() / ex.std(ddof=1)
    assert sharpe_ratio(net, rf)[0] == pytest.approx(expect, rel=1e-12)
    s, why = sharpe_ratio(np.full(5, 0.0001), np.full(5, 0.0001))
    assert np.isnan(s) and "zero" in why
    s, why = sharpe_ratio(np.array([0.01]), np.array([0.0]))
    assert np.isnan(s) and "fewer" in why


def test_split_anchor_keeps_first_return_and_resets_drawdown_peak():
    dates = ["2004-12-30", "2004-12-31", "2005-01-03", "2005-01-04", "2005-01-05"]
    navs = [100.0, 200.0, 180.0, 190.0, 150.0]  # big pre-split peak at 200
    d = _daily(navs, dates)
    m = split_metrics(
        d, pd.DataFrame(), SplitWindow("v", pd.Timestamp("2005-01-01"), pd.Timestamp("2005-12-31"))
    )
    assert m["anchor_nav"] == 200.0 and str(m["anchor_date"]) == "2004-12-31"
    assert m["total_return"] == pytest.approx(
        150 / 200 - 1
    )  # first in-split return (-10%) included
    # Peak resets at the anchor (200): max DD = 150/200 - 1, peak date = anchor.
    assert m["max_drawdown"] == pytest.approx(-0.25)
    assert str(m["mdd_peak_date"]) == "2004-12-31" and str(m["mdd_trough_date"]) == "2005-01-05"
    # A later split must not import the earlier 200 peak.
    m2 = split_metrics(
        d, pd.DataFrame(), SplitWindow("w", pd.Timestamp("2005-01-04"), pd.Timestamp("2005-12-31"))
    )
    assert m2["anchor_nav"] == 180.0
    assert m2["max_drawdown"] == pytest.approx(150 / 190 - 1)


def test_cagr_actual_elapsed_time():
    v = cagr(121.0, 100.0, pd.Timestamp("2000-01-01"), pd.Timestamp("2002-01-01"))
    years = 731 / 365.25
    assert v == pytest.approx(1.21 ** (1 / years) - 1)


def test_fee_treatment_and_recurring_turnover():
    dates = pd.bdate_range("2005-01-03", periods=6)
    navs = [100.0, 99.0, 101.0, 100.0, 102.0, 103.0]
    fees = [0.0, 1.0, 0.0, 0.5, 0.0, 0.0]
    execs = [False, True, False, True, False, False]
    initial = [False, True, False, False, False, False]
    d = _daily(navs, dates, fees=fees, execs=execs, initial=initial)
    w = SplitWindow("x", pd.Timestamp("2005-01-04"), pd.Timestamp("2005-12-31"))
    m = split_metrics(d, pd.DataFrame(), w)
    assert m["total_fees"] == pytest.approx(1.5)
    assert m["initial_deployment_fees"] == pytest.approx(1.0) and m[
        "recurring_fees"
    ] == pytest.approx(0.5)
    assert m["recurring_half_turnover_ann"] == pytest.approx(0.2 / (5 / 252))
    assert m["gross_cagr_before_fees"] > m["net_cagr"]


def test_block_bootstrap_pairs_rows_and_preserves_blocks():
    rng = np.random.default_rng(0)
    idx = block_indices(100, 20, 50, rng)
    assert idx.shape == (50, 100)
    starts = idx[:, ::20]
    np.testing.assert_array_equal(idx[:, :20], starts[:, :1] + np.arange(20))
    dates = pd.bdate_range("2010-01-01", periods=300)
    r = rng.normal(0.0004, 0.01, 300)
    rows = []
    for sid, ret in (("EQW", r), ("SAME", r.copy()), ("OTHER", r + rng.normal(0, 0.002, 300))):
        rows.append(pd.DataFrame({"date": dates, "strategy": sid, "net_return": ret, "rf": 0.0001}))
    daily = pd.concat(rows)
    out = paired_block_bootstrap(
        daily, SplitWindow("s", dates[0], dates[-1]), 20, 200, 1
    ).set_index("strategy")
    # Identical return series => paired differences are exactly zero in every replication.
    assert out.loc["SAME", "boot_se"] == 0.0 and out.loc["SAME", "ci_low"] == 0.0
    assert not out.loc["SAME", "interval_excludes_zero"]
    assert out.loc["OTHER", "boot_se"] > 0
