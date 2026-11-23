"""Signal/execution lag, no look-ahead, pending proposals, drift vs. solver violations (SYNTHETIC)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from portfolio_engine.backtest import RunSpec, month_end_signals, run_backtest
from portfolio_engine.optimizer import Optimizer
from portfolio_engine.strategies import STRATEGY_IDS, AllocationParams
from tests.conftest import gaussian_panel, make_panel


def test_month_end_signals():
    d = pd.bdate_range("2001-01-01", "2001-03-14")
    s = month_end_signals(d)
    assert list(s.strftime("%Y-%m-%d")) == [
        "2001-01-31",
        "2001-02-28",
    ]  # mid-March end is not a signal


def test_three_date_example_execution_lag():
    """Signal 2000-12-29 -> execute 2001-01-02 close -> first return of new weights on 2001-01-03."""
    dates = pd.bdate_range("2000-11-01", "2001-01-10", name="date")
    dates = dates[dates != pd.Timestamp("2001-01-01")]  # New Year's Day holiday: no trading row
    r = pd.DataFrame({"A": 0.0, "B": 0.0}, index=dates)
    r.loc["2001-01-02", "A"] = 0.50  # execution-day return must NOT accrue to the new target
    r.loc["2001-01-03", "A"] = 0.10  # first return earned by the new target
    rf = pd.Series(0.0001, index=dates, name="rf")
    panel = make_panel(r, rf)
    spec = RunSpec(
        "t", ("EQW",), AllocationParams(5, 21, 0.75, 0.5, 0.001), 5, 1e6, dt.date(2001, 1, 1)
    )
    res = run_backtest(panel, spec, Optimizer())
    d = res.daily.set_index("date")
    assert d.loc["2000-12-29", "is_anchor"] and d.loc["2000-12-29", "is_signal"]
    assert d.loc["2001-01-02", "is_execution"]
    # Jan 2: all cash earns RF; then the fee for buying from cash is deducted at the close.
    assert d.loc["2001-01-02", "gross_return"] == pytest.approx(0.0001, abs=1e-15)
    nav_pre = 1e6 * 1.0001
    fee = nav_pre * 0.001 / 1.001
    assert d.loc["2001-01-02", "fee"] == pytest.approx(fee, rel=1e-10)
    # Jan 3: the new 50/50 target earns 0.5 * 10% + 0.5 * 0%.
    assert d.loc["2001-01-03", "gross_return"] == pytest.approx(0.05, rel=1e-12)
    ev = res.events.iloc[0]
    assert ev["signal_date"] == pd.Timestamp("2000-12-29") and ev["execution_date"] == pd.Timestamp(
        "2001-01-02"
    )


def test_future_perturbation_cannot_change_earlier_targets():
    base = gaussian_panel(n=6, end="2002-06-28")
    cut = pd.Timestamp("2001-06-29")  # a month-end signal date
    pert_r = base.returns.copy()
    rng = np.random.default_rng(99)
    after = pert_r.index > cut
    pert_r.loc[after] = rng.normal(0.01, 0.05, (after.sum(), pert_r.shape[1]))
    pert = make_panel(pert_r, base.rf)
    spec = RunSpec(
        "t", STRATEGY_IDS, AllocationParams(5, 21, 0.75, 0.3, 0.001), 63, 1e6, dt.date(2000, 7, 1)
    )
    a = run_backtest(base, spec, Optimizer())
    b = run_backtest(pert, spec, Optimizer())
    for res in (a, b):
        assert (res.events["status"] != "failed").all()
    wa = a.weights[
        (a.weights["signal_date"] <= cut)
        & a.weights["weight_type"].isin(["target", "signal_holding"])
    ]
    wb = b.weights[
        (b.weights["signal_date"] <= cut)
        & b.weights["weight_type"].isin(["target", "signal_holding"])
    ]
    pd.testing.assert_frame_equal(wa.reset_index(drop=True), wb.reset_index(drop=True))
    ea = a.estimation[a.estimation["signal_date"] <= cut].reset_index(drop=True)
    eb = b.estimation[b.estimation["signal_date"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(ea, eb)
    assert (ea["window_end"] <= cut).all()
    # ...while later targets do change.
    later_a = a.weights[
        (a.weights["signal_date"] > cut)
        & (a.weights["weight_type"] == "target")
        & (a.weights["strategy"] == "SAMPLE_MV")
    ]
    later_b = b.weights[
        (b.weights["signal_date"] > cut)
        & (b.weights["weight_type"] == "target")
        & (b.weights["strategy"] == "SAMPLE_MV")
    ]
    assert not np.allclose(later_a["value"].to_numpy(), later_b["value"].to_numpy())


def test_final_signal_is_pending_and_signal_vs_execution_weights_differ():
    panel = gaussian_panel(n=6)
    spec = RunSpec(
        "t",
        ("COST_AWARE_REG_MV",),
        AllocationParams(5, 21, 0.75, 0.3, 0.001),
        63,
        1e6,
        dt.date(2000, 7, 1),
    )
    res = run_backtest(panel, spec, Optimizer())
    last = res.events.iloc[-1]
    assert last["status"] == "pending" and pd.isna(last["execution_date"])
    assert last["signal_date"] == panel.dates[-1]
    ex = res.events[
        (res.events["status"] == "executed") & ~res.events["is_initial_deployment"].astype(bool)
    ]
    # Execution-day drift: realized trades differ from those planned at the signal close.
    assert (ex["exec_drift_from_signal"] > 0).all()
    assert not np.allclose(ex["planned_trade_fraction"], ex["realized_trade_fraction"])


def test_drift_exceedance_reported_separately_from_solver_violations():
    panel = gaussian_panel(n=6)
    spec = RunSpec(
        "t", ("SAMPLE_MV",), AllocationParams(5, 21, 0.75, 0.2, 0.001), 63, 1e6, dt.date(2000, 7, 1)
    )
    res = run_backtest(panel, spec, Optimizer())
    assert res.daily["drift_cap_exceedances"].sum() > 0
    assert not res.events["feasibility_violation"].any()
    tgt = res.weights[res.weights["weight_type"].isin(["target", "exec_posttrade"])]
    assert tgt["value"].max() <= 0.2 + 1e-9


def test_failed_allocation_keeps_holdings(monkeypatch):
    from portfolio_engine import backtest as bt
    from portfolio_engine.optimizer import SolveResult

    panel = gaussian_panel(n=4)
    calls = {"n": 0}
    real = bt.propose

    def flaky(spec, est, w, p, opt):
        calls["n"] += 1
        if calls["n"] == 3:
            return SolveResult(
                None,
                "failed",
                None,
                None,
                None,
                0.0,
                {"max": np.nan},
                attempts=[{"solver": "X", "outcome": "forced"}],
            )
        return real(spec, est, w, p, opt)

    monkeypatch.setattr(bt, "propose", flaky)
    spec = RunSpec(
        "t", ("SAMPLE_MV",), AllocationParams(5, 21, 0.75, 0.5, 0.001), 63, 1e6, dt.date(2000, 7, 1)
    )
    res = run_backtest(panel, spec, Optimizer())
    failed = res.events[res.events["status"] == "failed"]
    assert len(failed) == 1 and res.failures == 1
    sd = failed["signal_date"].iloc[0]
    nxt = panel.dates[panel.dates.get_loc(sd) + 1]
    day = res.daily.set_index("date").loc[nxt]
    assert not day["is_execution"] and day["fee"] == 0.0
