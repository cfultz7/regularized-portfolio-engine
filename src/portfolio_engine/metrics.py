"""Split-level performance metrics with explicit NAV anchors.

Definitions (daily decimal returns; 252 trading days per year for annualization):

* Split anchor: post-event NAV at the last trading close *before* the split's first date
  (the initial NAV if the path starts inside the split). Returns and drawdowns are measured
  from the anchor, so the first in-split return is never lost and the drawdown peak resets.
* Net CAGR: ``(NAV_end / NAV_anchor) ** (1 / years) - 1`` with
  ``years = (end_date - anchor_date).days / 365.25`` (actual elapsed calendar time).
* Annualized volatility: ``sqrt(252) * std(net_return, ddof=1)`` of raw daily net returns.
* Sharpe: ``sqrt(252) * mean(net_return - rf) / std(net_return - rf, ddof=1)`` using daily
  arithmetic excess returns. Square-root-of-time scaling assumes serially uncorrelated
  returns; with dependence it is only a convention.
* Maximum drawdown: largest peak-to-trough decline of the anchored net NAV path.
* Recurring half-gross turnover (annualized): sum of half-gross turnover over non-initial
  executions in the split divided by (number of trading days in the split / 252).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class SplitWindow:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def anchored_slice(
    daily_one: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, nav_col: str = "nav_post"
) -> tuple[pd.DataFrame, float, pd.Timestamp]:
    """Rows in [start, end] plus the anchor NAV/date immediately preceding ``start``."""
    d = daily_one.sort_values("date")
    inside = d[(d["date"] >= start) & (d["date"] <= end)]
    before = d[d["date"] < start]
    if len(before):
        anchor_nav, anchor_date = float(before[nav_col].iloc[-1]), before["date"].iloc[-1]
    else:
        first = d.iloc[0]
        anchor_nav, anchor_date = float(first[nav_col]), first["date"]
        inside = inside[inside["date"] > anchor_date]
    return inside, anchor_nav, pd.Timestamp(anchor_date)


def drawdown(nav: np.ndarray, anchor: float) -> np.ndarray:
    path = np.r_[anchor, nav]
    peak = np.maximum.accumulate(path)
    return (path / peak - 1.0)[1:]


def max_drawdown(
    dates: pd.Series, nav: np.ndarray, anchor: float, anchor_date: pd.Timestamp
) -> dict:
    if len(nav) == 0:
        return {
            "max_drawdown": np.nan,
            "mdd_peak_date": None,
            "mdd_trough_date": None,
            "mdd_recovery_date": None,
        }
    path = np.r_[anchor, nav]
    all_dates = [anchor_date, *list(pd.to_datetime(dates))]
    peak = np.maximum.accumulate(path)
    dd = path / peak - 1.0
    trough = int(np.argmin(dd))
    if dd[trough] >= 0:
        return {
            "max_drawdown": 0.0,
            "mdd_peak_date": None,
            "mdd_trough_date": None,
            "mdd_recovery_date": None,
        }
    peak_idx = int(np.argmax(path[: trough + 1]))
    rec = np.nonzero(path[trough:] >= path[peak_idx])[0]
    return {
        "max_drawdown": float(dd[trough]),
        "mdd_peak_date": all_dates[peak_idx].date(),
        "mdd_trough_date": all_dates[trough].date(),
        "mdd_recovery_date": all_dates[trough + int(rec[0])].date() if len(rec) else None,
    }


def sharpe_ratio(net: np.ndarray, rf: np.ndarray) -> tuple[float, str | None]:
    ex = np.asarray(net, dtype=float) - np.asarray(rf, dtype=float)
    if ex.size < 2:
        return np.nan, "fewer than two observations"
    sd = ex.std(ddof=1)
    if not np.isfinite(sd) or sd <= 1e-15:
        return np.nan, "zero excess-return volatility"
    return float(np.sqrt(TRADING_DAYS) * ex.mean() / sd), None


def cagr(
    nav_end: float, nav_anchor: float, anchor_date: pd.Timestamp, end_date: pd.Timestamp
) -> float:
    years = (pd.Timestamp(end_date) - pd.Timestamp(anchor_date)).days / 365.25
    if years <= 0 or nav_anchor <= 0:
        return np.nan
    return float((nav_end / nav_anchor) ** (1.0 / years) - 1.0)


def split_metrics(daily_one: pd.DataFrame, events_one: pd.DataFrame, window: SplitWindow) -> dict:
    inside, anchor_nav, anchor_date = anchored_slice(daily_one, window.start, window.end)
    out: dict = {
        "split": window.name,
        "split_start": window.start.date(),
        "split_end": window.end.date(),
        "anchor_date": anchor_date.date(),
        "anchor_nav": anchor_nav,
        "n_days": len(inside),
    }
    if len(inside) == 0:
        out["note"] = "no observations"
        return out
    nav = inside["nav_post"].to_numpy()
    net = inside["net_return"].to_numpy()
    gross = inside["gross_return"].to_numpy()
    rf = inside["rf"].to_numpy()
    end_date = inside["date"].iloc[-1]
    _, bf_anchor, _ = anchored_slice(daily_one, window.start, window.end, "nav_before_fees")
    bf_end = float(inside["nav_before_fees"].iloc[-1])
    sr, why = sharpe_ratio(net, rf)
    sr_g, _ = sharpe_ratio(gross, rf)
    rf_growth = float(np.prod(1.0 + rf))
    out.update(
        {
            "end_nav": float(nav[-1]),
            "total_return": float(nav[-1] / anchor_nav - 1.0),
            "net_cagr": cagr(nav[-1], anchor_nav, anchor_date, end_date),
            "gross_cagr_before_fees": cagr(bf_end, bf_anchor, anchor_date, end_date),
            "ann_vol": float(np.sqrt(TRADING_DAYS) * net.std(ddof=1)) if len(net) > 1 else np.nan,
            "sharpe": sr,
            "sharpe_undefined_reason": why,
            "sharpe_gross_before_fees": sr_g,
            "rf_cagr": cagr(rf_growth, 1.0, anchor_date, end_date),
            **max_drawdown(inside["date"], nav, anchor_nav, anchor_date),
        }
    )
    invested = inside[inside["hhi"] > 0]
    out["avg_hhi"] = float(invested["hhi"].mean()) if len(invested) else np.nan
    out["avg_effective_n"] = float(invested["effective_n"].mean()) if len(invested) else np.nan
    out["avg_max_weight"] = float(invested["max_weight"].mean()) if len(invested) else np.nan
    ex = inside[inside["is_execution"]]
    rec = ex[~ex["is_initial_deployment"]]
    years = len(inside) / TRADING_DAYS
    out.update(
        {
            "n_executions": len(ex),
            "recurring_half_turnover_ann": float(rec["turnover_half"].sum() / years)
            if years > 0
            else np.nan,
            "recurring_gross_turnover_ann": float(rec["turnover_gross"].sum() / years)
            if years > 0
            else np.nan,
            "initial_deployment_in_split": bool(ex["is_initial_deployment"].any()),
            "initial_deployment_fees": float(ex.loc[ex["is_initial_deployment"], "fee"].sum()),
            "total_fees": float(inside["fee"].sum()),
            "recurring_fees": float(rec["fee"].sum()),
            "fees_pct_of_anchor_nav": float(inside["fee"].sum() / anchor_nav),
            "drift_cap_exceedance_days": int((inside["drift_cap_exceedances"] > 0).sum()),
        }
    )
    if len(events_one):
        ev = events_one[
            (events_one["signal_date"] >= window.start) & (events_one["signal_date"] <= window.end)
        ]
        out["n_signals"] = len(ev)
        out["target_cap_hits"] = float(ev["target_cap_hits"].sum(skipna=True))
        out["avg_target_cap_hits"] = (
            float(ev["target_cap_hits"].mean(skipna=True)) if len(ev) else np.nan
        )
        out["solver_failures"] = int((ev["status"] == "failed").sum())
        out["feasibility_violations"] = int(ev["feasibility_violation"].sum())
        out["accepted_inaccurate"] = int((ev["solver_status"] == "optimal_inaccurate").sum())
    return out


def compute_metrics(
    daily: pd.DataFrame, events: pd.DataFrame, windows: list[SplitWindow]
) -> pd.DataFrame:
    rows = []
    for (scenario, sid), g in daily.groupby(["scenario", "strategy"], sort=False):
        ev = (
            events[(events["strategy"] == sid) & (events["scenario"] == scenario)]
            if len(events)
            else events
        )
        for w in windows:
            rows.append({"strategy": sid, "scenario": scenario, **split_metrics(g, ev, w)})
    return pd.DataFrame(rows)


def rebased_paths(
    daily: pd.DataFrame, window: SplitWindow, nav_col: str = "nav_post"
) -> pd.DataFrame:
    """Wealth relative to each strategy's split anchor (anchor row included as 1.0)."""
    frames = []
    for (scenario, sid), g in daily.groupby(["scenario", "strategy"], sort=False):
        inside, anchor, adate = anchored_slice(g, window.start, window.end, nav_col)
        vals = inside[nav_col].to_numpy() / anchor
        dd = drawdown(inside["nav_post"].to_numpy(), anchored_slice(g, window.start, window.end)[1])
        frames.append(
            pd.DataFrame(
                {
                    "date": [adate, *inside["date"]],
                    "strategy": sid,
                    "scenario": scenario,
                    "wealth": np.r_[1.0, vals],
                    "drawdown": np.r_[0.0, dd],
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
