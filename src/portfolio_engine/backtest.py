"""Chronological monthly-rebalancing simulation with daily holdings accounting.

Event order on each trading date ``t`` (each return row is the close(t-1) -> close(t) return):

1. Apply today's asset returns to held risky dollars and the aligned RF return to cash.
2. Compute pretrade NAV and drifted weights.
3. If a target frozen at an earlier signal is due today, trade to it at today's close and
   deduct the exact self-financing fee.
4. If today is a scheduled signal (last trading date of the month), estimate with data through
   today, freeze a target using only today's drifted weights, and queue it for the next
   trading date's close. It therefore first earns the close-to-close return *after* that
   execution close.
5. Record post-event holdings and the day's returns. A final signal with no later trading date
   is recorded as a pending proposal, never as an executed allocation.

Three-date example (signal Fri 1989-12-29, execute Tue 1990-01-02, first return 1990-01-03):
the 1990-01-02 row's return accrues to the *old* holdings (cash at RF here), the trade happens
at that close, and the new target's first return is the 1990-01-03 row.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .accounting import execute_trade
from .constraints import FeasibleSet
from .data import ReturnPanel
from .estimates import MomentEstimates, estimate_moments
from .optimizer import Optimizer, SolveResult
from .strategies import STRATEGIES, AllocationParams, propose

ESTIMATION_COLUMNS = [
    "signal_date",
    "lookback",
    "window_start",
    "window_end",
    "n_obs",
    "lw_alpha",
    "rho",
    "mu_sample_xs_mean",
    "mu_sample_xs_std",
    "mu_shrunk_xs_std",
    "trace_sample",
    "min_eig_sample",
    "max_eig_sample",
    "cond_sample",
    "min_eig_lw",
    "max_eig_lw",
    "cond_lw",
    "sample_repaired",
    "lw_repaired",
    "avg_correlation",
    "notes",
]


def month_end_signals(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available trading date of each calendar month.

    The final date of the panel counts only if it is the last weekday of its month (a panel
    that ends mid-month has no observed month-end signal for that month).
    """
    d = pd.DatetimeIndex(dates)
    if len(d) == 0:
        return d
    per = d.to_period("M")
    is_last = np.r_[per[1:] != per[:-1], False]
    last = d[-1]
    last_weekday = pd.Timestamp(last.year, last.month, 1) + pd.offsets.BMonthEnd(0)
    is_last[-1] = last.normalize() == last_weekday.normalize()
    return d[is_last]


@dataclass(frozen=True)
class RunSpec:
    scenario: str
    strategies: tuple[str, ...]
    params: AllocationParams
    lookback: int
    initial_nav: float
    sim_start: dt.date
    record_detail: bool = True


@dataclass
class BacktestResult:
    spec: RunSpec
    daily: pd.DataFrame
    events: pd.DataFrame
    weights: pd.DataFrame
    reallocations: pd.DataFrame
    estimation: pd.DataFrame
    estimation_vectors: pd.DataFrame
    covariances: pd.DataFrame
    assets: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def failures(self) -> int:
        return int((self.events["status"] == "failed").sum()) if len(self.events) else 0


class EstimateCache:
    """Signal-date estimates shared by all strategies and scenarios with the same lookback."""

    def __init__(self, panel: ReturnPanel, rel_tol: float = 1e-10):
        self.panel = panel
        self.rel_tol = rel_tol
        self._values = panel.returns.to_numpy(dtype=float)
        self._pos = {d: i for i, d in enumerate(panel.dates)}
        self._cache: dict[tuple[pd.Timestamp, int], MomentEstimates] = {}

    def window_bounds(self, signal: pd.Timestamp, lookback: int) -> tuple[int, int]:
        end = self._pos[signal] + 1  # inclusive of the signal date
        start = end - lookback
        if start < 0:
            raise ValueError(
                f"insufficient history: {lookback} observations required through {signal.date()}, {end} available"
            )
        return start, end

    def get(self, signal: pd.Timestamp, lookback: int) -> MomentEstimates:
        key = (signal, lookback)
        if key not in self._cache:
            s, e = self.window_bounds(signal, lookback)
            self._cache[key] = estimate_moments(self._values[s:e], self.rel_tol)
        return self._cache[key]


class AllocationCache:
    """Targets of path-independent strategies depend only on signal-date estimates and params."""

    def __init__(self) -> None:
        self._cache: dict[tuple, SolveResult] = {}

    def key(self, sid: str, signal: pd.Timestamp, lookback: int, p: AllocationParams) -> tuple:
        spec = STRATEGIES[sid]
        if spec.objective == "equal":
            return (sid, signal, p.max_weight)
        if spec.objective == "minvar":
            return (sid, signal, lookback, p.max_weight)
        rho = p.rho if spec.mean_model == "shrunk" else None
        return (sid, signal, lookback, p.gamma, p.horizon, rho, p.max_weight)

    def get(self, key: tuple) -> SolveResult | None:
        return self._cache.get(key)

    def put(self, key: tuple, res: SolveResult) -> None:
        self._cache[key] = res


def _first_signal(signals: pd.DatetimeIndex, sim_start: dt.date) -> int:
    before = np.nonzero(signals < pd.Timestamp(sim_start))[0]
    if len(before) == 0:
        raise ValueError(
            f"no month-end signal before simulation start {sim_start}; extend the data start"
        )
    return int(before[-1])


def estimation_diagnostics(
    est: MomentEstimates,
    signal: pd.Timestamp,
    lookback: int,
    rho: float,
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> dict:
    mu = est.mu_sample
    from .estimates import shrink_mean

    s = est.sample.matrix
    sd = np.sqrt(np.clip(np.diag(s), 1e-300, None))
    corr = s / np.outer(sd, sd)
    n = len(mu)
    off = corr[~np.eye(n, dtype=bool)]
    return {
        "signal_date": signal,
        "lookback": lookback,
        "window_start": window[0],
        "window_end": window[1],
        "n_obs": est.n_obs,
        "lw_alpha": est.lw_alpha,
        "rho": rho,
        "mu_sample_xs_mean": float(mu.mean()),
        "mu_sample_xs_std": float(mu.std(ddof=0)),
        "mu_shrunk_xs_std": float(shrink_mean(mu, rho).std(ddof=0)),
        "trace_sample": float(np.trace(s)),
        "min_eig_sample": float(est.sample.eigenvalues[0]),
        "max_eig_sample": float(est.sample.eigenvalues[-1]),
        "cond_sample": est.sample.condition_number,
        "min_eig_lw": float(est.lw.eigenvalues[0]),
        "max_eig_lw": float(est.lw.eigenvalues[-1]),
        "cond_lw": est.lw.condition_number,
        "sample_repaired": est.sample.repaired,
        "lw_repaired": est.lw.repaired,
        "avg_correlation": float(off.mean()) if off.size else float("nan"),
        "notes": "; ".join(est.notes),
    }


def run_backtest(
    panel: ReturnPanel,
    spec: RunSpec,
    optimizer: Optimizer,
    estimates: EstimateCache | None = None,
    alloc_cache: AllocationCache | None = None,
    progress: Callable[[str], None] | None = None,
) -> BacktestResult:
    estimates = estimates or EstimateCache(panel)
    alloc_cache = alloc_cache or AllocationCache()
    p = spec.params
    assets = panel.assets
    n = len(assets)
    feas = FeasibleSet(n, p.max_weight, optimizer.tol)
    all_dates = panel.dates
    signals_all = month_end_signals(all_dates)
    k0 = _first_signal(signals_all, spec.sim_start)
    signals = signals_all[k0:]
    anchor = signals[0]
    pos = {d: i for i, d in enumerate(all_dates)}
    a = pos[anchor]
    path = all_dates[a:]
    R = panel.returns.to_numpy(dtype=float)[a:]
    RF = panel.rf.to_numpy(dtype=float)[a:]
    signal_set = set(signals)
    exec_of = {s: (all_dates[pos[s] + 1] if pos[s] + 1 < len(all_dates) else None) for s in signals}
    c = p.cost_rate

    # Shared estimation diagnostics (independent of strategy).
    est_rows, vec_rows, cov_rows = [], [], []
    needs_est = any(STRATEGIES[sid].uses_estimates for sid in spec.strategies)
    for s in signals if needs_est else pd.DatetimeIndex([]):
        st, en = estimates.window_bounds(s, spec.lookback)
        est = estimates.get(s, spec.lookback)
        est_rows.append(
            estimation_diagnostics(est, s, spec.lookback, p.rho, (all_dates[st], all_dates[en - 1]))
        )
        if spec.record_detail:
            from .estimates import shrink_mean

            mu_sh = shrink_mean(est.mu_sample, p.rho)
            for i, asset in enumerate(assets):
                vec_rows.append(
                    (
                        s,
                        asset,
                        i,
                        est.mu_sample[i],
                        mu_sh[i],
                        est.sample.eigenvalues[::-1][i],
                        est.lw.eigenvalues[::-1][i],
                    )
                )
            for kind, mat in (("sample", est.sample.matrix), ("lw", est.lw.matrix)):
                for i, asset in enumerate(assets):
                    cov_rows.append((s, kind, asset, *mat[i]))

    daily_frames, event_rows, weight_rows, realloc_rows = [], [], [], []
    for sid in spec.strategies:
        strat = STRATEGIES[sid]
        holdings = np.zeros(n)
        cash = float(spec.initial_nav)
        nav_bf = float(spec.initial_nav)
        pending: dict | None = None
        T = len(path)
        cols = {
            k: np.zeros(T)
            for k in (
                "nav_pre",
                "nav_post",
                "nav_before_fees",
                "gross_return",
                "net_return",
                "fee",
                "gross_traded",
                "turnover_gross",
                "cash_weight",
                "hhi",
                "max_weight",
            )
        }
        eff_n = np.full(T, np.nan)
        drift_ex = np.zeros(T, dtype=np.int64)
        flags = {
            k: np.zeros(T, dtype=bool)
            for k in ("is_anchor", "is_signal", "is_execution", "is_initial_deployment")
        }
        status_col = np.array([""] * T, dtype=object)
        ever_invested = False
        for t in range(T):
            d = path[t]
            nav_prev = holdings.sum() + cash
            if t == 0:
                flags["is_anchor"][0] = True
                nav_pre = nav_prev
                gross = np.nan
            else:
                holdings = holdings * (1.0 + R[t])
                cash = cash * (1.0 + RF[t])
                nav_pre = holdings.sum() + cash
                gross = nav_pre / nav_prev - 1.0
                nav_bf *= 1.0 + gross
            if nav_pre <= 0:
                raise RuntimeError(f"{sid}: nonpositive NAV on {d.date()}")
            fee = 0.0
            if pending is not None and pending["execution_date"] == d:
                initial = not ever_invested
                tr = execute_trade(holdings, cash, pending["target"], c)
                holdings, cash = tr.post_holdings, tr.cash_post
                fee = tr.fee
                ever_invested = True
                flags["is_execution"][t] = True
                flags["is_initial_deployment"][t] = initial
                cols["gross_traded"][t] = tr.gross_traded
                cols["turnover_gross"][t] = tr.turnover_gross
                ev = pending["event"]
                ev.update(
                    {
                        "status": "executed",
                        "is_initial_deployment": initial,
                        "nav_pre_exec": tr.v_pre,
                        "nav_post_exec": tr.v_post,
                        "fee": tr.fee,
                        "realized_trade_fraction": tr.turnover_gross,
                        "realized_half_turnover": tr.turnover_half,
                        "realized_cost_fraction": tr.fee / tr.v_pre,
                        "cost_fraction_p": tr.p,
                        "exec_drift_from_signal": float(
                            np.abs(tr.pre_holdings / tr.v_pre - pending["w_signal"]).sum()
                        ),
                    }
                )
                if spec.record_detail:
                    sd = pending["event"]["signal_date"]
                    for i, asset in enumerate(assets):
                        weight_rows.append(
                            (d, sd, sid, asset, "exec_pretrade", tr.pre_holdings[i] / tr.v_pre)
                        )
                        weight_rows.append(
                            (d, sd, sid, asset, "exec_posttrade", tr.post_holdings[i] / tr.v_post)
                        )
                        realloc_rows.append(
                            (
                                sd,
                                d,
                                sid,
                                asset,
                                tr.pre_holdings[i],
                                tr.post_holdings[i],
                                tr.traded[i],
                            )
                        )
                pending = None
            nav_post = holdings.sum() + cash
            w_now = holdings / nav_post
            if d in signal_set:
                flags["is_signal"][t] = True
                w_signal = w_now.copy()
                est_s: MomentEstimates | None = (
                    estimates.get(d, spec.lookback) if strat.uses_estimates else None
                )
                key = alloc_cache.key(sid, d, spec.lookback, p)
                res = None if strat.path_dependent else alloc_cache.get(key)
                if res is None:
                    res = propose(strat, est_s, w_signal, p, optimizer)
                    if not strat.path_dependent:
                        alloc_cache.put(key, res)
                est_lw = estimates.get(d, spec.lookback) if needs_est else None
                ev = {
                    "strategy": sid,
                    "scenario": spec.scenario,
                    "signal_date": d,
                    "execution_date": exec_of[d],
                    "status": "failed"
                    if not res.ok
                    else ("pending" if exec_of[d] is None else "queued"),
                    "solver": res.solver,
                    "solver_status": res.solver_status,
                    "objective": res.objective,
                    "solve_time_s": res.solve_time,
                    "residual_budget": res.residuals.get("budget"),
                    "residual_lower": res.residuals.get("lower"),
                    "residual_upper": res.residuals.get("upper"),
                    "residual_max": res.residuals.get("max"),
                    "cleanup_adjustment": res.cleanup,
                    "solver_attempts": json.dumps(res.attempts),
                    "nav_signal": nav_post,
                    "signal_cash_weight": cash / nav_post,
                    "target_cap_hits": feas.cap_hits(res.weights)
                    if res.weights is not None
                    else np.nan,
                    "feasibility_violation": bool(
                        res.ok and res.residuals.get("max", 0.0) > feas.tol
                    ),
                    "is_initial_deployment": False,
                }
                if res.weights is not None:
                    w = res.weights
                    planned = float(np.abs(w - w_signal).sum())
                    ev.update(
                        {
                            "planned_trade_fraction": planned,
                            "planned_half_turnover": 0.5 * planned,
                            "planned_cost_fraction": c * planned,
                            "forecast_vol_lw_ann": float(np.sqrt(252.0 * w @ est_lw.lw.matrix @ w))
                            if est_lw
                            else np.nan,
                            "forecast_vol_sample_ann": float(
                                np.sqrt(252.0 * w @ est_lw.sample.matrix @ w)
                            )
                            if est_lw
                            else np.nan,
                            "target_hhi": float(w @ w),
                            "target_effective_n": float(1.0 / (w @ w)),
                            "target_max_weight": float(w.max()),
                        }
                    )
                    if spec.record_detail:
                        for i, asset in enumerate(assets):
                            weight_rows.append((d, d, sid, asset, "signal_holding", w_signal[i]))
                            weight_rows.append((d, d, sid, asset, "target", w[i]))
                    if exec_of[d] is not None:
                        pending = {
                            "execution_date": exec_of[d],
                            "target": w,
                            "w_signal": w_signal,
                            "event": ev,
                        }
                elif spec.record_detail:
                    for i, asset in enumerate(assets):
                        weight_rows.append((d, d, sid, asset, "signal_holding", w_signal[i]))
                event_rows.append(ev)
                status_col[t] = ev["status"]
            cols["nav_pre"][t] = nav_pre
            cols["nav_post"][t] = nav_post
            cols["nav_before_fees"][t] = nav_bf
            cols["gross_return"][t] = gross
            cols["net_return"][t] = np.nan if t == 0 else nav_post / nav_prev - 1.0
            cols["fee"][t] = fee
            cols["cash_weight"][t] = cash / nav_post
            hhi = float(w_now @ w_now)
            cols["hhi"][t] = hhi
            cols["max_weight"][t] = float(w_now.max())
            if hhi > 0:
                eff_n[t] = 1.0 / hhi
            drift_ex[t] = feas.drift_exceedances(w_now) if not flags["is_execution"][t] else 0
        df = pd.DataFrame(
            {
                "date": path,
                "strategy": sid,
                "scenario": spec.scenario,
                **cols,
                "rf": np.r_[np.nan, RF[1:]],
                "turnover_half": cols["turnover_gross"] / 2.0,
                "effective_n": eff_n,
                "drift_cap_exceedances": drift_ex,
                **flags,
                "event_status": status_col.astype(str),
            }
        )
        daily_frames.append(df)
        if progress:
            progress(sid)

    daily = pd.concat(daily_frames, ignore_index=True)
    events = pd.DataFrame(event_rows)
    if len(events):
        events.loc[events["status"] == "queued", "status"] = "unexecuted"
        events = _add_realized_vol(events, daily)
    weights = pd.DataFrame(
        weight_rows, columns=["date", "signal_date", "strategy", "asset", "weight_type", "value"]
    )
    weights["scenario"] = spec.scenario
    realloc = pd.DataFrame(
        realloc_rows,
        columns=[
            "signal_date",
            "execution_date",
            "strategy",
            "asset",
            "pretrade_dollars",
            "posttrade_dollars",
            "change_dollars",
        ],
    )
    realloc["scenario"] = spec.scenario
    estimation = pd.DataFrame(est_rows, columns=ESTIMATION_COLUMNS)
    vectors = pd.DataFrame(
        vec_rows,
        columns=[
            "signal_date",
            "asset",
            "rank",
            "mu_sample",
            "mu_shrunk",
            "eig_sample_desc",
            "eig_lw_desc",
        ],
    )
    covs = pd.DataFrame(cov_rows, columns=["signal_date", "kind", "asset", *assets])
    return BacktestResult(spec, daily, events, weights, realloc, estimation, vectors, covs, assets)


def _add_realized_vol(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Ex post realized volatility of gross policy returns over each complete holding interval.

    Interval: trading days after an execution close up to and including the next execution
    close (about one month, so estimates are noisy). Purely diagnostic: never used in allocation.
    """
    events = events.copy()
    events["realized_vol_ann"] = np.nan
    events["holding_days"] = np.nan
    for sid, g in events[events["status"] == "executed"].groupby("strategy"):
        d = daily[daily["strategy"] == sid].set_index("date")["gross_return"]
        ex = list(g["execution_date"])
        for k in range(len(ex) - 1):
            seg = d.loc[(d.index > ex[k]) & (d.index <= ex[k + 1])]
            if len(seg) >= 2:
                idx = g.index[k]
                events.loc[idx, "realized_vol_ann"] = float(np.sqrt(252.0) * seg.std(ddof=1))
                events.loc[idx, "holding_days"] = len(seg)
    return events
