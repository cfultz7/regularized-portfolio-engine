"""Primary run, prespecified one-factor-at-a-time sweeps, uncertainty analysis, and the
``reproduce`` pipeline that writes a complete, manifest-tracked run directory."""

from __future__ import annotations

import json
import math
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    Manifest,
    RunPaths,
    code_identifier,
    dependency_versions,
    environment_info,
    make_run_id,
    save_table,
    utc_now_iso,
    write_csv,
    write_json,
)
from .backtest import AllocationCache, BacktestResult, EstimateCache, RunSpec, run_backtest
from .config import Config
from .data import ReturnPanel, load_panel
from .metrics import SplitWindow, compute_metrics
from .optimizer import Optimizer
from .strategies import STRATEGIES, STRATEGY_IDS, AllocationParams

PRIMARY = "primary"
Log = Callable[[str], None]


def _print(msg: str) -> None:
    print(msg, flush=True)


def split_windows(cfg: Config) -> list[SplitWindow]:
    return [
        SplitWindow(s.name, pd.Timestamp(s.start), pd.Timestamp(s.end)) for s in cfg.splits.splits
    ]


def base_params(cfg: Config, **over: Any) -> AllocationParams:
    p = cfg.portfolio
    vals: dict[str, Any] = {
        "gamma": p.gamma,
        "horizon": p.horizon,
        "rho": p.rho,
        "max_weight": p.max_weight,
        "cost_rate": p.cost_rate,
    }
    vals.update(over)
    return AllocationParams(**vals)


def primary_spec(cfg: Config) -> RunSpec:
    return RunSpec(
        PRIMARY,
        STRATEGY_IDS,
        base_params(cfg),
        cfg.portfolio.lookback,
        cfg.portfolio.initial_nav,
        cfg.splits.simulation_start,
    )


# --------------------------------------------------------------------------------------
# Sensitivity sweeps (one factor at a time around the frozen primary configuration)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepCell:
    sweep: str
    parameter: str
    value: float
    strategies: tuple[str, ...]
    params: AllocationParams
    lookback: int
    is_primary: bool

    @property
    def scenario(self) -> str:
        v = int(self.value) if float(self.value).is_integer() else self.value
        return f"{self.parameter}={v}"


def sweep_cells(cfg: Config) -> list[SweepCell]:
    p = cfg.portfolio
    e = cfg.experiments
    cells = []
    for bps in e.cost_bps:
        cells.append(
            SweepCell(
                "cost",
                "cost_bps",
                bps,
                STRATEGY_IDS,
                base_params(cfg, cost_rate=bps / 1e4),
                p.lookback,
                math.isclose(bps, p.cost_bps),
            )
        )
    for lb in e.lookback:
        cells.append(
            SweepCell(
                "lookback", "lookback", lb, STRATEGY_IDS, base_params(cfg), lb, lb == p.lookback
            )
        )
    for rho in e.rho:
        cells.append(
            SweepCell(
                "rho",
                "rho",
                rho,
                ("MEAN_COV_SHRINK_MV",),
                base_params(cfg, rho=rho),
                p.lookback,
                math.isclose(rho, p.rho),
            )
        )
    for g in e.gamma:
        cells.append(
            SweepCell(
                "gamma",
                "gamma",
                g,
                ("SAMPLE_MV", "MEAN_COV_SHRINK_MV"),
                base_params(cfg, gamma=g),
                p.lookback,
                math.isclose(g, p.gamma),
            )
        )
    return cells


def run_sensitivity(
    cfg: Config,
    panel: ReturnPanel,
    primary: BacktestResult,
    optimizer: Optimizer,
    estimates: EstimateCache,
    alloc: AllocationCache,
    log: Log = _print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = split_windows(cfg)
    prim_metrics = compute_metrics(primary.daily, primary.events, windows)
    keep = [
        "date",
        "strategy",
        "scenario",
        "nav_post",
        "nav_before_fees",
        "net_return",
        "gross_return",
        "rf",
        "fee",
        "turnover_half",
    ]
    rows, dailies = [], []
    for cell in sweep_cells(cfg):
        if cell.is_primary:
            m = prim_metrics[prim_metrics["strategy"].isin(cell.strategies)].copy()
            d = primary.daily[primary.daily["strategy"].isin(cell.strategies)][keep].copy()
        else:
            t0 = time.perf_counter()
            spec = RunSpec(
                cell.scenario,
                cell.strategies,
                cell.params,
                cell.lookback,
                cfg.portfolio.initial_nav,
                cfg.splits.simulation_start,
                record_detail=False,
            )
            res = run_backtest(panel, spec, optimizer, estimates, alloc)
            m = compute_metrics(res.daily, res.events, windows)
            d = res.daily[keep].copy()
            log(
                f"  sweep {cell.scenario:<16} {len(cell.strategies)} strategies  {time.perf_counter() - t0:6.1f}s  failures={res.failures}"
            )
        m.insert(0, "is_primary", cell.is_primary)
        m.insert(0, "value", cell.value)
        m.insert(0, "parameter", cell.parameter)
        m.insert(0, "sweep", cell.sweep)
        m["scenario"] = cell.scenario
        d["scenario"] = cell.scenario
        d["sweep"] = cell.sweep
        rows.append(m)
        dailies.append(d)
    return pd.concat(rows, ignore_index=True), pd.concat(
        dailies, ignore_index=True
    ).drop_duplicates(["date", "strategy", "scenario", "sweep"])


# --------------------------------------------------------------------------------------
# Paired moving-block bootstrap of Sharpe differences vs EQW
# --------------------------------------------------------------------------------------


def block_indices(n: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    if block < 1 or block > n:
        raise ValueError("block length must lie in [1, n]")
    nb = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, nb * block)
    return idx[:, :n]


def _sharpe_rows(ex: np.ndarray) -> np.ndarray:
    sd = ex.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, np.sqrt(252.0) * ex.mean(axis=-1) / sd, np.nan)


def paired_block_bootstrap(
    daily: pd.DataFrame,
    window: SplitWindow,
    block: int,
    reps: int,
    seed: int,
    confidence: float = 0.95,
    reference: str = "EQW",
    label: str = "prespecified",
) -> pd.DataFrame:
    """Resample aligned daily rows (all strategies and RF together) in moving blocks."""
    d = daily[(daily["date"] >= window.start) & (daily["date"] <= window.end)]
    wide = d.pivot(index="date", columns="strategy", values="net_return").sort_index()
    rf = d.drop_duplicates("date").set_index("date")["rf"].reindex(wide.index).to_numpy()
    if wide.isna().any().any() or reference not in wide:
        raise ValueError(
            "bootstrap requires complete aligned strategy rows including the reference"
        )
    n = len(wide)
    rng = np.random.default_rng(seed)
    idx = block_indices(n, block, reps, rng)
    ex_all = wide.to_numpy() - rf[:, None]
    ref_col = list(wide.columns).index(reference)
    ref_obs = _sharpe_rows(ex_all[:, ref_col])
    ref_boot = _sharpe_rows(ex_all[idx, ref_col])
    a = (1.0 - confidence) / 2.0
    rows = []
    for j, sid in enumerate(wide.columns):
        if sid == reference:
            continue
        obs = float(_sharpe_rows(ex_all[:, j]) - ref_obs)
        diff = _sharpe_rows(ex_all[idx, j]) - ref_boot
        lo, hi = np.nanquantile(diff, [a, 1 - a])
        rows.append(
            {
                "split": window.name,
                "strategy": sid,
                "reference": reference,
                "statistic": "sharpe_difference",
                "observed_difference": obs,
                "boot_mean": float(np.nanmean(diff)),
                "boot_se": float(np.nanstd(diff, ddof=1)),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "ci_level": confidence,
                "interval_excludes_zero": bool(lo > 0 or hi < 0),
                "share_boot_diff_positive": float(np.nanmean(diff > 0)),
                "block_length": block,
                "replications": reps,
                "seed": seed,
                "n_days": n,
                "analysis": label,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Historical block-bootstrap stress view of the latest completed allocation
# --------------------------------------------------------------------------------------


def stress_view(
    cfg: Config, panel: ReturnPanel, primary: BacktestResult
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sc = cfg.experiments.stress
    ev = primary.events[primary.events["status"] == "executed"]
    last_exec = ev["execution_date"].max()
    sel = ev[ev["execution_date"] == last_exec]
    signal = sel["signal_date"].iloc[0]
    w = primary.weights
    post = w[(w["weight_type"] == "exec_posttrade") & (w["date"] == last_exec)]
    hist_df = panel.returns.loc[:signal]
    available = len(hist_df)
    used = min(sc.history, available)
    hist = hist_df.iloc[-used:].to_numpy()
    info = {
        "signal_date": str(pd.Timestamp(signal).date()),
        "execution_date": str(pd.Timestamp(last_exec).date()),
        "history_requested": sc.history,
        "history_used": used,
        "history_limitation": None
        if used == sc.history
        else f"only {used} returns available through the signal date",
        "horizon_days": sc.horizon,
        "block_length": sc.block_length,
        "paths": sc.paths,
        "seed": sc.seed,
        "assumptions": "Asset returns resampled in moving blocks from history through the signal date only; posttrade weights drift with no further trades and no fees; starting wealth 1. Scenario summaries, not calibrated crash probabilities or prediction intervals.",
    }
    if used < sc.block_length:
        raise ValueError("stress history shorter than one block")
    # Horizon-length paths built from moving blocks of the pre-signal history.
    rng = np.random.default_rng(sc.seed)
    nb = math.ceil(sc.horizon / sc.block_length)
    starts = rng.integers(0, used - sc.block_length + 1, size=(sc.paths, nb))
    pidx = (starts[:, :, None] + np.arange(sc.block_length)).reshape(sc.paths, -1)[:, : sc.horizon]
    asset_growth = np.cumprod(1.0 + hist[pidx], axis=1)  # paths x horizon x assets
    rows, bands = [], []
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    for sid, g in post.groupby("strategy"):
        wv = g.set_index("asset")["value"].reindex(primary.assets).to_numpy()
        wealth = asset_growth @ wv  # paths x horizon
        peak = np.maximum.accumulate(
            np.concatenate([np.ones((sc.paths, 1)), wealth], axis=1), axis=1
        )[:, 1:]
        mdd = (wealth / peak - 1.0).min(axis=1)
        term = wealth[:, -1]
        r = {
            "strategy": sid,
            **{f"terminal_wealth_q{int(q * 100):02d}": float(np.quantile(term, q)) for q in qs},
            **{f"max_drawdown_q{int(q * 100):02d}": float(np.quantile(mdd, q)) for q in qs},
            "share_paths_loss_gt_10pct": float(np.mean(term < 0.9)),
            **{
                k: info[k]
                for k in (
                    "signal_date",
                    "execution_date",
                    "history_used",
                    "horizon_days",
                    "block_length",
                    "paths",
                    "seed",
                )
            },
        }
        rows.append(r)
        qv = np.quantile(wealth, [0.05, 0.5, 0.95], axis=0)
        bands.append(
            pd.DataFrame(
                {
                    "strategy": sid,
                    "day": np.arange(1, sc.horizon + 1),
                    "q05": qv[0],
                    "q50": qv[1],
                    "q95": qv[2],
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(bands, ignore_index=True), info


# --------------------------------------------------------------------------------------
# Ex ante frontier snapshot (estimated, secondary)
# --------------------------------------------------------------------------------------


def frontier_snapshot(
    cfg: Config, primary: BacktestResult, estimates: EstimateCache, optimizer: Optimizer
) -> pd.DataFrame:
    p = cfg.portfolio
    ev = primary.events[primary.events["status"] == "executed"]
    signal = ev["signal_date"].max()
    est = estimates.get(signal, p.lookback)
    rows = []
    gammas = np.geomspace(0.5, 200.0, cfg.experiments.frontier_points)
    for model, cov, mu in (
        ("sample", est.sample, est.mu_sample),
        ("shrunk_mean_lw", est.lw, est.mean("shrunk", p.rho)),
    ):
        for g in gammas:
            r = optimizer.mean_variance(cov.factor, mu, float(g), p.horizon, p.max_weight)
            if r.ok:
                w = r.weights
                rows.append(
                    {
                        "signal_date": signal,
                        "moments": model,
                        "kind": "frontier",
                        "gamma": g,
                        "strategy": None,
                        "exante_ann_return": 252 * float(mu @ w),
                        "exante_ann_vol": math.sqrt(252 * float(w @ cov.matrix @ w)),
                    }
                )
        tw = primary.weights[
            (primary.weights["weight_type"] == "target")
            & (primary.weights["signal_date"] == signal)
        ]
        for sid, g in tw.groupby("strategy"):
            w = g.set_index("asset")["value"].reindex(primary.assets).to_numpy()
            rows.append(
                {
                    "signal_date": signal,
                    "moments": model,
                    "kind": "strategy_target",
                    "gamma": None,
                    "strategy": sid,
                    "exante_ann_return": 252 * float(mu @ w),
                    "exante_ann_vol": math.sqrt(252 * float(w @ cov.matrix @ w)),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Reproduce pipeline
# --------------------------------------------------------------------------------------

EXPERIMENTS = (
    "data_validation",
    "design_freeze",
    "primary_backtest",
    "metrics",
    "sensitivity",
    "bootstrap",
    "stress_view",
    "frontier_snapshot",
    "simulation",
    "reports",
)


def write_design_freeze(
    paths: RunPaths, cfg: Config, panel: ReturnPanel, run_id: str, preset: str
) -> dict:
    freeze = {
        "run_id": run_id,
        "frozen_utc": utc_now_iso(),
        "statement": "Written before any backtest, metric, or final-period summary is computed in this run. "
        "The historical final period (2015-2025) was publicly available beforehand, so it is not "
        "claimed to be prospectively unseen.",
        "preset": preset,
        "config": cfg.to_jsonable(),
        "config_fingerprint": cfg.fingerprint(),
        "strategies": [s.to_dict() for s in STRATEGIES.values()],
        "objective": "minimize (gamma/2)*H*w'Sigma w - H*mu'w + kappa*||w - w_signal||_1  s.t. sum(w)=1, 0<=w<=max_weight",
        "code": code_identifier(),
        "data_hashes": panel.data_hashes(),
        "synthetic": panel.synthetic,
    }
    if paths.design_freeze.exists():
        raise FileExistsError(f"{paths.design_freeze} already exists and is immutable")
    blob = json.dumps(freeze, indent=2, default=str).encode("utf-8")
    with paths.design_freeze.open("xb") as fh:  # exclusive create: never overwrite a freeze
        fh.write(blob)
    return freeze


def save_primary(manifest: Manifest, res: BacktestResult, panel: ReturnPanel) -> None:
    label = "SYNTHETIC " if panel.synthetic else ""
    save_table(
        manifest,
        "daily_results",
        res.daily,
        f"{label}Daily pre/post-cost NAV, gross policy and net returns, RF, fees, turnover, concentration, and event flags per strategy.",
    )
    save_table(
        manifest,
        "weights",
        res.weights,
        "Long-format weights: signal_holding (drifted at signal close), target (frozen), exec_pretrade, exec_posttrade.",
    )
    ev = res.events.copy()
    save_table(
        manifest,
        "rebalance_events",
        ev,
        "One row per strategy and signal: dates, status, planned vs realized trade fraction and cost, NAV, solver diagnostics, residuals, forecast vs realized risk.",
        csv=True,
    )
    re_ = res.reallocations.copy()
    re_.insert(
        0,
        "interpretation",
        "hypothetical notional reallocation between academic industry return portfolios; not orders or share quantities",
    )
    save_table(
        manifest,
        "notional_reallocations",
        re_,
        "Per execution and asset: pretrade and posttrade hypothetical dollars and the signed change.",
        csv=True,
        parquet=False,
    )
    save_table(
        manifest,
        "estimation_diagnostics",
        res.estimation,
        "Per signal date: Ledoit-Wolf alpha (covariance shrinkage), fixed rho (mean shrinkage), eigenvalue and condition diagnostics.",
    )
    save_table(
        manifest,
        "estimation_vectors",
        res.estimation_vectors,
        "Per signal date and asset: sample and shrunk daily mean forecasts; eigenvalues (descending) of sample and LW covariance.",
    )
    save_table(
        manifest,
        "covariance_matrices",
        res.covariances,
        "Per signal date: sample and Ledoit-Wolf daily covariance matrices (row asset x asset columns).",
    )


def reproduce(
    cfg: Config,
    preset: str | None = None,
    log: Log = _print,
    run_id: str | None = None,
    with_reports: bool = True,
) -> RunPaths:
    from . import simulation

    preset = preset or cfg.default_preset
    t_start = time.perf_counter()
    run_id = run_id or make_run_id(cfg.run_name, cfg.fingerprint())
    paths = RunPaths(Path(cfg.output_dir) / run_id).ensure()
    manifest = Manifest.create(
        paths,
        run_id=run_id,
        preset=preset,
        synthetic=cfg.is_synthetic,
        config=cfg.to_jsonable(),
        config_fingerprint=cfg.fingerprint(),
        strategies=[s.to_dict() for s in STRATEGIES.values()],
        code=code_identifier(),
        dependencies=dependency_versions(),
        environment=environment_info(),
        seeds={
            "bootstrap": cfg.experiments.bootstrap.seed,
            "stress": cfg.experiments.stress.seed,
            "simulation_base": cfg.experiments.simulation.base_seed,
            "synthetic_data": (cfg.data.synthetic.seed if cfg.data.synthetic else None),
        },
        timings={},
    )
    for e in EXPERIMENTS:
        manifest.data["experiments"][e] = {"status": "pending"}
    manifest.save()
    log(f"run_id: {run_id}")
    log(f"output: {paths.root.resolve()}")

    def stage(name: str, fn: Callable[[], Any], critical: bool = False) -> Any:
        manifest.set_experiment(name, "running")
        t0 = time.perf_counter()
        try:
            out = fn()
        except Exception as exc:
            manifest.set_experiment(
                name,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc()[-4000:],
            )
            log(f"[failed] {name}: {exc}")
            if critical:
                manifest.data["run_status"] = "failed"
                manifest.save()
                raise
            return None
        dt_ = time.perf_counter() - t0
        manifest.data["timings"][name] = round(dt_, 3)
        if manifest.data["experiments"][name]["status"] == "running":
            manifest.set_experiment(name, "completed", wall_time_s=round(dt_, 3))
        log(f"[done] {name:<18} {dt_:7.1f}s")
        return out

    panel: ReturnPanel = stage("data_validation", lambda: _validate(cfg, manifest), critical=True)
    stage(
        "design_freeze",
        lambda: write_design_freeze(paths, cfg, panel, run_id, preset),
        critical=True,
    )

    optimizer = Optimizer(cfg.solver.order, cfg.solver.feasibility_tol)
    estimates = EstimateCache(panel, cfg.solver.psd_repair_rel_tol)
    alloc = AllocationCache()
    windows = split_windows(cfg)

    def _primary() -> BacktestResult:
        res = run_backtest(
            panel,
            primary_spec(cfg),
            optimizer,
            estimates,
            alloc,
            progress=lambda s: log(f"  primary {s}"),
        )
        save_primary(manifest, res, panel)
        if res.failures:
            manifest.data["unresolved_failures"] = res.failures
            manifest.save()
            log(
                f"WARNING: {res.failures} failed allocation events; results are not ready for headline conclusions"
            )
        return res

    primary: BacktestResult = stage("primary_backtest", _primary, critical=True)
    stage(
        "metrics",
        lambda: save_table(
            manifest,
            "metrics",
            compute_metrics(primary.daily, primary.events, windows),
            "Summary statistics per strategy, split, and scenario (primary run; sweeps in sensitivity_results).",
            csv=True,
            parquet=False,
        ),
        critical=True,
    )

    def _sens() -> None:
        sens, sdaily = run_sensitivity(cfg, panel, primary, optimizer, estimates, alloc, log)
        save_table(
            manifest,
            "sensitivity_results",
            sens,
            "One-factor-at-a-time sweeps (cost, lookback, rho, gamma) around the frozen primary configuration, all splits.",
            csv=True,
            parquet=False,
        )
        save_table(
            manifest, "sensitivity_daily", sdaily, "Daily NAV/returns for every sweep scenario."
        )

    stage("sensitivity", _sens)

    def _boot() -> None:
        b = cfg.experiments.bootstrap
        frames = [
            paired_block_bootstrap(
                primary.daily, w, b.block_length, b.replications, b.seed, b.confidence
            )
            for w in windows
        ]
        if preset == "full":
            for L in b.exploratory_block_lengths:
                frames.append(
                    paired_block_bootstrap(
                        primary.daily,
                        windows[[w.name for w in windows].index(cfg.splits.headline)],
                        L,
                        b.replications,
                        b.seed,
                        b.confidence,
                        label="exploratory_block_length",
                    )
                )
        save_table(
            manifest,
            "bootstrap_comparisons",
            pd.concat(frames, ignore_index=True),
            "Paired moving-block bootstrap intervals for Sharpe(strategy) - Sharpe(EQW), conditional on observed returns.",
            csv=True,
            parquet=False,
        )

    stage("bootstrap", _boot)

    def _stress() -> None:
        summ, bands, info = stress_view(cfg, panel, primary)
        save_table(
            manifest,
            "stress_summary",
            summ,
            "Historical block-bootstrap scenario summaries for the latest completed allocation.",
            csv=True,
            parquet=False,
        )
        save_table(
            manifest,
            "stress_bands",
            bands,
            "Per-day 5/50/95% wealth quantiles of the stress scenarios.",
        )
        manifest.set_experiment("stress_view", "running", details=info)

    stage("stress_view", _stress)
    stage(
        "frontier_snapshot",
        lambda: save_table(
            manifest,
            "frontier_snapshot",
            frontier_snapshot(cfg, primary, estimates, optimizer),
            "Ex ante estimated frontier at the latest executed signal (in-sample to its window; not an out-of-sample result).",
            csv=True,
            parquet=False,
        ),
    )

    def _sim() -> None:
        status = simulation.run_simulation(cfg, preset, manifest, optimizer, log)
        manifest.set_experiment("simulation", "running", **status)

    stage("simulation", _sim)
    manifest.data["timings"]["engine_total"] = round(time.perf_counter() - t_start, 3)
    manifest.data["benchmark"] = {
        "workload": f"reproduce preset={preset}: primary backtest (6 strategies) + sweeps + bootstrap + stress + frontier + simulation",
        "environment": environment_info(),
        "wall_time_s_by_stage": dict(manifest.data["timings"]),
        "solver_calls_note": "targets of path-independent strategies are cached across sweep scenarios sharing the same inputs",
    }
    manifest.save()
    if with_reports:
        from . import reporting

        stage("reports", lambda: reporting.generate_all(paths, log=log))
    else:
        manifest.set_experiment("reports", "skipped", reason="with_reports=False")
    failed = [k for k, v in manifest.data["experiments"].items() if v["status"] == "failed"]
    manifest.data["run_status"] = "partial" if failed else "completed"
    manifest.data["completed_utc"] = utc_now_iso()
    manifest.data["timings"]["total"] = round(time.perf_counter() - t_start, 3)
    manifest.save()
    log(f"run_status: {manifest.data['run_status']}  ({manifest.data['timings']['total']:.1f}s)")
    log(f"manifest: {paths.manifest.resolve()}")
    return paths


def _validate(cfg: Config, manifest: Manifest) -> ReturnPanel:
    panel = load_panel(cfg)
    manifest.data["data"] = {
        "provenance": panel.provenance,
        "assets": panel.asset_names,
        "synthetic": panel.synthetic,
        "hashes": panel.data_hashes(),
    }
    manifest.save()
    write_csv(manifest.paths.tables / "data_quality.csv", panel.quality.astype({"value": str}))
    write_csv(manifest.paths.tables / "asset_summary.csv", panel.asset_summary)
    write_csv(manifest.paths.tables / "large_returns.csv", panel.large_returns)
    write_json(manifest.paths.tables / "provenance.json", panel.provenance)
    summary = validation_summary(panel)
    (manifest.paths.tables / "validation_summary.md").write_text(summary, encoding="utf-8")
    return panel


def validation_summary(panel: ReturnPanel) -> str:
    q = panel.quality
    fails = q[q["status"] == "fail"]
    warns = q[q["status"] == "warn"]
    cov = panel.provenance.get("coverage", {})
    lines = [
        "# Data validation summary",
        "",
        f"- Dataset: {panel.provenance.get('dataset')}",
        f"- {'SYNTHETIC DATA - not market data' if panel.synthetic else 'Empirical data'}",
        f"- Coverage: {cov.get('first_date')} to {cov.get('last_date')}, {cov.get('n_dates')} dates x {cov.get('n_assets')} assets",
        f"- Checks: {len(q)} total, {len(fails)} failed, {len(warns)} warnings",
        f"- Large |returns| flagged (diagnostic only): {len(panel.large_returns)}",
    ]
    for _, r in q.iterrows():
        lines.append(f"  - {r['check']}: {r['status']} ({r['value']}) {r['detail']}")
    return "\n".join(lines) + "\n"
