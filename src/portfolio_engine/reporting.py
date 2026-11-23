"""Figures, HTML report, PDF technical note, and presentation assets from SAVED artifacts only.

Nothing here re-runs an optimization; every number is read from the run's tables via
:mod:`portfolio_engine.diagnostics`. Synthetic runs are labeled on every output.
"""

from __future__ import annotations

import base64
import html
import json
import math
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import Manifest, RunPaths, atomic_write_text, write_json
from .diagnostics import (
    RunData,
    eigen_spectrum,
    forecast_vs_realized,
    mean_forecasts,
    risk_contribution_table,
    signal_dates,
    wealth_paths,
)
from .strategies import STRATEGIES, STRATEGY_COLORS, STRATEGY_IDS

Log = Callable[[str], None]
LABELS = {s: s for s in STRATEGY_IDS}
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


# --------------------------------------------------------------------------------------
# Facts and findings (all derived from saved tables)
# --------------------------------------------------------------------------------------


def _pct(x: float, nd: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}%"


def _num(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def run_tag(run: RunData) -> str:
    cfg = run.config
    w = run.window(run.headline)
    kind = "SYNTHETIC DATA" if run.synthetic else "Historical (academic industry portfolios)"
    return (
        f"{kind} | {run.dataset_label()} | {w.name} {w.start.date()} to {w.end.date()} | "
        f"cost {cfg['portfolio']['cost_bps']:g} bps/$ | monthly signal, next-close execution | run {run.run_id}"
    )


def facts(run: RunData) -> dict:
    """Numbers quoted in the report, technical note, and share images."""
    m = run.metrics
    h = run.headline
    fm = m[(m["split"] == h) & (m["scenario"] == "primary")].set_index("strategy")
    f: dict = {
        "run_id": run.run_id,
        "synthetic": run.synthetic,
        "headline_split": h,
        "split_dates": [str(run.window(h).start.date()), str(run.window(h).end.date())],
    }
    f["metrics"] = {
        sid: {
            k: (None if pd.isna(fm.loc[sid, k]) else fm.loc[sid, k])
            for k in (
                "net_cagr",
                "gross_cagr_before_fees",
                "ann_vol",
                "sharpe",
                "max_drawdown",
                "avg_effective_n",
                "recurring_half_turnover_ann",
                "fees_pct_of_anchor_nav",
                "avg_target_cap_hits",
                "solver_failures",
            )
        }
        for sid in fm.index
    }
    if run.has("bootstrap_comparisons"):
        b = run.table("bootstrap_comparisons")
        b = b[(b["split"] == h) & (b["analysis"] == "prespecified")]
        f["bootstrap"] = {
            r["strategy"]: {
                "diff": r["observed_difference"],
                "lo": r["ci_low"],
                "hi": r["ci_high"],
                "excludes_zero": bool(r["interval_excludes_zero"]),
            }
            for _, r in b.iterrows()
        }
        f["bootstrap_settings"] = (
            {
                "block": int(b["block_length"].iloc[0]),
                "reps": int(b["replications"].iloc[0]),
                "level": float(b["ci_level"].iloc[0]),
            }
            if len(b)
            else {}
        )
    est = run.table("estimation_diagnostics")
    f["lw_alpha_mean"] = float(est["lw_alpha"].mean())
    f["lw_alpha_range"] = [float(est["lw_alpha"].min()), float(est["lw_alpha"].max())]
    f["cond_sample_median"] = float(est["cond_sample"].median())
    f["cond_lw_median"] = float(est["cond_lw"].median())
    w = run.weights
    t = w[w["weight_type"] == "target"].pivot_table(
        index=["signal_date", "asset"], columns="strategy", values="value"
    )

    def l1(a: str, b: str) -> float:
        return float((t[a] - t[b]).abs().groupby(level=0).sum().mean())

    f["mean_l1_sample_vs_covshrink"] = l1("SAMPLE_MV", "COV_SHRINK_MV")
    f["mean_l1_sample_vs_meancov"] = l1("SAMPLE_MV", "MEAN_COV_SHRINK_MV")
    f["mean_l1_meancov_vs_costaware"] = l1("MEAN_COV_SHRINK_MV", "COST_AWARE_REG_MV")
    ev = run.events
    f["n_signals"] = int(ev["signal_date"].nunique())
    f["n_solver_calls"] = int((ev["solver"].notna() & (ev["solver"] != "analytic")).sum())
    f["n_failures"] = int((ev["status"] == "failed").sum())
    f["n_pending"] = int((ev["status"] == "pending").sum())
    f["n_executions"] = int((ev["status"] == "executed").sum())
    f["first_signal"] = str(pd.Timestamp(ev["signal_date"].min()).date())
    f["last_signal"] = str(pd.Timestamp(ev["signal_date"].max()).date())
    f["experiments"] = {k: v.get("status") for k, v in run.manifest.get("experiments", {}).items()}
    f["timings"] = run.manifest.get("timings", {})
    if run.has("sensitivity_results"):
        s = run.table("sensitivity_results")
        c = s[(s["sweep"] == "cost") & (s["split"] == h)]
        f["cost_sweep_sharpe"] = {
            sid: dict(zip(g["value"].astype(float), g["sharpe"], strict=True))
            for sid, g in c.groupby("strategy")
        }
        lb = s[(s["sweep"] == "lookback") & (s["split"] == h)]
        f["lookback_sweep_sharpe"] = {
            sid: dict(zip(g["value"].astype(int), g["sharpe"], strict=True))
            for sid, g in lb.groupby("strategy")
        }
    if run.has("simulation_summary"):
        ss = run.table("simulation_summary")
        f["simulation"] = {
            "cells": len(ss.groupby(["case", "n_assets", "n_obs"])),
            "replications": int(ss["replications"].max()),
        }
    return f


def findings(f: dict) -> list[str]:
    """Plain-English, restrained statements generated from the saved numbers."""
    m = f["metrics"]
    out = []
    e, s, ca = m.get("EQW", {}), m.get("SAMPLE_MV", {}), m.get("COST_AWARE_REG_MV", {})
    b = f.get("bootstrap", {})
    n_excl = sum(1 for v in b.values() if v["excludes_zero"])
    lo_hi = ", ".join(
        f"{k} [{v['lo']:+.2f}, {v['hi']:+.2f}]"
        for k, v in b.items()
        if k in ("SAMPLE_MV", "COST_AWARE_REG_MV")
    )
    out.append(
        f"{f['headline_split'].capitalize()} {f['split_dates'][0][:4]}-{f['split_dates'][1][:4]}: net Sharpe EQW {_num(e.get('sharpe'))}, "
        f"SAMPLE_MV {_num(s.get('sharpe'))}, COST_AWARE_REG_MV {_num(ca.get('sharpe'))}. "
        + (
            f"{len(b) - n_excl} of {len(b)} bootstrap intervals for the Sharpe difference vs 1/N include zero ({lo_hi}), so these differences are not resolved."
            if b and n_excl < len(b)
            else (f"{n_excl} of {len(b)} bootstrap intervals exclude zero." if b else "")
        )
    )
    mc = m.get("MEAN_COV_SHRINK_MV", {})
    if mc and ca:
        out.append(
            f"The cost penalty cut recurring half-gross turnover from {_num(mc['recurring_half_turnover_ann'])}x to "
            f"{_num(ca['recurring_half_turnover_ann'])}x per year relative to identical forecasts without it "
            f"(fees {_pct(mc['fees_pct_of_anchor_nav'])} vs {_pct(ca['fees_pct_of_anchor_nav'])} of starting split NAV); "
            f"net Sharpe {_num(mc['sharpe'])} vs {_num(ca['sharpe'])}."
        )
    lbs = f.get("lookback_sweep_sharpe", {})
    if "EQW" in lbs and "SAMPLE_MV" in lbs and len(lbs["SAMPLE_MV"]) > 1:
        ks = sorted(lbs["SAMPLE_MV"])
        parts = ", ".join(f"{k}d {lbs['SAMPLE_MV'][k]:.2f}" for k in ks)
        out.append(
            f"The point advantage depends on the lookback: SAMPLE_MV net Sharpe by window {parts} "
            f"vs EQW {lbs['EQW'][ks[0]]:.2f} (prespecified one-at-a-time sweep, same split)."
        )
    red = 1.0 - f["cond_lw_median"] / f["cond_sample_median"]
    out.append(
        f"Ledoit-Wolf shrinkage was light (mean alpha {f['lw_alpha_mean']:.3f}) yet cut the median condition number by {red:.0%} "
        f"({f['cond_sample_median']:.0f} to {f['cond_lw_median']:.0f}); targets moved little (mean L1 distance "
        f"{f['mean_l1_sample_vs_covshrink']:.3f} vs SAMPLE_MV), while mean shrinkage moved them far more ({f['mean_l1_sample_vs_meancov']:.3f})."
    )
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def _label(ax, run: RunData, text: str | None = None) -> None:
    lab = text or (
        "SYNTHETIC DATA - not market evidence"
        if run.synthetic
        else "Historical backtest - hypothetical"
    )
    ax.text(
        0.995,
        0.01,
        lab,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#B00020" if run.synthetic else "#666666",
    )


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_wealth(
    run: RunData, split: str, out: Path, strategies=STRATEGY_IDS, log_scale: bool = False
) -> Path:
    wp = wealth_paths(run, split)
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )
    for sid in strategies:
        g = wp[wp["strategy"] == sid]
        lw = 2.2 if sid in ("EQW", "SAMPLE_MV", "COST_AWARE_REG_MV") else 1.2
        a1.plot(g["date"], g["wealth"], color=STRATEGY_COLORS[sid], lw=lw, label=sid)
        a2.plot(g["date"], g["drawdown"] * 100, color=STRATEGY_COLORS[sid], lw=1.0)
    if log_scale:
        a1.set_yscale("log")
    w = run.window(split)
    a1.set_title(
        f"Growth of $1 (net of simulated costs), {split} split {w.start.date()} to {w.end.date()}"
    )
    a1.set_ylabel("wealth relative to split anchor")
    a2.set_ylabel("drawdown %")
    a1.legend(ncol=3, fontsize=8, loc="upper left")
    _label(a2, run)
    return _save(fig, out)


def fig_full_history(run: RunData, out: Path) -> Path:
    d = run.daily
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for sid in STRATEGY_IDS:
        g = d[d["strategy"] == sid]
        ax.plot(
            g["date"],
            g["nav_post"] / g["nav_post"].iloc[0],
            color=STRATEGY_COLORS[sid],
            lw=1.2,
            label=sid,
        )
    ax.set_yscale("log")
    shades = ["#F5F5F5", "#FFFFFF", "#EEF4FA"]
    for k, w in enumerate(run.windows()):
        ax.axvspan(w.start, w.end, color=shades[k % 3], zorder=0)
        ax.text(w.start, ax.get_ylim()[1], f" {w.name}", va="top", fontsize=8, color="#555555")
    ax.set_title("Continuous chronological path (no resets at split boundaries), log scale")
    ax.set_ylabel("net wealth / initial NAV")
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    _label(ax, run)
    return _save(fig, out)


def fig_gross_net(run: RunData, split: str, out: Path) -> Path:
    m = run.metrics
    fm = (
        m[(m["split"] == split) & (m["scenario"] == "primary")]
        .set_index("strategy")
        .reindex(STRATEGY_IDS)
    )
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(fm))
    ax.bar(
        x - 0.2,
        fm["gross_cagr_before_fees"] * 100,
        0.4,
        color="#BBBBBB",
        label="before fees (actual policy path)",
    )
    ax.bar(
        x + 0.2,
        fm["net_cagr"] * 100,
        0.4,
        color=[STRATEGY_COLORS[s] for s in fm.index],
        label="net of fees",
    )
    ax.set_xticks(x, fm.index, rotation=15, fontsize=8)
    ax.set_ylabel("CAGR %")
    ax.set_title(f"Fee deduction on each recorded policy path ({split})")
    ax.legend(fontsize=8)
    _label(ax, run)
    return _save(fig, out)


def fig_turnover(run: RunData, split: str, out: Path) -> Path:
    m = run.metrics
    fm = (
        m[(m["split"] == split) & (m["scenario"] == "primary")]
        .set_index("strategy")
        .reindex(STRATEGY_IDS)
    )
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    cols = [STRATEGY_COLORS[s] for s in fm.index]
    a1.barh(fm.index, fm["recurring_half_turnover_ann"], color=cols)
    a1.set_title("Recurring half-gross turnover per year")
    a1.invert_yaxis()
    a2.barh(fm.index, fm["avg_effective_n"], color=cols)
    a2.set_title("Average effective number of holdings 1/sum(w^2)")
    a2.invert_yaxis()
    a2.set_yticklabels([])
    _label(a2, run)
    return _save(fig, out)


def fig_planned_realized(run: RunData, out: Path) -> Path:
    e = run.events
    e = e[(e["status"] == "executed") & ~e["is_initial_deployment"].astype(bool)]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for sid in STRATEGY_IDS:
        g = e[e["strategy"] == sid]
        ax.scatter(
            g["planned_half_turnover"],
            g["realized_half_turnover"],
            s=8,
            alpha=0.5,
            color=STRATEGY_COLORS[sid],
            label=sid,
        )
    lim = (
        float(np.nanmax([e["planned_half_turnover"].max(), e["realized_half_turnover"].max()]))
        * 1.05
    )
    ax.plot([0, lim], [0, lim], color="#999999", lw=1, ls="--")
    ax.set_xlabel("planned half-gross turnover at signal close")
    ax.set_ylabel("realized half-gross turnover at next close")
    ax.set_title("Planned vs realized turnover (execution-day drift)")
    ax.legend(fontsize=7, markerscale=2)
    _label(ax, run)
    return _save(fig, out)


def fig_estimation(run: RunData, out: Path) -> Path:
    est = run.table("estimation_diagnostics")
    est["signal_date"] = pd.to_datetime(est["signal_date"])
    sd = signal_dates(run)[-1]
    sp = eigen_spectrum(run, sd)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].semilogy(
        sp["rank"] + 1,
        sp["eig_sample_desc"],
        "o-",
        ms=3,
        color=STRATEGY_COLORS["SAMPLE_MV"],
        label="sample",
    )
    ax[0].semilogy(
        sp["rank"] + 1,
        sp["eig_lw_desc"],
        "o-",
        ms=3,
        color=STRATEGY_COLORS["MEAN_COV_SHRINK_MV"],
        label="Ledoit-Wolf",
    )
    ax[0].set_title(f"Eigenvalues at signal {pd.Timestamp(sd).date()}")
    ax[0].set_xlabel("rank")
    ax[0].set_ylabel("daily variance (log)")
    ax[0].legend()
    ax[1].semilogy(
        est["signal_date"], est["cond_sample"], color=STRATEGY_COLORS["SAMPLE_MV"], label="sample"
    )
    ax[1].semilogy(
        est["signal_date"],
        est["cond_lw"],
        color=STRATEGY_COLORS["MEAN_COV_SHRINK_MV"],
        label="Ledoit-Wolf",
    )
    ax[1].set_title("Condition number (log)")
    ax[1].legend()
    ax[2].plot(
        est["signal_date"],
        est["lw_alpha"],
        color="#30638E",
        label="LW covariance shrinkage alpha (estimated)",
    )
    ax[2].axhline(
        run.config["portfolio"]["rho"], color="#D1495B", ls="--", label="mean shrinkage rho (fixed)"
    )
    ax[2].set_ylim(0, 1)
    ax[2].set_title("Two different shrinkage coefficients")
    ax[2].legend(fontsize=8)
    for a in ax[1:]:
        a.xaxis.set_major_locator(mdates.YearLocator(10))
    _label(ax[2], run)
    return _save(fig, out)


def fig_means_and_risk(run: RunData, out: Path) -> Path:
    sd = signal_dates(run)[-1]
    mf = mean_forecasts(run, sd)
    rc = risk_contribution_table(run, "COST_AWARE_REG_MV", sd)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(mf))
    ax[0].bar(x - 0.2, mf["mu_sample"] * 252 * 100, 0.4, color="#BBBBBB", label="sample mean")
    ax[0].bar(
        x + 0.2,
        mf["mu_shrunk"] * 252 * 100,
        0.4,
        color=STRATEGY_COLORS["MEAN_COV_SHRINK_MV"],
        label=f"shrunk (rho={run.config['portfolio']['rho']})",
    )
    ax[0].set_xticks(x, mf["asset"], rotation=90, fontsize=7)
    ax[0].set_ylabel("forecast mean, % per year (x252)")
    ax[0].set_title(f"Mean forecasts at signal {pd.Timestamp(sd).date()}")
    ax[0].legend(fontsize=8)
    ax[1].bar(x - 0.2, rc["weight"] * 100, 0.4, color="#BBBBBB", label="target weight %")
    ax[1].bar(
        x + 0.2,
        rc["risk_contribution"] * 100,
        0.4,
        color=STRATEGY_COLORS["COST_AWARE_REG_MV"],
        label="share of forecast variance %",
    )
    ax[1].set_xticks(x, rc["asset"], rotation=90, fontsize=7)
    ax[1].set_title("COST_AWARE_REG_MV: weights vs variance contributions (LW)")
    ax[1].legend(fontsize=8)
    _label(ax[1], run)
    return _save(fig, out)


def fig_allocations(
    run: RunData, out: Path, strategies=("EQW", "SAMPLE_MV", "COST_AWARE_REG_MV")
) -> Path:
    w = run.weights
    fig, ax = plt.subplots(len(strategies), 1, figsize=(11, 2.6 * len(strategies)), sharex=True)
    for a, sid in zip(np.atleast_1d(ax), strategies, strict=True):
        g = w[(w["strategy"] == sid) & (w["weight_type"] == "exec_posttrade")]
        mat = g.pivot(index="date", columns="asset", values="value").reindex(columns=run.assets)
        im = a.imshow(
            mat.T.to_numpy(),
            aspect="auto",
            cmap="Blues",
            vmin=0,
            vmax=run.config["portfolio"]["max_weight"],
            extent=[
                mdates.date2num(mat.index[0]),
                mdates.date2num(mat.index[-1]),
                len(run.assets),
                0,
            ],
            interpolation="nearest",
        )
        a.set_yticks(np.arange(len(run.assets)) + 0.5, run.assets, fontsize=5)
        a.set_title(f"{sid}: posttrade target weights at each execution", fontsize=10)
        a.grid(False)
        a.xaxis_date()
    fig.colorbar(im, ax=ax, shrink=0.6, label="weight")
    _label(np.atleast_1d(ax)[-1], run)
    return _save(fig, out)


def fig_forecast_realized(run: RunData, out: Path) -> Path:
    fr = forecast_vs_realized(run)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for sid in STRATEGY_IDS:
        g = fr[fr["strategy"] == sid]
        ax.scatter(
            g["forecast_vol_lw_ann"] * 100,
            g["realized_vol_ann"] * 100,
            s=7,
            alpha=0.45,
            color=STRATEGY_COLORS[sid],
            label=sid,
        )
    lim = float(np.nanquantile(fr["realized_vol_ann"], 0.99)) * 100
    ax.plot([0, lim], [0, lim], ls="--", color="#999999", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("ex ante LW forecast vol at signal, % ann.")
    ax.set_ylabel("ex post realized vol over next holding interval, % ann.")
    ax.set_title("Forecast vs realized risk (~21-day intervals; noisy)")
    ax.legend(fontsize=7, markerscale=2)
    _label(ax, run)
    return _save(fig, out)


def fig_sensitivity(run: RunData, split: str, out: Path) -> Path | None:
    if not run.has("sensitivity_results"):
        return None
    s = run.table("sensitivity_results")
    s = s[s["split"] == split]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
    for a, sweep, xl in zip(
        ax,
        ("cost", "lookback", "rho", "gamma"),
        ("cost bps per $ traded", "lookback (days)", "mean shrinkage rho", "risk aversion gamma"),
        strict=True,
    ):
        g = s[s["sweep"] == sweep]
        for sid, gg in g.groupby("strategy"):
            gg = gg.sort_values("value")
            a.plot(
                gg["value"], gg["sharpe"], "o-", color=STRATEGY_COLORS[sid], label=sid, ms=4, lw=1.4
            )
            pr = gg[gg["is_primary"]]
            a.scatter(
                pr["value"], pr["sharpe"], s=90, facecolors="none", edgecolors="black", zorder=5
            )
        a.set_title(f"{sweep} sweep ({split})")
        a.set_xlabel(xl)
    ax[0].set_ylabel("net Sharpe")
    ax[0].legend(fontsize=7)
    _label(
        ax[-1],
        run,
        ("SYNTHETIC" if run.synthetic else "Historical")
        + " - full prespecified sweep; circle = primary",
    )
    return _save(fig, out)


def fig_bootstrap(run: RunData, out: Path) -> Path | None:
    if not run.has("bootstrap_comparisons"):
        return None
    b = run.table("bootstrap_comparisons")
    b = b[b["analysis"] == "prespecified"]
    splits = list(dict.fromkeys(b["split"]))
    fig, ax = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 3.6), sharey=True)
    for a, sp in zip(np.atleast_1d(ax), splits, strict=True):
        g = (
            b[b["split"] == sp]
            .set_index("strategy")
            .reindex([s for s in STRATEGY_IDS if s != "EQW"])
        )
        y = np.arange(len(g))
        a.errorbar(
            g["observed_difference"],
            y,
            xerr=[g["observed_difference"] - g["ci_low"], g["ci_high"] - g["observed_difference"]],
            fmt="o",
            color="#333333",
            ecolor="#888888",
            capsize=3,
        )
        a.axvline(0, color="#D1495B", lw=1)
        a.set_yticks(y, g.index, fontsize=10)
        a.tick_params(axis="x", labelsize=10)
        a.set_title(f"{sp}: Sharpe minus EQW Sharpe", fontsize=12)
    a0 = np.atleast_1d(ax)[0]
    a0.invert_yaxis()
    r = b.iloc[0]
    _label(
        np.atleast_1d(ax)[-1],
        run,
        f"paired moving-block bootstrap, block {int(r['block_length'])}, {int(r['replications'])} reps, {r['ci_level']:.0%} percentile",
    )
    return _save(fig, out)


def fig_simulation(run: RunData, out: Path) -> Path | None:
    if not run.has("simulation_summary"):
        return None
    s = run.table("simulation_summary")
    s = s[s["method"] != "ORACLE"]
    ns = sorted(s["n_assets"].unique())
    n = 30 if 30 in ns else ns[0]
    s = s[s["n_assets"] == n]
    cases = sorted(s["case"].unique())
    fig, ax = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4))
    for a, c in zip(np.atleast_1d(ax), cases, strict=True):
        g = s[s["case"] == c]
        for m, gg in g.groupby("method"):
            gg = gg.sort_values("n_obs")
            a.plot(
                gg["n_obs"],
                gg["mean_regret"] * 1e4,
                "o-",
                color=STRATEGY_COLORS.get(m, "#000000"),
                label=m,
                ms=4,
            )
        a.set_title(f"case {c}: {g['case_name'].iloc[0]}", fontsize=10)
        a.set_xlabel("estimation length T (days)")
        a.set_xscale("log", base=2)
        a.set_xticks(sorted(g["n_obs"].unique()), [str(v) for v in sorted(g["n_obs"].unique())])
    np.atleast_1d(ax)[0].set_ylabel("mean regret vs oracle (bp of H-day objective)")
    np.atleast_1d(ax)[0].legend(fontsize=7)
    fig.suptitle(
        f"SYNTHETIC Monte Carlo, N={n}, cap 3/N, zero costs: population objective regret vs oracle",
        fontsize=11,
        color="#B00020",
    )
    return _save(fig, out)


def fig_stress(run: RunData, out: Path) -> Path | None:
    if not run.has("stress_bands"):
        return None
    b = run.table("stress_bands")
    info = run.manifest["experiments"].get("stress_view", {}).get("details", {})
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for sid in ("EQW", "SAMPLE_MV", "COST_AWARE_REG_MV", "LW_MINVAR"):
        g = b[b["strategy"] == sid]
        if len(g) == 0:
            continue
        ax.plot(g["day"], g["q50"], color=STRATEGY_COLORS[sid], lw=1.8, label=f"{sid} median")
        ax.fill_between(g["day"], g["q05"], g["q95"], color=STRATEGY_COLORS[sid], alpha=0.10)
    ax.set_xlabel("trading days after execution")
    ax.set_ylabel("wealth (start = 1)")
    ax.set_title(
        f"Historical block-bootstrap scenarios for the allocation executed {info.get('execution_date', '?')}"
    )
    ax.legend(fontsize=8)
    _label(
        ax,
        run,
        f"scenario summary, not a prediction interval: {info.get('history_used')} pre-signal days, {info.get('block_length')}-day blocks, {info.get('paths')} paths, no further trades",
    )
    return _save(fig, out)


def fig_frontier(run: RunData, out: Path) -> Path | None:
    if not run.has("frontier_snapshot"):
        return None
    f = run.table("frontier_snapshot")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    for a, (mod, g) in zip(ax, f.groupby("moments"), strict=False):
        fr = g[g["kind"] == "frontier"].sort_values("exante_ann_vol")
        a.plot(
            fr["exante_ann_vol"] * 100,
            fr["exante_ann_return"] * 100,
            "-",
            color="#888888",
            label="estimated frontier (cap 10%)",
        )
        for _, r in g[g["kind"] == "strategy_target"].iterrows():
            a.scatter(
                r["exante_ann_vol"] * 100,
                r["exante_ann_return"] * 100,
                color=STRATEGY_COLORS[r["strategy"]],
                s=40,
                label=r["strategy"],
                zorder=4,
            )
        a.set_title(
            f"Ex ante, {mod} moments, signal {pd.Timestamp(g['signal_date'].iloc[0]).date()}",
            fontsize=10,
        )
        a.set_xlabel("forecast vol % ann.")
        a.set_ylabel("forecast return % ann.")
        a.legend(fontsize=6)
    _label(ax[-1], run, "ESTIMATED in-sample frontier - not an out-of-sample result")
    return _save(fig, out)


# --------------------------------------------------------------------------------------
# Presentation images (~1600 x 1000 px)
# --------------------------------------------------------------------------------------


def _pres_footer(fig, run: RunData) -> None:
    parts = run_tag(run).split(" | ")
    col = "#B00020" if run.synthetic else "#555555"
    fig.text(0.01, 0.034, " | ".join(parts[:3]), fontsize=12, color=col)
    fig.text(0.01, 0.010, " | ".join(parts[3:]), fontsize=12, color=col)


def presentation_images(run: RunData, out_dir: Path, fs: list[str]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    big = {
        "font.size": 15,
        "axes.titlesize": 19,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "savefig.bbox": None,
        "savefig.dpi": 100,
    }
    h = run.headline
    with plt.rc_context(big):
        # 1 overview
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(
            3,
            3,
            height_ratios=[2.4, 1, 0.05],
            width_ratios=[2.2, 0.05, 1.2],
            left=0.06,
            right=0.98,
            top=0.9,
            bottom=0.08,
            hspace=0.25,
        )
        a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        wp = wealth_paths(run, h)
        for sid in ("EQW", "SAMPLE_MV", "MEAN_COV_SHRINK_MV", "COST_AWARE_REG_MV", "LW_MINVAR"):
            g = wp[wp["strategy"] == sid]
            a1.plot(
                g["date"],
                g["wealth"],
                color=STRATEGY_COLORS[sid],
                lw=2.6 if sid in ("EQW", "SAMPLE_MV", "COST_AWARE_REG_MV") else 1.4,
                label=sid,
            )
            a2.plot(g["date"], g["drawdown"] * 100, color=STRATEGY_COLORS[sid], lw=1.4)
        a1.set_title("Net growth of $1 after simulated trading costs")
        a1.legend(loc="upper left", fontsize=12)
        a1.set_ylabel("wealth / split anchor")
        a2.set_ylabel("drawdown %")
        tx = fig.add_subplot(gs[0:2, 2])
        tx.axis("off")
        import textwrap

        body = "\n\n".join(
            "\n".join(textwrap.wrap(f"{k + 1}. {s}", 46)) for k, s in enumerate(fs[:3])
        )
        tx.text(0, 1, "What the saved results show", fontsize=18, weight="bold", va="top")
        tx.text(0, 0.92, body, fontsize=12.5, va="top", linespacing=1.35)
        fig.suptitle(
            "Regularized Portfolio Optimization Engine: 1/N vs. Markowitz",
            fontsize=24,
            weight="bold",
            x=0.06,
            ha="left",
        )
        _pres_footer(fig, run)
        p = out_dir / "01_overview.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

        # 2 estimation
        fig, ax = plt.subplots(
            1,
            2,
            figsize=(16, 10),
            gridspec_kw={"left": 0.07, "right": 0.98, "top": 0.86, "bottom": 0.14, "wspace": 0.25},
        )
        sd = signal_dates(run)[-1]
        w = run.weights
        tw = w[(w["weight_type"] == "target") & (w["signal_date"] == sd)]
        sel = ("SAMPLE_MV", "COV_SHRINK_MV", "MEAN_COV_SHRINK_MV", "COST_AWARE_REG_MV")
        piv = (
            tw[tw["strategy"].isin(sel)]
            .pivot(index="asset", columns="strategy", values="value")
            .reindex(run.assets)
        )
        order = piv["SAMPLE_MV"].sort_values(ascending=False).index
        piv = piv.loc[order]
        y = np.arange(len(piv))
        for k, sid in enumerate(sel):
            ax[0].barh(
                y + (k - 1.5) * 0.2, piv[sid] * 100, 0.2, color=STRATEGY_COLORS[sid], label=sid
            )
        ax[0].set_yticks(y, piv.index, fontsize=10)
        ax[0].invert_yaxis()
        ax[0].set_xlabel("target weight %")
        ax[0].set_title(f"Targets at signal {pd.Timestamp(sd).date()}")
        ax[0].legend(fontsize=11, loc="lower right")
        est = run.table("estimation_diagnostics")
        est["signal_date"] = pd.to_datetime(est["signal_date"])
        ax[1].semilogy(
            est["signal_date"],
            est["cond_sample"],
            color=STRATEGY_COLORS["SAMPLE_MV"],
            lw=2,
            label="sample covariance",
        )
        ax[1].semilogy(
            est["signal_date"],
            est["cond_lw"],
            color=STRATEGY_COLORS["MEAN_COV_SHRINK_MV"],
            lw=2,
            label="Ledoit-Wolf covariance",
        )
        ax[1].set_title("Covariance condition number at each signal")
        ax[1].legend()
        f = facts(run)
        fig.suptitle(
            "Estimation: conditioning the covariance vs. moving the weights",
            fontsize=22,
            weight="bold",
            x=0.07,
            ha="left",
        )
        fig.text(
            0.07,
            0.905,
            f"Mean LW alpha {f['lw_alpha_mean']:.3f}; mean L1 target distance vs SAMPLE_MV: covariance shrinkage {f['mean_l1_sample_vs_covshrink']:.3f}, plus mean shrinkage {f['mean_l1_sample_vs_meancov']:.3f}",
            fontsize=14,
        )
        _pres_footer(fig, run)
        p = out_dir / "02_estimation.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

        # 3 frictions
        fig, ax = plt.subplots(
            1,
            2,
            figsize=(16, 10),
            gridspec_kw={"left": 0.07, "right": 0.98, "top": 0.86, "bottom": 0.14, "wspace": 0.3},
        )
        if run.has("sensitivity_results"):
            s = run.table("sensitivity_results")
            s = s[(s["split"] == h) & (s["sweep"] == "cost")]
            for sid, g in s.groupby("strategy"):
                g = g.sort_values("value")
                ax[0].plot(
                    g["value"],
                    g["sharpe"],
                    "o-",
                    color=STRATEGY_COLORS[sid],
                    lw=2.4,
                    ms=8,
                    label=sid,
                )
            ax[0].axvline(run.config["portfolio"]["cost_bps"], color="#999999", ls="--")
            ax[0].set_xlabel("modeled cost, bps per $ traded (policies rerun)")
            ax[0].set_ylabel("net Sharpe")
            ax[0].set_title(f"Cost sensitivity, {h} split")
            ax[0].legend(fontsize=11)
        m = run.metrics
        fm = (
            m[(m["split"] == h) & (m["scenario"] == "primary")]
            .set_index("strategy")
            .reindex(STRATEGY_IDS)
        )
        ax[1].barh(
            fm.index,
            fm["recurring_half_turnover_ann"],
            color=[STRATEGY_COLORS[x] for x in fm.index],
        )
        ax[1].invert_yaxis()
        ax[1].set_xlabel(
            "recurring half-gross turnover per year\n(only COST_AWARE_REG_MV plans for costs)"
        )
        ax[1].set_title(f"Turnover at {run.config['portfolio']['cost_bps']:g} bps, {h} split")
        fig.suptitle(
            "Trading frictions: what the planned-cost penalty buys",
            fontsize=22,
            weight="bold",
            x=0.07,
            ha="left",
        )
        _pres_footer(fig, run)
        p = out_dir / "03_frictions.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

        # 4 synthetic oracle regret
        if run.has("simulation_summary"):
            ss = run.table("simulation_summary")
            ss = ss[ss["method"] != "ORACLE"]
            n = 30 if 30 in set(ss["n_assets"]) else int(ss["n_assets"].iloc[0])
            ss = ss[ss["n_assets"] == n]
            cases = sorted(ss["case"].unique())
            fig, ax = plt.subplots(
                1,
                len(cases),
                figsize=(16, 10),
                gridspec_kw={
                    "left": 0.06,
                    "right": 0.98,
                    "top": 0.78,
                    "bottom": 0.14,
                    "wspace": 0.25,
                },
            )
            for a, c in zip(np.atleast_1d(ax), cases, strict=True):
                g = ss[ss["case"] == c]
                for mth, gg in g.groupby("method"):
                    gg = gg.sort_values("n_obs")
                    a.plot(
                        gg["n_obs"],
                        gg["mean_regret"] * 1e4,
                        "o-",
                        lw=2.4,
                        ms=8,
                        color=STRATEGY_COLORS.get(mth),
                        label=mth,
                    )
                a.set_xscale("log", base=2)
                a.set_xticks(
                    sorted(g["n_obs"].unique()), [str(v) for v in sorted(g["n_obs"].unique())]
                )
                a.set_title(f"Case {c}: {g['case_name'].iloc[0].replace('_', ' ')}", fontsize=15)
                a.set_xlabel("estimation length T")
            np.atleast_1d(ax)[0].set_ylabel("mean regret vs oracle (bp)")
            hs, ls = np.atleast_1d(ax)[0].get_legend_handles_labels()
            fig.legend(hs, ls, loc="upper left", bbox_to_anchor=(0.06, 0.865), ncol=5, fontsize=12)
            fig.suptitle(
                "SYNTHETIC Monte Carlo: estimation error vs a known population optimum",
                fontsize=22,
                weight="bold",
                x=0.06,
                ha="left",
                color="#B00020",
            )
            fig.text(
                0.06,
                0.895,
                f"Gaussian returns with fixed true moments, N={n}, cap 3/N, zero costs, common random numbers. Not historical evidence.",
                fontsize=14,
            )
            fig.text(
                0.01,
                0.012,
                f"SYNTHETIC simulation | run {run.run_id}",
                fontsize=12,
                color="#B00020",
            )
            p = out_dir / "04_synthetic_oracle_regret.png"
            fig.savefig(p)
            plt.close(fig)
            paths.append(p)
    return paths


# --------------------------------------------------------------------------------------
# Tables, HTML report, PDF note, text drafts
# --------------------------------------------------------------------------------------

METRIC_COLS = [
    ("net_cagr", "Net CAGR", "pct"),
    ("gross_cagr_before_fees", "Before-fee CAGR", "pct"),
    ("ann_vol", "Ann. vol", "pct"),
    ("sharpe", "Sharpe", "num"),
    ("max_drawdown", "Max DD", "pct"),
    ("avg_effective_n", "Eff. N", "num1"),
    ("recurring_half_turnover_ann", "Half-turnover/yr", "num"),
    ("fees_pct_of_anchor_nav", "Fees % anchor NAV", "pct"),
    ("avg_target_cap_hits", "Avg cap hits", "num1"),
    ("solver_failures", "Solver failures", "int"),
]


def metrics_display(run: RunData, split: str) -> pd.DataFrame:
    m = run.metrics
    fm = (
        m[(m["split"] == split) & (m["scenario"] == "primary")]
        .set_index("strategy")
        .reindex(STRATEGY_IDS)
    )
    out = pd.DataFrame(index=fm.index)
    for col, lab, kind in METRIC_COLS:
        v = fm[col]
        out[lab] = [
            (
                _pct(x)
                if kind == "pct"
                else _num(x, 1)
                if kind == "num1"
                else ("n/a" if pd.isna(x) else str(int(x)))
                if kind == "int"
                else _num(x)
            )
            for x in v
        ]
    return out


def _img_tag(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'


def write_html(run: RunData, figs: dict[str, Path], fs: list[str], out: Path) -> None:
    def table(df: pd.DataFrame) -> str:
        return df.to_html(classes="t", border=0, escape=True)

    status = facts(run)["experiments"]
    parts = [
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RPO Engine Report</title><style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1d1d1f;background:#fff;max-width:1100px;margin:0 auto;padding:16px 20px;line-height:1.5}}
h1{{font-size:26px}} h2{{border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:32px}}
.tag{{font-size:13px;color:{"#B00020" if run.synthetic else "#555"};}} .t{{border-collapse:collapse;font-size:13px;width:100%}}
.t th,.t td{{padding:4px 8px;border-bottom:1px solid #eee;text-align:right}} .t th:first-child,.t td:first-child{{text-align:left}}
.note{{background:#F6F8FA;padding:10px 14px;border-left:3px solid #30638E;font-size:14px}} code{{background:#f3f3f3;padding:1px 4px}}
</style></head><body>
<h1>Regularized Portfolio Optimization Engine: 1/N vs. Markowitz</h1>
<p class="tag">{html.escape(run_tag(run))}</p>"""
    ]
    if run.synthetic:
        parts.append(
            '<p class="note"><b>SYNTHETIC DEMONSTRATION.</b> Every number on this page comes from simulated returns and is not market evidence.</p>'
        )
    parts.append(
        "<h2>Findings from saved results</h2><ul>"
        + "".join(f"<li>{html.escape(s)}</li>" for s in fs)
        + "</ul>"
    )
    parts.append(
        f"<p class='note'>Assets: {html.escape(run.manifest['data']['provenance'].get('universe_note', ''))} Costs are stylized reallocation costs between research portfolios, not execution-cost estimates. Weights and notional reallocations are hypothetical.</p>"
    )
    for sp in [w.name for w in run.windows()]:
        parts.append(f"<h2>Metrics: {sp} split</h2>" + table(metrics_display(run, sp)))
    order = [
        ("wealth", "Net growth and drawdown (headline split)"),
        ("full", "Continuous path"),
        ("gross_net", "Before-fee vs net"),
        ("turnover", "Turnover and concentration"),
        ("planned_realized", "Planned vs realized turnover"),
        ("allocations", "Allocations through time"),
        ("estimation", "Covariance conditioning and shrinkage"),
        ("means_risk", "Mean forecasts and risk contributions"),
        ("forecast_realized", "Forecast vs realized risk"),
        ("sensitivity", "Prespecified sensitivity"),
        ("bootstrap", "Bootstrap intervals"),
        ("stress", "Historical stress scenarios"),
        ("frontier", "Ex ante frontier snapshot (estimated)"),
        ("simulation", "SYNTHETIC Monte Carlo"),
    ]
    for key, title in order:
        if figs.get(key):
            parts.append(f"<h2>{title}</h2>{_img_tag(figs[key])}")
    parts.append(
        "<h2>Experiment status</h2>"
        + table(pd.DataFrame({"status": status}).rename_axis("experiment"))
    )
    parts.append(
        f"<h2>Reproduce</h2><p><code>rpo reproduce --config configs/{'smoke' if run.synthetic else 'default'}.yaml --preset {run.manifest.get('preset')}</code> &nbsp; (this run: <code>{run.run_id}</code>)</p></body></html>"
    )
    atomic_write_text(out, "\n".join(parts))


def _register_font() -> tuple[str, str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", str(base / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(base / "DejaVuSans-Bold.ttf")))
        return "DejaVu", "DejaVu-Bold"
    except Exception:  # pragma: no cover - font missing
        return "Helvetica", "Helvetica-Bold"


def write_pdf(run: RunData, figs: dict[str, Path], fs: list[str], out: Path) -> int:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font, bold = _register_font()
    st = {
        "h1": ParagraphStyle("h1", fontName=bold, fontSize=16, leading=20, spaceAfter=6),
        "h2": ParagraphStyle(
            "h2",
            fontName=bold,
            fontSize=12,
            leading=15,
            spaceBefore=8,
            spaceAfter=4,
            textColor=colors.HexColor("#30638E"),
        ),
        "b": ParagraphStyle(
            "b", fontName=font, fontSize=9.2, leading=12.2, alignment=TA_LEFT, spaceAfter=4
        ),
        "s": ParagraphStyle(
            "s", fontName=font, fontSize=8, leading=10, textColor=colors.HexColor("#555555")
        ),
        "f": ParagraphStyle(
            "f", fontName="Courier", fontSize=8.5, leading=11, leftIndent=8, spaceAfter=4
        ),
    }
    cfg = run.config
    p = cfg["portfolio"]
    f = facts(run)
    pages = {"n": 0}

    def on_page(canvas, doc):
        pages["n"] = doc.page
        canvas.setFont(font, 7)
        canvas.setFillColor(colors.HexColor("#B00020" if run.synthetic else "#777777"))
        kind = (
            "SYNTHETIC DATA"
            if run.synthetic
            else "Historical backtest of academic industry portfolios (hypothetical)"
        )
        canvas.drawString(0.6 * inch, 0.4 * inch, f"{kind} | run {run.run_id}")
        canvas.drawRightString(7.9 * inch, 0.4 * inch, f"page {doc.page}")

    def img(key: str, width: float = 6.9) -> list:
        path = figs.get(key)
        if not path:
            return []
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            w, h = im.size
        return [Image(str(path), width=width * inch, height=width * inch * h / w)]

    def B(text: str) -> Paragraph:
        return Paragraph(text, st["b"])

    story: list = [
        Paragraph("Regularized Portfolio Optimization Engine: 1/N vs. Markowitz", st["h1"]),
        Paragraph(
            "Technical note generated from saved run artifacts. " + html.escape(run_tag(run)),
            st["s"],
        ),
        Spacer(1, 6),
    ]
    if run.synthetic:
        story.append(Paragraph("<b>SYNTHETIC DEMONSTRATION - not market evidence.</b>", st["b"]))
    story += [
        Paragraph("1. Question and scope", st["h2"]),
        B(
            "How do estimation error, portfolio constraints, and turnover change the practical performance of mean-variance optimization "
            "relative to monthly equal weighting? This is an engineering project with an empirical analysis of one prespecified design; it is a "
            "narrower comparison than DeMiguel, Garlappi and Uppal (2009), not a replication, and it does not claim a new return signal."
        ),
        Paragraph("2. Data", st["h2"]),
        B(
            html.escape(run.manifest["data"]["provenance"].get("dataset", ""))
            + ". "
            + html.escape(run.manifest["data"]["provenance"].get("universe_note", ""))
            + " Daily decimal simple returns; RF aligned by trading date. "
            + html.escape(run.manifest["data"]["provenance"].get("vintage_note", ""))
        ),
        Paragraph(
            "3. Methods (common feasible set: fully invested, long only, cap %.0f%% per asset)"
            % (100 * p["max_weight"]),
            st["h2"],
        ),
    ]
    rows = [["ID", "Mean", "Covariance", "Plans costs", "Objective"]]
    for s in STRATEGIES.values():
        rows.append(
            [
                s.id,
                s.mean_model or "-",
                {"sample": "sample", "lw": "Ledoit-Wolf"}.get(s.cov_model or "", "-"),
                "yes" if s.cost_aware else "no",
                s.objective,
            ]
        )
    t = Table(rows, colWidths=[1.8 * inch, 0.8 * inch, 1.0 * inch, 0.9 * inch, 0.9 * inch])
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), font, 8.5),
                ("FONT", (0, 0), (-1, 0), bold, 8.5),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F8")]),
            ]
        )
    )
    story += [
        t,
        Spacer(1, 4),
        B(
            "All six methods pay the same realized proportional costs in the ledger; only COST_AWARE_REG_MV anticipates them in its objective."
        ),
        Paragraph("4. Estimation, objective, and timing", st["h2"]),
        B(
            f"Window: last {p['lookback']} daily observations through the signal date. Sample covariance S = Xc'Xc/T (divisor T, as in scikit-learn). "
            "Ledoit-Wolf (2004) scaled-identity target: Sigma_LW = (1 - alpha) S + alpha (tr S / N) I, alpha estimated at each fit. "
            f"Mean shrinkage: mu_shrunk = (1 - rho) mu + rho mean(mu) 1 with fixed rho = {p['rho']}; at rho = 1 all forecasts are equal and the problem reduces to minimum variance."
        ),
        Paragraph(
            f"minimize (gamma/2) H w' Sigma w - H mu' w + kappa ||w - w_signal||_1;  gamma={p['gamma']}, H={p['horizon']}, kappa={p['cost_bps'] / 1e4:g} for COST_AWARE_REG_MV else 0",
            st["f"],
        ),
        B(
            "Daily moments are scaled by H to approximate holding-period mean and variance (no serial-correlation model). Because weights are long only and "
            "fully invested, ||w||_1 = 1 is constant; the L1 term acts on <i>changes</i> in weights. w_signal is the drifted weight vector at the signal close."
        ),
        B(
            "Timeline: signal at the last trading close of each month; the frozen target executes at the next trading close (a simulation convention; "
            "close-to-close data do not show fills); the new weights first earn the following close-to-close return. Execution uses the exact self-financing rule "
            "p + c sum|p w - b| = 1 and charges c on every dollar bought or sold once."
        ),
        PageBreak(),
        Paragraph(
            f"5. Main results: {f['headline_split']} split {f['split_dates'][0]} to {f['split_dates'][1]}",
            st["h2"],
        ),
    ]
    md = metrics_display(run, run.headline).reset_index()
    hstyle = ParagraphStyle("th", fontName=bold, fontSize=7, leading=8.5)
    tbl = [[Paragraph(html.escape(str(c)), hstyle) for c in md.columns]] + md.to_numpy().tolist()
    t2 = Table(tbl, repeatRows=1, colWidths=[1.45 * inch] + [0.6 * inch] * (len(md.columns) - 1))
    t2.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), font, 7),
                ("FONT", (0, 0), (-1, 0), bold, 6.5),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F8")]),
            ]
        )
    )
    story += [t2, Spacer(1, 4)] + [B(html.escape(x)) for x in fs] + img("wealth", 5.4)
    story += [Paragraph("Development and validation context", st["h2"])]
    ctx = (
        run.metrics[(run.metrics["scenario"] == "primary")]
        .pivot_table(index="strategy", columns="split", values="sharpe")
        .reindex(STRATEGY_IDS)
    )
    ctx = ctx[[w.name for w in run.windows()]]
    t3 = Table(
        [["net Sharpe", *list(ctx.columns)]]
        + [[i] + [_num(v) for v in r] for i, r in zip(ctx.index, ctx.to_numpy(), strict=True)],
        colWidths=[1.8 * inch] + [1.0 * inch] * len(ctx.columns),
    )
    t3.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), font, 8),
                ("FONT", (0, 0), (-1, 0), bold, 8),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.grey),
            ]
        )
    )
    story += [
        t3,
        PageBreak(),
        Paragraph("6. Gross vs net, turnover, concentration, uncertainty", st["h2"]),
    ]
    story += img("gross_net", 6.2) + img("turnover", 6.4) + img("bootstrap", 6.9)
    story += [
        B(
            "Bootstrap: aligned daily rows of all strategies and RF are resampled together in moving blocks; the engine is not retrained in each replication, "
            "so intervals are conditional on the observed return paths. An interval that includes zero means the observed difference is not resolved; it is not "
            "evidence of equality. No adjustment is made for the multiple strategies and cells compared."
        )
    ]
    story += (
        [PageBreak(), Paragraph("7. Estimation diagnostics and sensitivity", st["h2"])]
        + img("estimation", 6.9)
        + img("sensitivity", 6.9)
    )
    story += [
        B(
            "Sweeps vary one factor at a time around the frozen primary design and report every prespecified cell; final-period sweeps are retrospective robustness checks."
        )
    ]
    story += [PageBreak(), Paragraph("8. Controlled simulation (SYNTHETIC)", st["h2"])] + img(
        "simulation", 6.9
    )
    story += [
        B(
            "Stationary Gaussian returns with known moments (three designed cases, base seed 1729, common random numbers, cap 3/N, zero cost). Regret is the oracle's "
            "population objective minus the method's; results describe estimation error under these assumptions only."
        )
    ]
    story += [
        Paragraph("9. Conclusions and limitations", st["h2"]),
        B(
            "Read the evidence with restraint: better conditioning is not better economic performance, lower turnover does not guarantee a higher Sharpe ratio, "
            "and lower volatility is not automatically superior if return, concentration, or drawdown change. Limitations: one current-vintage dataset of 30 "
            "academic industry portfolios; stylized linear costs; next-close execution convention; square-root-of-time annualization; a single prespecified "
            "parameter set with one-at-a-time sweeps; the 2015-2025 period was publicly available before this design was frozen."
        ),
        B(
            f"Experiments completed: {', '.join(k for k, v in f['experiments'].items() if v == 'completed')}. Solver failures: {f['n_failures']}. "
            f"Pending final proposals (no later execution date): {f['n_pending']}."
        ),
        Paragraph(
            f"Reproduce: rpo reproduce --config configs/{'smoke' if run.synthetic else 'default'}.yaml --preset {run.manifest.get('preset')}   (run {run.run_id})",
            st["f"],
        ),
        Paragraph(
            "References: Markowitz (1952); DeMiguel, Garlappi and Uppal (2009, RFS); Ledoit and Wolf (2004, J. Multivariate Analysis), scaled-identity target; "
            "scikit-learn LedoitWolf documentation; CVX Group portfolio optimization and rebalancing notebooks; Kenneth French Data Library.",
            st["s"],
        ),
    ]
    doc = SimpleDocTemplate(
        str(out) + ".tmp",
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="RPO Engine technical note",
        author="Chantry Fultz",
    )
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    Path(str(out) + ".tmp").replace(out)
    return pages["n"]


def generate_all(paths: RunPaths, log: Log = print) -> dict:
    run = RunData(paths)
    fd = paths.figures
    h = run.headline
    figs: dict[str, Path | None] = {
        "wealth": fig_wealth(run, h, fd / f"wealth_drawdown_{h}.png"),
        "full": fig_full_history(run, fd / "wealth_full_history.png"),
        "gross_net": fig_gross_net(run, h, fd / f"gross_vs_net_{h}.png"),
        "turnover": fig_turnover(run, h, fd / f"turnover_concentration_{h}.png"),
        "planned_realized": fig_planned_realized(run, fd / "planned_vs_realized_turnover.png"),
        "allocations": fig_allocations(run, fd / "allocations_heatmap.png"),
        "estimation": fig_estimation(run, fd / "estimation_conditioning.png"),
        "means_risk": fig_means_and_risk(run, fd / "mean_forecasts_risk_contributions.png"),
        "forecast_realized": fig_forecast_realized(run, fd / "forecast_vs_realized_vol.png"),
        "sensitivity": fig_sensitivity(run, h, fd / f"sensitivity_{h}.png"),
        "bootstrap": fig_bootstrap(run, fd / "bootstrap_sharpe_differences.png"),
        "stress": fig_stress(run, fd / "stress_scenarios.png"),
        "frontier": fig_frontier(run, fd / "exante_frontier_snapshot.png"),
        "simulation": fig_simulation(run, fd / "simulation_regret.png"),
    }
    figs_ok = {k: v for k, v in figs.items() if v is not None}
    f = facts(run)
    fs = findings(f)
    pres = presentation_images(run, paths.presentation, fs)
    write_html(run, figs_ok, fs, paths.root / "report.html")
    n_pages = write_pdf(run, figs_ok, fs, paths.root / "technical_note.pdf")
    write_json(paths.root / "findings.json", {"facts": f, "findings": fs, "pdf_pages": n_pages})
    man = Manifest.load(paths)
    man.data["reports"] = {
        "figures": sorted(p.name for p in figs_ok.values()),
        "presentation": [p.name for p in pres],
        "report_html": "report.html",
        "technical_note_pdf": "technical_note.pdf",
        "pdf_pages": n_pages,
    }
    man.save()
    log(
        f"  reports: {len(figs_ok)} figures, {len(pres)} presentation images, technical_note.pdf ({n_pages} pages)"
    )
    log(f"  report: {(paths.root / 'report.html').resolve()}")
    return {"figures": figs_ok, "presentation": pres, "pdf_pages": n_pages}


def _json_safe(o):  # pragma: no cover - helper for debugging
    return json.loads(json.dumps(o, default=str))


__all__ = ["facts", "findings", "generate_all", "math"]
