"""Streamlit dashboard over SAVED runs. Launch with ``rpo dashboard --run <run_id>``.

The app never downloads data or re-runs optimizations: every chart is built from the run's saved
tables through ``portfolio_engine.diagnostics`` (the same functions used by the report).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import components as C
from portfolio_engine import diagnostics as D
from portfolio_engine.artifacts import RunPaths, read_json
from portfolio_engine.metrics import SplitWindow, compute_metrics, rebased_paths
from portfolio_engine.strategies import STRATEGIES, STRATEGY_IDS

DEFAULT_SHOWN = ["EQW", "SAMPLE_MV", "COST_AWARE_REG_MV"]
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"

st.set_page_config(page_title="RPO Engine: 1/N vs Markowitz", layout="wide")
st.markdown(
    """<style>
.block-container{padding-top:1.4rem;max-width:1400px}
.small{font-size:0.86rem;color:#4a4a4a}
.synth{background:#FDECEE;border-left:4px solid #B00020;padding:8px 12px;color:#7a0016;font-weight:600}
</style>""",
    unsafe_allow_html=True,
)


def _cli_run() -> str | None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--run", default=None)
    p.add_argument("--output-dir", default="outputs")
    args, _ = p.parse_known_args(sys.argv[1:])
    return args.run or os.environ.get("RPO_RUN")


def _runs(output_dir: Path) -> list[Path]:
    return sorted((d for d in output_dir.glob("*") if (d / "manifest.json").exists()), reverse=True)


#: Trimmed copy of the headline run committed for hosted demos (see scripts/bundle_demo_run.py).
BUNDLED_RUNS = Path(__file__).resolve().parents[1] / "demo_runs"


def _bundled_runs() -> list[Path]:
    if os.environ.get("RPO_INCLUDE_BUNDLED", "1") == "0":
        return []
    return _runs(BUNDLED_RUNS)


@st.cache_resource(show_spinner=False)
def load_run(path: str, mtime: float) -> D.RunData:
    rd = D.RunData(RunPaths(Path(path)))
    _ = (
        rd.daily,
        rd.events,
        rd.weights,
        rd.metrics,
    )  # load once; cached per run directory and manifest version
    return rd


@st.cache_data(show_spinner=False)
def paths_for(
    path: str, mtime: float, split: str, scenario: str, start, end, nav_col: str
) -> pd.DataFrame:
    run = load_run(path, mtime)
    daily = run.daily if scenario == "primary" else _scenario_daily(run, scenario)
    w = SplitWindow(split, pd.Timestamp(start), pd.Timestamp(end))
    return rebased_paths(daily, w, nav_col)


def _scenario_daily(run: D.RunData, scenario: str) -> pd.DataFrame:
    d = run.table("sensitivity_daily")
    d = d[d["scenario"] == scenario].drop(columns=["sweep"]).drop_duplicates(["date", "strategy"])
    d["date"] = pd.to_datetime(d["date"])
    return d


# ------------------------------------------------------------------ run selection
initial = _cli_run()
out_dir = Path(initial).parent if initial and Path(initial).is_dir() else Path("outputs")
runs = _runs(out_dir)
runs += [b for b in _bundled_runs() if b.name not in {r.name for r in runs}]
if (
    initial
    and Path(initial).is_dir()
    and Path(initial).resolve() not in [r.resolve() for r in runs]
):
    runs.insert(0, Path(initial))
if not runs:
    st.error(
        f"No saved runs found under {out_dir.resolve()}. Create one with `rpo demo` or `rpo reproduce --config configs/default.yaml`."
    )
    st.stop()
names = [r.name for r in runs]
default_idx = names.index(Path(initial).name) if initial and Path(initial).name in names else 0

with st.sidebar:
    st.header("Controls")
    run_name = st.selectbox("Saved run", names, index=default_idx)
run_path = runs[names.index(run_name)]
try:
    mt = (run_path / "manifest.json").stat().st_mtime
    run = load_run(str(run_path), mt)
    manifest = run.manifest
except Exception as exc:
    st.error(
        f"Could not load run {run_name}: {type(exc).__name__}: {exc}. The run may be incomplete or from an incompatible schema version."
    )
    st.stop()

cfg = run.config
splits = [w.name for w in run.windows()]
scenarios = ["primary"]
# Sweep scenarios need their daily paths; trimmed demo bundles omit them (sweep metrics stay in tab D).
if run.has("sensitivity_results") and run.has("sensitivity_daily"):
    scenarios += [
        s for s in run.table("sensitivity_results")["scenario"].unique() if s and s not in scenarios
    ]

with st.sidebar:
    split = st.selectbox("Split", splits, index=splits.index(run.headline))
    scenario = st.selectbox(
        "Scenario",
        scenarios,
        index=0,
        help="primary = frozen design. Sweep scenarios rerun policies with one changed setting.",
    )
    shown = st.multiselect("Strategies to display", list(STRATEGY_IDS), default=DEFAULT_SHOWN)
    win = run.window(split)
    dr = st.date_input(
        "Date range (re-anchored at start)",
        value=(win.start.date(), win.end.date()),
        min_value=win.start.date(),
        max_value=win.end.date(),
    )
    view = st.radio("Return view", ["net", "before fees (same policy path)"], index=0)
    st.caption("Changing controls only re-slices saved results; nothing is re-optimized.")
start, end = dr if isinstance(dr, tuple) and len(dr) == 2 else (win.start.date(), win.end.date())
shown = shown or DEFAULT_SHOWN

# ------------------------------------------------------------------ header
st.title("Regularized Portfolio Optimization Engine: 1/N vs. Markowitz")
if run.synthetic:
    st.markdown(
        '<div class="synth">SYNTHETIC DEMONSTRATION RUN: simulated returns, not market evidence.</div>',
        unsafe_allow_html=True,
    )
p = cfg["portfolio"]
st.markdown(
    f"<div class='small'><b>Data:</b> {run.dataset_label()} &nbsp;|&nbsp; <b>Evaluated:</b> {split} {start} to {end} &nbsp;|&nbsp; "
    f"<b>Costs:</b> {run.cost_label()} &nbsp;|&nbsp; <b>Rebalance:</b> month-end signal, next-close execution &nbsp;|&nbsp; "
    f"<b>Run:</b> <code>{run.run_id}</code> ({manifest.get('run_status')}) &nbsp;|&nbsp; <b>Scenario:</b> {scenario}</div>",
    unsafe_allow_html=True,
)
if manifest.get("run_status") != "completed":
    st.warning(
        f"Run status is '{manifest.get('run_status')}'. Some experiments may be missing or failed."
    )
if manifest.get("unresolved_failures"):
    st.error(
        f"{manifest['unresolved_failures']} failed allocation events: not ready for headline conclusions."
    )

tabs = st.tabs(
    [
        "A. Overview",
        "B. Allocations & Rebalancing",
        "C. Estimation & Risk",
        "D. Robustness & Simulation",
        "E. Methods & Data",
    ]
)

# ------------------------------------------------------------------ A. Overview
with tabs[0]:
    nav_col = "nav_post" if view == "net" else "nav_before_fees"
    wp = paths_for(str(run_path), mt, split, scenario, start, end, nav_col)
    st.plotly_chart(
        C.wealth_drawdown(
            wp,
            shown,
            f"Growth of $1 ({'net of simulated costs' if view == 'net' else 'before ledger fees, actual policy path'})",
        ),
        key="wealth",
    )
    if scenario == "primary":
        custom = (pd.Timestamp(start), pd.Timestamp(end)) != (win.start, win.end)
        if custom:
            mt_df = compute_metrics(
                run.daily,
                run.events,
                [SplitWindow(f"{split} (custom range)", pd.Timestamp(start), pd.Timestamp(end))],
            )
        else:
            mt_df = run.metrics[
                (run.metrics["split"] == split) & (run.metrics["scenario"] == "primary")
            ]
    else:
        s = run.table("sensitivity_results")
        mt_df = s[(s["scenario"] == scenario) & (s["split"] == split)]
    cols = [
        "strategy",
        "net_cagr",
        "gross_cagr_before_fees",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "mdd_trough_date",
        "avg_effective_n",
        "recurring_half_turnover_ann",
        "total_fees",
        "target_cap_hits",
        "solver_failures",
        "feasibility_violations",
        "drift_cap_exceedance_days",
    ]
    table = mt_df[[c for c in cols if c in mt_df]].set_index("strategy")
    st.subheader("All strategies (sortable)")
    st.dataframe(
        table.style.format(
            {c: "{:.2%}" for c in ("net_cagr", "gross_cagr_before_fees", "ann_vol", "max_drawdown")}
            | {
                "sharpe": "{:.3f}",
                "avg_effective_n": "{:.1f}",
                "recurring_half_turnover_ann": "{:.2f}",
                "total_fees": "${:,.0f}",
            },
            na_rep="n/a",
        )
    )
    st.download_button(
        "Download metrics CSV",
        table.to_csv().encode("utf-8"),
        file_name=f"metrics_{run.run_id}_{split}_{scenario}.csv",
        mime="text/csv",
    )
    sel = st.selectbox(
        "Metric cards for",
        list(table.index),
        index=list(table.index).index("COST_AWARE_REG_MV")
        if "COST_AWARE_REG_MV" in table.index
        else 0,
    )
    if "EQW" in table.index:
        r, e = table.loc[sel], table.loc["EQW"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Net CAGR",
            f"{r['net_cagr']:.2%}",
            f"{(r['net_cagr'] - e['net_cagr']) * 100:+.2f} pp vs 1/N",
            delta_color="off",
        )
        c2.metric(
            "Sharpe",
            f"{r['sharpe']:.3f}",
            f"{r['sharpe'] - e['sharpe']:+.3f} vs 1/N",
            delta_color="off",
        )
        c3.metric(
            "Max drawdown",
            f"{r['max_drawdown']:.1%}",
            f"{(r['max_drawdown'] - e['max_drawdown']) * 100:+.1f} pp vs 1/N",
            delta_color="off",
        )
        c4.metric(
            "Half-turnover / yr",
            f"{r['recurring_half_turnover_ann']:.2f}",
            f"{r['recurring_half_turnover_ann'] - e['recurring_half_turnover_ann']:+.2f} vs 1/N",
            delta_color="off",
        )
    if scenario == "primary":
        st.plotly_chart(C.gross_net(D.gross_vs_net(run, split), shown), key="gn")
    fj = run.paths.root / "findings.json"
    if fj.exists():
        st.subheader("Interpretation (generated from saved results)")
        for line in read_json(fj)["findings"]:
            st.markdown(f"- {line}")
    st.caption(
        "No strategy is labelled a winner; differences should be read with the bootstrap intervals in tab D."
    )

# ------------------------------------------------------------------ B. Allocations
with tabs[1]:
    if scenario != "primary":
        st.info("Allocation detail is stored for the frozen primary run only; showing primary.")
    strat = st.selectbox(
        "Strategy",
        list(STRATEGY_IDS),
        index=list(STRATEGY_IDS).index("COST_AWARE_REG_MV"),
        key="alloc_s",
    )
    mat = D.allocation_matrix(run, strat)
    mat = mat[(mat.index >= pd.Timestamp(start)) & (mat.index <= pd.Timestamp(end))]
    st.plotly_chart(
        C.weight_heatmap(
            mat, f"{strat}: posttrade weights at each execution", float(p["max_weight"])
        ),
        key="heat",
    )
    ev = run.events[(run.events["strategy"] == strat)]
    ev_sel = ev[
        (ev["signal_date"] >= pd.Timestamp(start)) & (ev["signal_date"] <= pd.Timestamp(end))
    ]
    labels = [
        f"signal {a.date()} -> exec {b.date() if pd.notna(b) else 'pending'} [{s}]"
        for a, b, s in zip(
            ev_sel["signal_date"], ev_sel["execution_date"], ev_sel["status"], strict=True
        )
    ]
    if labels:
        pick = st.selectbox(
            "Rebalance event (signal and execution dates)", labels, index=len(labels) - 1
        )
        row = ev_sel.iloc[labels.index(pick)]
        det = D.event_detail(run, strat, row["signal_date"])
        c1, c2 = st.columns(2)
        c1.plotly_chart(C.event_bars(det), key="evbars")
        c2.plotly_chart(C.dollar_changes(det), key="evdollars")
        k = st.columns(6)
        k[0].metric("Status", str(row["status"]))
        k[1].metric(
            "Planned half-turnover", f"{row.get('planned_half_turnover', float('nan')):.3f}"
        )
        k[2].metric(
            "Realized half-turnover", f"{row.get('realized_half_turnover', float('nan')):.3f}"
        )
        k[3].metric("Realized fee", f"${row.get('fee', 0) or 0:,.0f}")
        k[4].metric("Solver", f"{row['solver']} ({row['solver_status']})")
        k[5].metric(
            "Max residual", f"{row['residual_max']:.1e}" if pd.notna(row["residual_max"]) else "n/a"
        )
        st.caption(
            f"Target cap hits: {row['target_cap_hits']}. Caps bind targets at execution; later drift above the cap is reported separately "
            "(drift_cap_exceedance_days), not as a solver violation."
        )
        st.dataframe(
            det.style.format(
                {
                    c: "{:.2%}"
                    for c in ("signal_holding", "target", "exec_pretrade", "exec_posttrade")
                }
                | {
                    c: "${:,.0f}"
                    for c in ("pretrade_dollars", "posttrade_dollars", "change_dollars")
                },
                na_rep="-",
            )
        )
    st.plotly_chart(
        C.planned_realized(
            run.events[
                (run.events["status"] == "executed")
                & ~run.events["is_initial_deployment"].astype(bool)
            ],
            shown,
        ),
        key="pr",
    )
    show_cols = [
        "signal_date",
        "execution_date",
        "status",
        "planned_half_turnover",
        "realized_half_turnover",
        "planned_cost_fraction",
        "realized_cost_fraction",
        "fee",
        "nav_pre_exec",
        "nav_post_exec",
        "solver",
        "solver_status",
        "residual_max",
        "target_cap_hits",
        "is_initial_deployment",
    ]
    st.dataframe(ev[[c for c in show_cols if c in ev]], hide_index=True)
    st.download_button(
        "Download event table",
        ev.to_csv(index=False).encode("utf-8"),
        file_name=f"events_{strat}_{run.run_id}.csv",
        mime="text/csv",
    )

# ------------------------------------------------------------------ C. Estimation & risk
with tabs[2]:
    sds = D.signal_dates(run)
    sd = st.select_slider("Signal date", options=[d.date() for d in sds], value=sds[-1].date())
    c1, c2 = st.columns(2)
    kind = c1.radio("Matrix", ["Ledoit-Wolf", "sample"], horizontal=True)
    cov = D.covariance_at(run, sd, "lw" if kind == "Ledoit-Wolf" else "sample")
    c1.plotly_chart(
        C.corr_heatmap(D.correlation_from_cov(cov), f"{kind} correlation at {sd}"), key="corr"
    )
    c2.plotly_chart(C.eigen_plot(D.eigen_spectrum(run, sd)), key="eig")
    est = run.table("estimation_diagnostics").copy()
    est["signal_date"] = pd.to_datetime(est["signal_date"])
    row = est[est["signal_date"] == pd.Timestamp(sd)].iloc[0]
    c2.markdown(
        f"Condition number: sample **{row['cond_sample']:.0f}**, LW **{row['cond_lw']:.0f}**. LW alpha **{row['lw_alpha']:.3f}** "
        f"(estimated covariance shrinkage) vs fixed mean-shrinkage rho **{p['rho']}**. Better conditioning does not by itself imply better economic performance."
    )
    c3, c4 = st.columns(2)
    c3.plotly_chart(
        C.series_plot(
            est,
            "signal_date",
            {"lw_alpha": ("LW alpha (estimated)", "#30638E")},
            "Covariance shrinkage alpha through time",
            hline=(p["rho"], "mean shrinkage rho (fixed)"),
        ),
        key="alpha",
    )
    c4.plotly_chart(
        C.series_plot(
            est,
            "signal_date",
            {"cond_sample": ("sample", "#D1495B"), "cond_lw": ("Ledoit-Wolf", "#00798C")},
            "Condition number",
            log=True,
        ),
        key="cond",
    )
    c5, c6 = st.columns(2)
    c5.plotly_chart(C.mean_bars(D.mean_forecasts(run, sd), p["rho"]), key="means")
    rs = c6.selectbox(
        "Risk contributions for",
        [s for s in STRATEGY_IDS],
        index=list(STRATEGY_IDS).index("COST_AWARE_REG_MV"),
    )
    c6.plotly_chart(C.risk_contrib(D.risk_contribution_table(run, rs, sd), rs), key="rc")
    tgt = run.events[(run.events["signal_date"] == pd.Timestamp(sd))][
        [
            "strategy",
            "target_effective_n",
            "target_hhi",
            "target_max_weight",
            "target_cap_hits",
            "forecast_vol_lw_ann",
        ]
    ]
    st.dataframe(tgt.set_index("strategy"), width="stretch")
    st.plotly_chart(C.forecast_realized(D.forecast_vs_realized(run), shown), key="fr")

# ------------------------------------------------------------------ D. Robustness & simulation
with tabs[3]:
    st.markdown(
        "Prespecified one-factor-at-a-time sweeps around the frozen design. Every cell is shown; the primary value is circled. Final-period sweeps are retrospective robustness checks, not new test sets."
    )
    if run.has("sensitivity_results"):
        s = run.table("sensitivity_results")
        s = s[s["split"] == split]
        metric = st.selectbox(
            "Metric",
            ["sharpe", "net_cagr", "ann_vol", "max_drawdown", "recurring_half_turnover_ann"],
            index=0,
        )
        a, b = st.columns(2)
        for k, sw in enumerate(["cost", "lookback", "rho", "gamma"]):
            (a if k % 2 == 0 else b).plotly_chart(
                C.sweep_plot(s, sw, metric, metric), key=f"sw{sw}"
            )
        st.download_button(
            "Download sensitivity_results.csv",
            run.table("sensitivity_results").to_csv(index=False).encode("utf-8"),
            "sensitivity_results.csv",
            "text/csv",
        )
    else:
        st.info(
            f"Sensitivity sweeps not available (status: {run.experiment_status('sensitivity')})."
        )
    if run.has("bootstrap_comparisons"):
        bt = run.table("bootstrap_comparisons")
        bsel = bt[(bt["split"] == split) & (bt["analysis"] == "prespecified")]
        st.plotly_chart(C.bootstrap_plot(bsel), key="boot")
        st.caption(
            "Aligned daily rows of all strategies and RF are resampled together in moving blocks; the engine is not retrained per replication. "
            "An interval including zero means the observed difference is not resolved, not that the strategies are equal. No multiplicity adjustment."
        )
        st.dataframe(bt[bt["split"] == split], hide_index=True)
    else:
        st.info(f"Bootstrap not available (status: {run.experiment_status('bootstrap')}).")
    st.divider()
    st.markdown(
        '<div class="synth">SYNTHETIC Monte Carlo: Gaussian returns with known moments. Separate from the historical evidence.</div>',
        unsafe_allow_html=True,
    )
    if run.has("simulation_summary"):
        ss = run.table("simulation_summary")
        cols3 = st.columns(3)
        for k, case in enumerate(sorted(ss["case"].unique())):
            cols3[k % 3].plotly_chart(C.simulation_plot(ss, int(case)), key=f"sim{case}")
        sim_info = manifest["experiments"].get("simulation", {})
        st.caption(
            f"Preset {sim_info.get('preset')}, {sim_info.get('cells')} cells x {sim_info.get('replications_per_cell')} replications, base seed {sim_info.get('base_seed')}, cap {sim_info.get('cap_rule')}. {sim_info.get('cost_aware_note', '')}"
        )
        st.dataframe(ss, hide_index=True)
    else:
        st.info(
            f"Simulation has not run for this run (status: {run.experiment_status('simulation')}). No values are shown."
        )
    st.divider()
    if run.has("stress_bands"):
        info = manifest["experiments"].get("stress_view", {}).get("details", {})
        st.plotly_chart(
            C.stress_plot(
                run.table("stress_bands"),
                shown,
                f"Historical block-bootstrap SCENARIOS for the allocation executed {info.get('execution_date')}",
            ),
            key="stress",
        )
        st.caption(
            info.get("assumptions", "")
            + f" History used: {info.get('history_used')} days; horizon {info.get('horizon_days')}; blocks {info.get('block_length')}; paths {info.get('paths')}; seed {info.get('seed')}."
        )
        st.dataframe(run.table("stress_summary"), hide_index=True)
    if run.has("frontier_snapshot"):
        with st.expander(
            "Ex ante frontier snapshot (estimated, secondary; never an out-of-sample result)"
        ):
            st.dataframe(run.table("frontier_snapshot"), hide_index=True)

# ------------------------------------------------------------------ E. Methods & data
with tabs[4]:
    st.subheader("Objective")
    st.latex(
        r"\min_w \ \tfrac{\gamma}{2} H\, w^\top \Sigma w - H\,\mu^\top w + \kappa \lVert w - w_{signal}\rVert_1 \quad \text{s.t.}\ \textstyle\sum w = 1,\ 0 \le w \le w_{max}"
    )
    st.markdown(
        f"gamma = {p['gamma']}, H = {p['horizon']}, rho = {p['rho']}, w_max = {p['max_weight']}, kappa = {p['cost_bps'] / 1e4:g} for COST_AWARE_REG_MV and 0 otherwise. "
        "All methods pay realized costs in the ledger; **only COST_AWARE_REG_MV anticipates costs in its objective.**"
    )
    st.dataframe(pd.DataFrame([s.to_dict() for s in STRATEGIES.values()]).set_index("id"))
    st.subheader("Timeline")
    st.markdown(
        "1. Signal at the last trading close of the month using the trailing window through that close.  \n"
        "2. Frozen target executes at the next trading close (simulation convention; close-to-close data show no fills).  \n"
        "3. New weights first earn the following close-to-close return. A final signal without a later date is a pending proposal."
    )
    st.subheader("Metric definitions")
    st.markdown(
        "Sharpe = sqrt(252) mean(net - RF) / std(net - RF, ddof=1) on daily simple returns; volatility = sqrt(252) std(net). "
        "CAGR uses actual elapsed days / 365.25 from the split anchor (last close before the split). Drawdowns reset their peak at the anchor. "
        "Half-gross turnover = 0.5 x sum|traded| / pretrade NAV; costs are charged on the full bought-plus-sold notional. "
        "Square-root-of-time annualization ignores serial dependence."
    )
    st.subheader("Data and provenance")
    prov = manifest["data"]["provenance"]
    st.markdown(
        f"{prov.get('dataset')}. {prov.get('universe_note', '')}  \n[Kenneth French Data Library]({FRENCH})  \n{prov.get('vintage_note', '')}"
    )
    dq = run.paths.tables / "data_quality.csv"
    if dq.exists():
        st.dataframe(pd.read_csv(dq), hide_index=True)
    with st.expander("Run provenance (manifest)"):
        st.code(
            json.dumps(
                {
                    k: manifest.get(k)
                    for k in (
                        "run_id",
                        "run_status",
                        "preset",
                        "code",
                        "dependencies",
                        "environment",
                        "seeds",
                        "timings",
                        "experiments",
                    )
                },
                indent=2,
                default=str,
            ),
            language="json",
        )
    st.subheader("Reports and tables")
    for fname, mime in (("report.html", "text/html"), ("technical_note.pdf", "application/pdf")):
        f = run.paths.root / fname
        if f.exists():
            st.download_button(
                f"Download {fname}", f.read_bytes(), file_name=fname, mime=mime, key=f"dl_{fname}"
            )
    for f in sorted(run.paths.tables.glob("*.csv")):
        st.download_button(
            f"tables/{f.name}",
            f.read_bytes(),
            file_name=f.name,
            mime="text/csv",
            key=f"dl_{f.name}",
        )
    st.caption(
        "Limitations: current-vintage academic industry portfolios (not tradable instruments); stylized linear costs; next-close execution convention; "
        "one prespecified design; the 2015-2025 period was public before the design was frozen."
    )
