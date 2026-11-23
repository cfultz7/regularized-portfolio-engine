"""Plotly figure builders for the dashboard. Inputs are saved run tables only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from portfolio_engine.strategies import STRATEGY_COLORS, STRATEGY_IDS

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13, color="#1d1d1f"),
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation="h", y=-0.15),
    hovermode="x unified",
)


def _fmt(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(**LAYOUT, title=dict(text=title, x=0, font=dict(size=15)), height=height)
    return fig


def wealth_drawdown(
    paths: pd.DataFrame, strategies: list[str], title: str, value_col: str = "wealth"
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.05
    )
    for sid in strategies:
        g = paths[paths["strategy"] == sid]
        c = STRATEGY_COLORS.get(sid, "#000")
        fig.add_trace(
            go.Scatter(
                x=g["date"],
                y=g[value_col],
                name=sid,
                line=dict(
                    color=c, width=2.2 if sid in ("EQW", "SAMPLE_MV", "COST_AWARE_REG_MV") else 1.4
                ),
            ),
            1,
            1,
        )
        fig.add_trace(
            go.Scatter(
                x=g["date"],
                y=g["drawdown"] * 100,
                name=sid,
                showlegend=False,
                line=dict(color=c, width=1),
            ),
            2,
            1,
        )
    fig.update_yaxes(title_text="growth of $1", row=1, col=1)
    fig.update_yaxes(title_text="drawdown %", row=2, col=1)
    return _fmt(fig, title, 560)


def gross_net(gn: pd.DataFrame, strategies: list[str]) -> go.Figure:
    fig = go.Figure()
    for sid in strategies:
        g = gn[gn["strategy"] == sid]
        c = STRATEGY_COLORS.get(sid, "#000")
        fig.add_trace(
            go.Scatter(
                x=g["date"],
                y=g["before_fees"],
                name=f"{sid} before fees",
                line=dict(color=c, dash="dot", width=1.4),
            )
        )
        fig.add_trace(
            go.Scatter(x=g["date"], y=g["net"], name=f"{sid} net", line=dict(color=c, width=2))
        )
    return _fmt(fig, "Same recorded policy path: before ledger fees (dotted) vs net (solid)")


def weight_heatmap(mat: pd.DataFrame, title: str, cap: float) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=mat.T.to_numpy(),
            x=mat.index,
            y=list(mat.columns),
            colorscale="Blues",
            zmin=0,
            zmax=cap,
            colorbar=dict(title="weight"),
        )
    )
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=9))
    return _fmt(fig, title, 620)


def event_bars(detail: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, color, name in (
        ("exec_pretrade", "#BBBBBB", "pretrade (drifted)"),
        ("target", "#30638E", "frozen target"),
        ("exec_posttrade", "#00798C", "posttrade"),
    ):
        if col in detail:
            fig.add_trace(
                go.Bar(x=detail.index, y=detail[col] * 100, name=name, marker_color=color)
            )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="weight %")
    return _fmt(fig, "Weights at the selected rebalance", 380)


def dollar_changes(detail: pd.DataFrame) -> go.Figure:
    v = detail["change_dollars"].fillna(0.0)
    fig = go.Figure(
        go.Bar(x=detail.index, y=v, marker_color=np.where(v >= 0, "#00798C", "#D1495B"))
    )
    fig.update_yaxes(title_text="hypothetical $ change")
    return _fmt(
        fig, "Signed hypothetical notional reallocation (academic portfolios, not orders)", 340
    )


def planned_realized(ev: pd.DataFrame, strategies: list[str]) -> go.Figure:
    fig = go.Figure()
    for sid in strategies:
        g = ev[ev["strategy"] == sid]
        fig.add_trace(
            go.Scatter(
                x=g["planned_half_turnover"],
                y=g["realized_half_turnover"],
                mode="markers",
                name=sid,
                marker=dict(color=STRATEGY_COLORS.get(sid), size=6, opacity=0.6),
                text=g["signal_date"].dt.strftime("%Y-%m-%d"),
                hovertemplate="%{text}<br>planned %{x:.3f}<br>realized %{y:.3f}",
            )
        )
    m = float(
        np.nanmax([ev["planned_half_turnover"].max(), ev["realized_half_turnover"].max(), 0.01])
    )
    fig.add_trace(
        go.Scatter(
            x=[0, m], y=[0, m], mode="lines", line=dict(color="#999", dash="dash"), showlegend=False
        )
    )
    fig.update_xaxes(title_text="planned half-gross turnover (signal close)")
    fig.update_yaxes(title_text="realized (next close)")
    fig.update_layout(hovermode="closest")
    return _fmt(fig, "Planned vs realized turnover")


def corr_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            reversescale=True,
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _fmt(fig, title, 560)


def eigen_plot(sp: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sp["rank"] + 1,
            y=sp["eig_sample_desc"],
            name="sample",
            mode="lines+markers",
            line=dict(color=STRATEGY_COLORS["SAMPLE_MV"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sp["rank"] + 1,
            y=sp["eig_lw_desc"],
            name="Ledoit-Wolf",
            mode="lines+markers",
            line=dict(color=STRATEGY_COLORS["MEAN_COV_SHRINK_MV"]),
        )
    )
    fig.update_yaxes(type="log", title_text="daily variance")
    fig.update_xaxes(title_text="eigenvalue rank")
    fig.update_layout(hovermode="closest")
    return _fmt(fig, "Eigenvalue spectrum (log scale)", 380)


def series_plot(
    df: pd.DataFrame,
    x: str,
    ys: dict[str, str],
    title: str,
    log: bool = False,
    hline: tuple[float, str] | None = None,
) -> go.Figure:
    fig = go.Figure()
    for col, (name, color) in ys.items():
        fig.add_trace(go.Scatter(x=df[x], y=df[col], name=name, line=dict(color=color)))
    if hline:
        fig.add_hline(y=hline[0], line_dash="dash", line_color="#D1495B", annotation_text=hline[1])
    if log:
        fig.update_yaxes(type="log")
    return _fmt(fig, title, 360)


def mean_bars(mf: pd.DataFrame, rho: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=mf["asset"], y=mf["mu_sample"] * 252 * 100, name="sample mean", marker_color="#BBBBBB"
        )
    )
    fig.add_trace(
        go.Bar(
            x=mf["asset"],
            y=mf["mu_shrunk"] * 252 * 100,
            name=f"shrunk (rho={rho})",
            marker_color=STRATEGY_COLORS["MEAN_COV_SHRINK_MV"],
        )
    )
    fig.update_yaxes(title_text="% per year (daily x 252)")
    fig.update_layout(barmode="group")
    return _fmt(fig, "Sample vs shrunk mean forecasts", 380)


def risk_contrib(rc: pd.DataFrame, strategy: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=rc["asset"], y=rc["weight"] * 100, name="weight %", marker_color="#BBBBBB")
    )
    fig.add_trace(
        go.Bar(
            x=rc["asset"],
            y=rc["risk_contribution"] * 100,
            name="variance contribution %",
            marker_color=STRATEGY_COLORS.get(strategy),
        )
    )
    fig.update_layout(barmode="group")
    return _fmt(
        fig, f"{strategy}: weights vs fractional forecast-variance contributions (sum to 100%)", 380
    )


def forecast_realized(fr: pd.DataFrame, strategies: list[str]) -> go.Figure:
    fig = go.Figure()
    for sid in strategies:
        g = fr[fr["strategy"] == sid]
        fig.add_trace(
            go.Scatter(
                x=g["forecast_vol_lw_ann"] * 100,
                y=g["realized_vol_ann"] * 100,
                mode="markers",
                name=sid,
                marker=dict(color=STRATEGY_COLORS.get(sid), size=6, opacity=0.55),
            )
        )
    m = float(np.nanquantile(fr["realized_vol_ann"], 0.99) * 100) if len(fr) else 1.0
    fig.add_trace(
        go.Scatter(
            x=[0, m], y=[0, m], mode="lines", line=dict(color="#999", dash="dash"), showlegend=False
        )
    )
    fig.update_xaxes(title_text="ex ante LW forecast vol, % ann.")
    fig.update_yaxes(title_text="ex post realized vol next interval, % ann.")
    fig.update_layout(hovermode="closest")
    return _fmt(
        fig, "Forecast vs subsequently realized volatility (~21-day intervals; noisy, ex post only)"
    )


def sweep_plot(s: pd.DataFrame, sweep: str, metric: str, label: str) -> go.Figure:
    g = s[s["sweep"] == sweep]
    fig = go.Figure()
    for sid in [x for x in STRATEGY_IDS if x in set(g["strategy"])]:
        gg = g[g["strategy"] == sid].sort_values("value")
        fig.add_trace(
            go.Scatter(
                x=gg["value"],
                y=gg[metric],
                mode="lines+markers",
                name=sid,
                line=dict(color=STRATEGY_COLORS.get(sid)),
            )
        )
        pr = gg[gg["is_primary"]]
        fig.add_trace(
            go.Scatter(
                x=pr["value"],
                y=pr[metric],
                mode="markers",
                showlegend=False,
                marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="black", width=2)),
                hoverinfo="skip",
            )
        )
    fig.update_xaxes(title_text=sweep)
    fig.update_yaxes(title_text=label)
    fig.update_layout(hovermode="closest")
    return _fmt(fig, f"{sweep} sweep (circle = frozen primary value)", 360)


def bootstrap_plot(b: pd.DataFrame) -> go.Figure:
    g = (
        b.set_index("strategy")
        .reindex([s for s in STRATEGY_IDS if s != "EQW"])
        .dropna(subset=["observed_difference"])
    )
    fig = go.Figure(
        go.Scatter(
            x=g["observed_difference"],
            y=g.index,
            mode="markers",
            marker=dict(size=10, color="#333"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=g["ci_high"] - g["observed_difference"],
                arrayminus=g["observed_difference"] - g["ci_low"],
                color="#888",
            ),
        )
    )
    fig.add_vline(x=0, line_color="#D1495B")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Sharpe(strategy) - Sharpe(EQW)")
    fig.update_layout(hovermode="closest")
    return _fmt(fig, "Paired moving-block bootstrap intervals", 340)


def simulation_plot(ss: pd.DataFrame, case: int, metric: str = "mean_regret") -> go.Figure:
    g = ss[(ss["case"] == case) & (ss["method"] != "ORACLE")]
    fig = go.Figure()
    for (m, n), gg in g.groupby(["method", "n_assets"]):
        gg = gg.sort_values("n_obs")
        fig.add_trace(
            go.Scatter(
                x=gg["n_obs"],
                y=gg[metric] * (1e4 if "regret" in metric else 1),
                mode="lines+markers",
                name=f"{m} (N={n})",
                line=dict(color=STRATEGY_COLORS.get(m)),
            )
        )
    fig.update_xaxes(type="log", title_text="estimation length T")
    fig.update_yaxes(title_text="mean regret vs oracle (bp)" if "regret" in metric else metric)
    fig.update_layout(hovermode="closest")
    name = g["case_name"].iloc[0] if len(g) else ""
    return _fmt(fig, f"SYNTHETIC case {case}: {name}", 380)


def stress_plot(bands: pd.DataFrame, strategies: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for sid in strategies:
        g = bands[bands["strategy"] == sid]
        if g.empty:
            continue
        c = STRATEGY_COLORS.get(sid, "#000")
        fig.add_trace(
            go.Scatter(
                x=g["day"], y=g["q95"], line=dict(width=0), showlegend=False, hoverinfo="skip"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=g["day"],
                y=g["q05"],
                fill="tonexty",
                line=dict(width=0),
                fillcolor=_rgba(c, 0.12),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(x=g["day"], y=g["q50"], name=f"{sid} median", line=dict(color=c, width=2))
        )
    fig.update_xaxes(title_text="trading days after execution")
    fig.update_yaxes(title_text="wealth (start = 1)")
    return _fmt(fig, title, 400)


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"
