"""Read-only diagnostics computed from *saved* run artifacts.

The report generator and the dashboard both call these functions, so displayed numbers come
from one implementation and never require re-running an optimization.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import RunPaths, find_run, load_table, read_json
from .estimates import risk_contributions
from .metrics import SplitWindow, rebased_paths

DateLike = str | dt.date | pd.Timestamp


@dataclass
class RunData:
    """Lazy loader for a saved run directory."""

    paths: RunPaths

    @classmethod
    def open(cls, output_dir: str | Path, run: str | None) -> RunData:
        return cls(find_run(Path(output_dir), run))

    @cached_property
    def manifest(self) -> dict:
        return read_json(self.paths.manifest)

    @cached_property
    def freeze(self) -> dict | None:
        return read_json(self.paths.design_freeze) if self.paths.design_freeze.exists() else None

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    @property
    def synthetic(self) -> bool:
        return bool(self.manifest.get("synthetic"))

    @property
    def config(self) -> dict:
        return self.manifest["config"]

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._tables:
            self._tables[name] = load_table(self.paths, name)
        return self._tables[name]

    def has(self, name: str) -> bool:
        t = self.paths.tables
        return (t / f"{name}.parquet").exists() or (t / f"{name}.csv").exists()

    def __post_init__(self) -> None:
        self._tables: dict[str, pd.DataFrame] = {}

    def experiment_status(self, name: str) -> str:
        return self.manifest.get("experiments", {}).get(name, {}).get("status", "absent")

    @cached_property
    def daily(self) -> pd.DataFrame:
        d = self.table("daily_results")
        d["date"] = pd.to_datetime(d["date"])
        return d

    @cached_property
    def events(self) -> pd.DataFrame:
        e = self.table("rebalance_events")
        for c in ("signal_date", "execution_date"):
            e[c] = pd.to_datetime(e[c])
        return e

    @cached_property
    def weights(self) -> pd.DataFrame:
        w = self.table("weights")
        w["date"] = pd.to_datetime(w["date"])
        w["signal_date"] = pd.to_datetime(w["signal_date"])
        return w

    @cached_property
    def metrics(self) -> pd.DataFrame:
        return self.table("metrics")

    def windows(self) -> list[SplitWindow]:
        return [
            SplitWindow(s["name"], pd.Timestamp(s["start"]), pd.Timestamp(s["end"]))
            for s in self.config["splits"]["splits"]
        ]

    def window(self, name: str) -> SplitWindow:
        return next(w for w in self.windows() if w.name == name)

    @property
    def headline(self) -> str:
        return self.config["splits"]["headline"]

    @property
    def assets(self) -> list[str]:
        return list(self.manifest["data"]["assets"].keys())

    @property
    def asset_names(self) -> dict[str, str]:
        return self.manifest["data"]["assets"]

    def dataset_label(self) -> str:
        if self.synthetic:
            return "SYNTHETIC demonstration panel (not market data)"
        return "Ken French 30 industry portfolios (value-weighted, daily)"

    def cost_label(self) -> str:
        return f"{self.config['portfolio']['cost_bps']:g} bps per $ traded"


def wealth_paths(run: RunData, split: str, nav_col: str = "nav_post") -> pd.DataFrame:
    return rebased_paths(run.daily, run.window(split), nav_col)


def gross_vs_net(run: RunData, split: str) -> pd.DataFrame:
    w = run.window(split)
    net = rebased_paths(run.daily, w, "nav_post").rename(columns={"wealth": "net"})
    gross = rebased_paths(run.daily, w, "nav_before_fees").rename(columns={"wealth": "before_fees"})
    return net.merge(gross[["date", "strategy", "before_fees"]], on=["date", "strategy"])


def allocation_matrix(
    run: RunData, strategy: str, weight_type: str = "exec_posttrade"
) -> pd.DataFrame:
    w = run.weights
    g = w[(w["strategy"] == strategy) & (w["weight_type"] == weight_type)]
    return g.pivot(index="date", columns="asset", values="value").reindex(columns=run.assets)


def event_detail(run: RunData, strategy: str, signal_date: DateLike) -> pd.DataFrame:
    sd = pd.Timestamp(signal_date)
    w = run.weights
    g = w[(w["strategy"] == strategy) & (w["signal_date"] == sd)]
    wide = g.pivot_table(
        index="asset", columns="weight_type", values="value", aggfunc="first"
    ).reindex(run.assets)
    re_ = run.table("notional_reallocations")
    re_ = re_[(re_["strategy"] == strategy) & (pd.to_datetime(re_["signal_date"]) == sd)].set_index(
        "asset"
    )
    for c in ("pretrade_dollars", "posttrade_dollars", "change_dollars"):
        wide[c] = re_[c].reindex(wide.index) if len(re_) else np.nan
    wide.insert(0, "industry", [run.asset_names.get(a, a) for a in wide.index])
    return wide


def covariance_at(run: RunData, signal_date: DateLike, kind: str = "lw") -> pd.DataFrame:
    c = run.table("covariance_matrices")
    c = c[(pd.to_datetime(c["signal_date"]) == pd.Timestamp(signal_date)) & (c["kind"] == kind)]
    return c.set_index("asset")[run.assets].reindex(run.assets)


def correlation_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    sd = np.sqrt(np.diag(cov.to_numpy()))
    return pd.DataFrame(cov.to_numpy() / np.outer(sd, sd), index=cov.index, columns=cov.columns)


def risk_contribution_table(
    run: RunData, strategy: str, signal_date: DateLike, kind: str = "lw"
) -> pd.DataFrame:
    cov = covariance_at(run, signal_date, kind)
    w = run.weights
    tw = w[
        (w["strategy"] == strategy)
        & (w["signal_date"] == pd.Timestamp(signal_date))
        & (w["weight_type"] == "target")
    ]
    wv = tw.set_index("asset")["value"].reindex(run.assets).to_numpy()
    rc = risk_contributions(wv, cov.to_numpy())
    return pd.DataFrame({"asset": run.assets, "weight": wv, "risk_contribution": rc})


def eigen_spectrum(run: RunData, signal_date: DateLike) -> pd.DataFrame:
    v = run.table("estimation_vectors")
    v = v[pd.to_datetime(v["signal_date"]) == pd.Timestamp(signal_date)]
    return v[["rank", "eig_sample_desc", "eig_lw_desc"]].sort_values("rank")


def mean_forecasts(run: RunData, signal_date: DateLike) -> pd.DataFrame:
    v = run.table("estimation_vectors")
    v = v[pd.to_datetime(v["signal_date"]) == pd.Timestamp(signal_date)]
    return v[["asset", "mu_sample", "mu_shrunk"]]


def forecast_vs_realized(run: RunData) -> pd.DataFrame:
    e = run.events
    cols = [
        "strategy",
        "signal_date",
        "execution_date",
        "forecast_vol_lw_ann",
        "forecast_vol_sample_ann",
        "realized_vol_ann",
        "holding_days",
    ]
    return e[e["realized_vol_ann"].notna()][cols]


def signal_dates(run: RunData) -> list[pd.Timestamp]:
    e = run.events
    return sorted(e.loc[e["status"] == "executed", "signal_date"].unique())
