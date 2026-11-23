"""Seeded determinism, immutable design freeze, manifest status, stable simulation seeds (SYNTHETIC)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_engine.artifacts import RunPaths, read_json
from portfolio_engine.experiments import reproduce, write_design_freeze
from portfolio_engine.optimizer import Optimizer
from portfolio_engine.simulation import population, rng_for, run_cell

TABLES = [
    "daily_results",
    "rebalance_events",
    "weights",
    "metrics",
    "sensitivity_results",
    "bootstrap_comparisons",
    "simulation_summary",
]


def _load(paths: RunPaths, name: str) -> pd.DataFrame:
    p = paths.tables / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(paths.tables / f"{name}.csv")


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory, smoke_config):
    out = []
    for k in range(2):
        cfg = smoke_config.model_copy(update={"output_dir": tmp_path_factory.mktemp(f"run{k}")})
        out.append(reproduce(cfg, "core", log=lambda m: None, run_id=f"run{k}", with_reports=False))
    return out


def test_runs_complete_and_reproduce(two_runs):
    a, b = two_runs
    for p in (a, b):
        m = read_json(p.manifest)
        assert m["run_status"] == "completed"
        assert all(v["status"] in ("completed", "skipped") for v in m["experiments"].values())
    for name in TABLES:
        ta, tb = _load(a, name), _load(b, name)
        drop = [c for c in ("solve_time_s",) if c in ta]
        pd.testing.assert_frame_equal(
            ta.drop(columns=drop), tb.drop(columns=drop), check_exact=False, rtol=1e-9, atol=1e-12
        )


def test_design_freeze_written_before_results_and_immutable(two_runs, smoke_config):
    a = two_runs[0]
    freeze = read_json(a.design_freeze)
    m = read_json(a.manifest)
    assert freeze["data_hashes"] == m["data"]["hashes"]
    assert freeze["frozen_utc"] <= m["experiments"]["primary_backtest"]["updated_utc"]
    assert {s["id"] for s in freeze["strategies"]} == {
        "EQW",
        "SAMPLE_MV",
        "COV_SHRINK_MV",
        "MEAN_COV_SHRINK_MV",
        "COST_AWARE_REG_MV",
        "LW_MINVAR",
    }
    from portfolio_engine.data import load_panel

    with pytest.raises(FileExistsError):
        write_design_freeze(a, smoke_config, load_panel(smoke_config), "x", "core")


def test_simulation_seeds_stable_and_extendable(smoke_config):
    g1 = rng_for(1729, 2, 30, 252, 5).standard_normal(3)
    g2 = rng_for(1729, 2, 30, 252, 5).standard_normal(3)
    np.testing.assert_array_equal(g1, g2)
    assert not np.allclose(g1, rng_for(1729, 2, 30, 252, 6).standard_normal(3))
    opt = Optimizer()
    small = run_cell(smoke_config, 3, 6, 63, 2, opt)
    big = run_cell(smoke_config, 3, 6, 63, 3, opt)
    cols = ["rep", "method", "regret", "pop_ann_vol"]
    pd.testing.assert_frame_equal(small[cols], big[big["rep"] < 2][cols].reset_index(drop=True))


def test_case1_oracle_is_equal_weight(smoke_config):
    df = run_cell(smoke_config, 1, 6, 63, 2, Optimizer())
    oracle = df[df["method"] == "ORACLE"]
    w = oracle[[f"w{i:02d}" for i in range(6)]].to_numpy().ravel()
    np.testing.assert_allclose(w, np.full(6, 1 / 6), atol=1e-5)
    eqw = df[df["method"] == "EQW"]["regret"]
    assert np.allclose(eqw, 0.0, atol=1e-10)
    assert (df.loc[df["method"] != "ORACLE", "regret"] >= -1e-9).all()  # oracle is optimal
    mu, sigma = population(3, 30)
    assert mu[0] == pytest.approx(0.00005) and np.sqrt(sigma[-1, -1]) == pytest.approx(0.020)
