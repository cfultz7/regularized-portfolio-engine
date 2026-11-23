"""Shared helpers. All data built here are SYNTHETIC."""

from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_engine.config import Config, DataConfig
from portfolio_engine.data import ReturnPanel, finalize_panel

FIXTURES = Path(__file__).parent / "fixtures"


def make_panel(returns: pd.DataFrame, rf: pd.Series | float = 0.0) -> ReturnPanel:
    if not isinstance(rf, pd.Series):
        rf = pd.Series(float(rf), index=returns.index, name="rf")
    cfg = DataConfig(
        source="synthetic",
        start=returns.index[0].date(),
        end=returns.index[-1].date(),
        expected_assets=None,
    )
    return finalize_panel(
        returns,
        rf,
        {c: c for c in returns.columns},
        {"dataset": "SYNTHETIC test"},
        cfg,
        synthetic=True,
    )


def gaussian_panel(
    n: int = 6,
    start: str = "2000-01-03",
    end: str = "2002-12-31",
    seed: int = 7,
    rf: float = 0.0001,
) -> ReturnPanel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end, name="date")
    x = 0.0003 + rng.standard_normal((len(dates), n)) * np.linspace(0.006, 0.015, n)
    df = pd.DataFrame(x, index=dates, columns=[f"A{i}" for i in range(n)])
    return make_panel(df, rf)


@pytest.fixture
def french_cache(tmp_path: Path) -> DataConfig:
    """Zip the SYNTHETIC French-layout fixtures into a temporary cache directory."""
    cache = tmp_path / "raw"
    cache.mkdir()
    for csv_name, zip_name in (
        ("french_industries_fixture.csv", "30_Industry_Portfolios_daily_CSV.zip"),
        ("french_factors_fixture.csv", "F-F_Research_Data_Factors_daily_CSV.zip"),
    ):
        with zipfile.ZipFile(cache / zip_name, "w") as zf:
            zf.write(FIXTURES / csv_name, arcname=csv_name)
    return DataConfig(
        source="french",
        cache_dir=cache,
        start=dt.date(1985, 1, 1),
        end=dt.date(2025, 12, 31),
        expected_assets=3,
        missing_policy="drop_dates",
    )


@pytest.fixture(scope="session")
def smoke_config() -> Config:
    return Config.model_validate(
        {
            "run_name": "pytest_synthetic",
            "data": {
                "source": "synthetic",
                "start": "2000-01-01",
                "end": "2003-12-31",
                "expected_assets": 6,
                "synthetic": {
                    "n_assets": 6,
                    "start": "2000-01-03",
                    "end": "2003-12-31",
                    "seed": 11,
                },
            },
            "portfolio": {"lookback": 63, "max_weight": 0.3},
            "splits": {
                "simulation_start": "2000-07-01",
                "headline": "final",
                "splits": [
                    {"name": "development", "start": "2000-07-01", "end": "2001-12-31"},
                    {"name": "validation", "start": "2002-01-01", "end": "2002-12-31"},
                    {"name": "final", "start": "2003-01-01", "end": "2003-12-31"},
                ],
            },
            "experiments": {
                "lookback": [42, 63, 126],
                "bootstrap": {"replications": 50},
                "stress": {"history": 252, "paths": 50},
                "frontier_points": 4,
                "simulation": {
                    "core": {"n_assets": [6], "lengths": [63], "replications": 3},
                    "full": {"n_assets": [6], "lengths": [63], "replications": 3},
                },
            },
        }
    )
