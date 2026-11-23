"""Data adapters: section selection, sentinels, one-time unit conversion, invalid inputs (SYNTHETIC fixtures)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from portfolio_engine.config import CsvSource, DataConfig
from portfolio_engine.data import (
    DataValidationError,
    load_csv_panel,
    load_french_panel,
    parse_french_sections,
    select_section,
)
from tests.conftest import FIXTURES


def test_parser_finds_both_sections_and_selects_value_weighted():
    text = (FIXTURES / "french_industries_fixture.csv").read_text(encoding="latin-1")
    secs = parse_french_sections(text)
    titles = [s.title for s in secs]
    assert titles == [
        "Average Value Weighted Returns -- Daily",
        "Average Equal Weighted Returns -- Daily",
    ]
    vw = select_section(secs, "Average Value Weighted Returns -- Daily")
    assert vw.header == ["Food", "Beer", "Smoke"]
    assert len(vw.rows) == 5  # sections are not concatenated
    with pytest.raises(DataValidationError):
        select_section(secs, "Average Value Weighted Returns -- Monthly")


def test_french_panel_units_sentinel_and_date_range(french_cache: DataConfig):
    panel = load_french_panel(french_cache)
    r = panel.returns
    # 2026 row excluded; 2024-12-31 dropped explicitly because Smoke holds the -99.99 sentinel.
    assert list(r.index.strftime("%Y-%m-%d")) == ["2024-12-30", "2025-01-02", "2025-01-03"]
    # percent -> decimal exactly once; equal-weighted 50% values never appear.
    assert r.loc["2024-12-30", "Food"] == pytest.approx(0.01)
    assert r.loc["2025-01-02", "Food"] == pytest.approx(-0.0125)
    assert r.to_numpy().max() < 0.5
    assert panel.rf.loc["2025-01-02"] == pytest.approx(0.0001)
    prov = panel.provenance["files"][0]
    assert prov["sentinels_converted"] == 1
    assert prov["selected_section"] == "Average Value Weighted Returns -- Daily"
    q = panel.quality.set_index("check")
    assert q.loc["dates_dropped_for_missing", "value"] == 1
    assert q.loc["excluded_after_end", "value"] == 1


def test_missing_values_error_by_default(french_cache: DataConfig):
    strict = french_cache.model_copy(update={"missing_policy": "error"})
    with pytest.raises(DataValidationError, match="never forward-filled"):
        load_french_panel(strict)


def _write(tmp_path, name, df):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def _csv_cfg(tmp_path, rets, rf, units="decimal"):
    rp = _write(tmp_path, "r.csv", rets)
    fp = _write(tmp_path, "rf.csv", rf)
    src = CsvSource(returns_path=rp, rf_path=fp, units=units, frequency="daily")
    cfg = DataConfig(
        source="csv",
        csv=src,
        start=dt.date(2000, 1, 1),
        end=dt.date(2030, 1, 1),
        expected_assets=None,
    )
    return src, cfg


def _frames(n=30, scale=0.01, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    rets = pd.DataFrame({"date": dates, "X": rng.normal(0, scale, n), "Y": rng.normal(0, scale, n)})
    rf = pd.DataFrame({"date": dates, "rf": 0.0001 * (scale / 0.01)})
    return rets, rf


def test_generic_csv_roundtrip_decimal(tmp_path):
    rets, rf = _frames()
    src, cfg = _csv_cfg(tmp_path, rets, rf)
    panel = load_csv_panel(src, cfg)
    np.testing.assert_allclose(panel.returns["X"].to_numpy(), rets["X"].to_numpy())


def test_generic_csv_percent_converted_once(tmp_path):
    rets, rf = _frames(scale=1.0)
    src, cfg = _csv_cfg(tmp_path, rets, rf, units="percent")
    panel = load_csv_panel(src, cfg)
    np.testing.assert_allclose(panel.returns["X"].to_numpy(), rets["X"].to_numpy() / 100.0)


def test_generic_csv_rejects_percent_declared_as_decimal(tmp_path):
    rets, rf = _frames(scale=1.0)
    src, cfg = _csv_cfg(tmp_path, rets, rf, units="decimal")
    with pytest.raises(DataValidationError, match="look like percentages"):
        load_csv_panel(src, cfg)


def test_generic_csv_duplicate_and_unsorted_dates_fail(tmp_path):
    rets, rf = _frames()
    dup = pd.concat([rets, rets.iloc[[3]]]).reset_index(drop=True)
    src, cfg = _csv_cfg(tmp_path, dup, rf)
    with pytest.raises(DataValidationError, match="duplicate dates"):
        load_csv_panel(src, cfg)
    unsorted = rets.iloc[::-1]
    src, cfg = _csv_cfg(tmp_path, unsorted, rf)
    with pytest.raises(DataValidationError, match="not strictly increasing"):
        load_csv_panel(src, cfg)


def test_generic_csv_impossible_return_and_non_numeric(tmp_path):
    rets, rf = _frames()
    bad = rets.copy()
    bad.loc[5, "X"] = -1.2
    src, cfg = _csv_cfg(tmp_path, bad, rf)
    with pytest.raises(DataValidationError):
        load_csv_panel(src, cfg)
    txt = rets.astype({"X": object})
    txt.loc[2, "X"] = "abc"
    src, cfg = _csv_cfg(tmp_path, txt, rf)
    with pytest.raises(DataValidationError, match="non-numeric"):
        load_csv_panel(src, cfg)


def test_misaligned_rf_dates_are_excluded_and_reported(tmp_path):
    rets, rf = _frames()
    src, cfg = _csv_cfg(tmp_path, rets, rf.drop(index=[4]))
    panel = load_csv_panel(src, cfg)
    assert len(panel.returns) == len(rets) - 1
    q = panel.quality.set_index("check")
    assert q.loc["asset_dates_without_rf", "status"] == "warn"


def test_csv_requires_explicit_units():
    with pytest.raises(ValueError):
        CsvSource.model_validate(
            {"returns_path": "r.csv", "rf_path": "f.csv", "frequency": "daily"}
        )


def test_generic_csv_rejects_decimal_declared_as_percent(tmp_path):
    rets, rf = _frames(scale=0.012)  # ordinary decimal daily returns
    src, cfg = _csv_cfg(tmp_path, rets, rf, units="percent")
    with pytest.raises(DataValidationError, match="look like decimals"):
        load_csv_panel(src, cfg)


def test_user_panel_api_returns_feasible_target():
    from portfolio_engine.optimizer import optimize_user_panel
    from tests.conftest import gaussian_panel

    panel = gaussian_panel(n=8)
    w0 = pd.Series(1 / 8, index=panel.assets)
    target, info = optimize_user_panel(
        panel.returns, "2002-06-28", w0, max_weight=0.3, lookback=126
    )
    assert target.sum() == pytest.approx(1.0) and target.max() <= 0.3 + 1e-9 and target.min() >= 0
    assert info["as_of"] == "2002-06-28" and info["n_obs"] == 126
    with pytest.raises(ValueError, match="need 126 observations"):
        optimize_user_panel(panel.returns, "2000-02-01", w0, lookback=126)
