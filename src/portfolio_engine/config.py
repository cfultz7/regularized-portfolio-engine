"""Validated configuration objects.

All paths in a configuration file are interpreted relative to the current working
directory (run commands from the repository root). Paths are handled with
``pathlib`` so the same YAML works on macOS, Linux, and Windows.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FrenchSource(_Strict):
    industries_url: str = f"{FRENCH_BASE}/ftp/30_Industry_Portfolios_daily_CSV.zip"
    factors_url: str = f"{FRENCH_BASE}/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    definitions_url: str = f"{FRENCH_BASE}/ftp/Siccodes30.zip"
    landing_page: str = f"{FRENCH_BASE}/data_library.html"
    description_page: str = f"{FRENCH_BASE}/data_library/det_30_ind_port.html"
    section: str = "Average Value Weighted Returns -- Daily"
    rf_column: str = "RF"


class CsvSource(_Strict):
    """Generic user panel: ``date`` plus one decimal simple-return column per asset."""

    returns_path: Path
    rf_path: Path
    rf_column: str = "rf"
    units: Literal["decimal", "percent"]
    frequency: Literal["daily"]
    date_format: str | None = None


class SyntheticSource(_Strict):
    n_assets: int = 8
    start: dt.date = dt.date(2000, 1, 3)
    end: dt.date = dt.date(2007, 12, 31)
    seed: int = 20240101


class DataConfig(_Strict):
    source: Literal["french", "csv", "synthetic"] = "french"
    cache_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    start: dt.date = dt.date(1985, 1, 1)
    end: dt.date = dt.date(2025, 12, 31)
    expected_assets: int | None = 30
    missing_policy: Literal["error", "drop_dates"] = "error"
    large_return_threshold: float = 0.25
    french: FrenchSource = FrenchSource()
    csv: CsvSource | None = None
    synthetic: SyntheticSource | None = None

    @model_validator(mode="after")
    def _check_source(self) -> DataConfig:
        if self.source == "csv" and self.csv is None:
            raise ValueError("data.source='csv' requires a data.csv block with explicit units")
        if self.end < self.start:
            raise ValueError("data.end must not precede data.start")
        return self


class PortfolioConfig(_Strict):
    initial_nav: float = Field(1_000_000.0, gt=0)
    lookback: int = Field(252, ge=2)
    gamma: float = Field(5.0, gt=0)
    horizon: int = Field(21, ge=1)
    rho: float = Field(0.75, ge=0, le=1)
    max_weight: float = Field(0.10, gt=0, le=1)
    cost_bps: float = Field(10.0, ge=0)
    rebalance: Literal["month_end"] = "month_end"
    execution_lag: int = Field(1, ge=1, le=1)

    @property
    def cost_rate(self) -> float:
        return self.cost_bps / 10_000.0


class SolverConfig(_Strict):
    order: tuple[str, ...] = ("CLARABEL", "OSQP", "SCS")
    feasibility_tol: float = 1e-6
    psd_repair_rel_tol: float = 1e-10


class Split(_Strict):
    name: str
    start: dt.date
    end: dt.date


class SplitConfig(_Strict):
    simulation_start: dt.date = dt.date(1990, 1, 1)
    splits: tuple[Split, ...] = (
        Split(name="development", start=dt.date(1990, 1, 1), end=dt.date(2004, 12, 31)),
        Split(name="validation", start=dt.date(2005, 1, 1), end=dt.date(2014, 12, 31)),
        Split(name="final", start=dt.date(2015, 1, 1), end=dt.date(2025, 12, 31)),
    )
    headline: str = "final"

    @model_validator(mode="after")
    def _check(self) -> SplitConfig:
        names = [s.name for s in self.splits]
        if len(set(names)) != len(names):
            raise ValueError("split names must be unique")
        if self.headline not in names:
            raise ValueError("headline split must be one of the splits")
        for s in self.splits:
            if s.end < s.start:
                raise ValueError(f"split {s.name} ends before it starts")
        return self

    def get(self, name: str) -> Split:
        for s in self.splits:
            if s.name == name:
                return s
        raise KeyError(name)


class BootstrapConfig(_Strict):
    block_length: int = 20
    replications: int = 1000
    seed: int = 20090301
    confidence: float = 0.95
    exploratory_block_lengths: tuple[int, ...] = (10, 40)


class StressConfig(_Strict):
    history: int = 1260
    horizon: int = 63
    block_length: int = 10
    paths: int = 1000
    seed: int = 6302025


class SimulationPreset(_Strict):
    n_assets: tuple[int, ...] = (30,)
    lengths: tuple[int, ...] = (126, 252, 504)
    replications: int = 100


class SimulationConfig(_Strict):
    base_seed: int = 1729
    cap_multiple: float = 3.0
    gamma: float = 5.0
    horizon: int = 21
    rho: float = 0.75
    core: SimulationPreset = SimulationPreset()
    full: SimulationPreset = SimulationPreset(
        n_assets=(10, 30, 60), lengths=(126, 252, 504), replications=200
    )


class ExperimentConfig(_Strict):
    cost_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 25.0)
    lookback: tuple[int, ...] = (126, 252, 504)
    rho: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0)
    gamma: tuple[float, ...] = (1.0, 5.0, 10.0)
    bootstrap: BootstrapConfig = BootstrapConfig()
    stress: StressConfig = StressConfig()
    simulation: SimulationConfig = SimulationConfig()
    frontier_points: int = 15


class Config(_Strict):
    run_name: str = "default"
    synthetic_label: bool = False
    data: DataConfig = DataConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    solver: SolverConfig = SolverConfig()
    splits: SplitConfig = SplitConfig()
    experiments: ExperimentConfig = ExperimentConfig()
    output_dir: Path = Path("outputs")
    default_preset: Literal["core", "full"] = "core"

    @field_validator("run_name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        if not v or any(ch in v for ch in '\\/:*?"<>| '):
            raise ValueError("run_name must be non-empty and contain no path or shell characters")
        return v

    @property
    def is_synthetic(self) -> bool:
        return self.data.source == "synthetic" or self.synthetic_label

    def to_jsonable(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_jsonable(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def with_portfolio(self, **changes: Any) -> Config:
        return self.model_copy(update={"portfolio": self.portfolio.model_copy(update=changes)})


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"configuration file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {p}")
    return Config.model_validate(raw)
