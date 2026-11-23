"""Data adapters: Kenneth French daily industry files, generic CSV panels, synthetic fixtures.

Every adapter returns a :class:`ReturnPanel` holding *decimal simple returns* indexed by
trading date, an aligned daily risk-free series, provider codes/names, provenance, and a
data-quality table. Percent-to-decimal conversion happens in exactly one place per adapter.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import atomic_write_bytes, atomic_write_text, utc_now_iso
from .config import Config, CsvSource, DataConfig, SyntheticSource

#: Missing-value sentinels documented by the provider ("-99.99 or -999"), in file units (percent).
FRENCH_SENTINELS = frozenset({"-99.99", "-999", "-999.0", "-999.00"})

#: Code-to-name dictionary for the 30-industry classification, transcribed from the provider's
#: ``Siccodes30.txt`` (industry definitions linked from det_30_ind_port.html).
FRENCH30_NAMES: dict[str, str] = {
    "Food": "Food Products",
    "Beer": "Beer & Liquor",
    "Smoke": "Tobacco Products",
    "Games": "Recreation",
    "Books": "Printing and Publishing",
    "Hshld": "Consumer Goods",
    "Clths": "Apparel",
    "Hlth": "Healthcare, Medical Equipment, Pharmaceutical Products",
    "Chems": "Chemicals",
    "Txtls": "Textiles",
    "Cnstr": "Construction and Construction Materials",
    "Steel": "Steel Works Etc",
    "FabPr": "Fabricated Products and Machinery",
    "ElcEq": "Electrical Equipment",
    "Autos": "Automobiles and Trucks",
    "Carry": "Aircraft, ships, and railroad equipment",
    "Mines": "Precious Metals, Non-Metallic, and Industrial Metal Mining",
    "Coal": "Coal",
    "Oil": "Petroleum and Natural Gas",
    "Util": "Utilities",
    "Telcm": "Communication",
    "Servs": "Personal and Business Services",
    "BusEq": "Business Equipment",
    "Paper": "Business Supplies and Shipping Containers",
    "Trans": "Transportation",
    "Whlsl": "Wholesale",
    "Rtail": "Retail",
    "Meals": "Restaurants, Hotels, Motels",
    "Fin": "Banking, Insurance, Real Estate, Trading",
    "Other": "Everything Else",
}

VINTAGE_NOTE = (
    "Current-vintage historical dataset: the provider reconstructs full return histories at each "
    "update and CRSP revisions can change past values. This is not a point-in-time (vintage) "
    "reconstruction. Beginning with the January 2025 release the provider builds US research "
    "returns from CRSP Flat File Format 2.0 (CIZ) instead of the discontinued legacy (FIZ) files; "
    "the provider highlights the change in how monthly returns are compounded (daily compounding with dividends reinvested on ex-dates). This project uses "
    "daily returns and does not quantify the transition's effect on this daily sample."
)
UNIVERSE_NOTE = (
    "Assets are the 30 Kenneth French value-weighted industry research portfolios (returns include "
    "distributions), not individual stocks or directly tradable ETFs."
)
USER_AGENT = "regularized-portfolio-engine/0.1 (research downloader)"


class DataValidationError(ValueError):
    """Raised when input data fail a hard validation rule."""


@dataclass
class ReturnPanel:
    returns: (
        pd.DataFrame
    )  # index: DatetimeIndex 'date'; columns: asset codes; decimal simple returns
    rf: pd.Series  # decimal simple daily risk-free return aligned to returns.index
    asset_names: dict[str, str]
    provenance: dict[str, Any] = field(default_factory=dict)
    quality: pd.DataFrame = field(default_factory=pd.DataFrame)
    asset_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    large_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    synthetic: bool = False

    @property
    def assets(self) -> list[str]:
        return [str(c) for c in self.returns.columns]

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.returns.index)

    def data_hashes(self) -> dict[str, str]:
        hashes = {f["role"]: f["sha256"] for f in self.provenance.get("files", []) if "sha256" in f}
        hashes["processed_returns"] = frame_sha256(self.returns)
        hashes["processed_rf"] = frame_sha256(self.rf.to_frame("rf"))
        return hashes


def frame_sha256(df: pd.DataFrame) -> str:
    """Hash of values, index, and columns; stable across platforms (little-endian float64)."""
    h = hashlib.sha256()
    h.update(json.dumps([str(c) for c in df.columns]).encode("utf-8"))
    h.update(np.asarray(df.index.astype("int64"), dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(df.to_numpy(dtype="<f8")).tobytes())
    return h.hexdigest()


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------------------
# Retrieval and cache
# --------------------------------------------------------------------------------------


def _filename_from_url(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".provenance.json")


def fetch_file(
    url: str, cache_dir: Path, role: str, force: bool = False, timeout: float = 60.0
) -> dict[str, Any]:
    """Download ``url`` into ``cache_dir`` keeping the original bytes; return provenance."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / _filename_from_url(url)
    side = _sidecar(target)
    if target.exists() and not force:
        return verify_cached_file(target, url, role)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        blob = resp.read()
        status = getattr(resp, "status", None)
    prov = {
        "role": role,
        "url": url,
        "filename": target.name,
        "retrieved_at_utc": utc_now_iso(),
        "retrieval": "downloaded",
        "http_status": status,
        "bytes": len(blob),
        "sha256": sha256_bytes(blob),
    }
    atomic_write_bytes(target, blob)
    atomic_write_text(side, json.dumps(prov, indent=2))
    return prov


def verify_cached_file(target: Path, url: str, role: str) -> dict[str, Any]:
    """Verify a cached (or manually placed) file against its provenance sidecar."""
    blob = target.read_bytes()
    digest = sha256_bytes(blob)
    side = _sidecar(target)
    if side.exists():
        prov = json.loads(side.read_text(encoding="utf-8"))
        if prov.get("sha256") != digest:
            raise DataValidationError(
                f"cached file {target} does not match its recorded SHA-256; re-run fetch with --force"
            )
        prov["role"] = role
        return prov
    mtime = dt.datetime.fromtimestamp(target.stat().st_mtime, tz=dt.UTC)
    prov = {
        "role": role,
        "url": url,
        "filename": target.name,
        "retrieved_at_utc": None,
        "file_mtime_utc": mtime.isoformat(timespec="seconds"),
        "retrieval": "manual placement (no download record)",
        "bytes": len(blob),
        "sha256": digest,
    }
    atomic_write_text(side, json.dumps(prov, indent=2))
    return prov


def fetch_french(cfg: DataConfig, force: bool = False) -> list[dict[str, Any]]:
    fr = cfg.french
    out = []
    for role, url in (
        ("industries_zip", fr.industries_url),
        ("factors_zip", fr.factors_url),
        ("definitions_zip", fr.definitions_url),
    ):
        out.append(fetch_file(url, cfg.cache_dir, role, force=force))
    return out


# --------------------------------------------------------------------------------------
# French CSV parsing
# --------------------------------------------------------------------------------------

_DATA_ROW = re.compile(r"^\s*(\d{6,8})\s*,")


@dataclass
class RawSection:
    title: str
    header: list[str]
    dates: list[str]
    rows: list[list[str]]


def read_single_csv_from_zip(blob: bytes) -> tuple[str, str]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
        if len(names) != 1:
            raise DataValidationError(
                f"expected exactly one CSV/TXT member in archive, found {names}"
            )
        text = zf.read(names[0]).decode("latin-1")
    return names[0], text


def parse_french_sections(text: str) -> list[RawSection]:
    """Split a French library CSV into its tables.

    A header row begins with a comma. Its title is the immediately preceding line when that
    line is non-blank text; otherwise the table is unnamed (e.g. the factors file).
    """
    lines = text.splitlines()
    sections: list[RawSection] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(","):
            prev = lines[i - 1].strip() if i > 0 else ""
            title = prev if prev and not _DATA_ROW.match(lines[i - 1]) else ""
            header = [h.strip() for h in line.split(",")[1:]]
            dates: list[str] = []
            rows: list[list[str]] = []
            j = i + 1
            while j < len(lines) and _DATA_ROW.match(lines[j]):
                parts = [p.strip() for p in lines[j].split(",")]
                dates.append(parts[0])
                rows.append(parts[1:])
                j += 1
            sections.append(RawSection(title=title, header=header, dates=dates, rows=rows))
            i = j
        else:
            i += 1
    return sections


def select_section(sections: list[RawSection], title: str) -> RawSection:
    matches = [s for s in sections if s.title == title]
    if len(matches) != 1:
        available = [s.title or "<unnamed>" for s in sections]
        raise DataValidationError(
            f"section {title!r} found {len(matches)} times; available: {available}"
        )
    return matches[0]


def section_to_decimal_frame(sec: RawSection, context: str) -> tuple[pd.DataFrame, int]:
    """Convert a raw percent-unit section to decimal returns. Returns (frame, n_sentinels)."""
    if any(len(d) != 8 for d in sec.dates):
        raise DataValidationError(f"{context}: section is not daily (non-YYYYMMDD dates present)")
    ncol = len(sec.header)
    bad = [k for k, r in enumerate(sec.rows) if len(r) != ncol]
    if bad:
        raise DataValidationError(
            f"{context}: {len(bad)} rows have a column count different from the header"
        )
    raw = np.array(sec.rows, dtype=object)
    sentinel_mask = np.isin(raw, list(FRENCH_SENTINELS))
    raw = np.where(sentinel_mask, "nan", raw)
    try:
        values = raw.astype(np.float64)
    except ValueError as exc:
        raise DataValidationError(f"{context}: non-numeric value found: {exc}") from exc
    values = values / 100.0  # percent -> decimal, the single conversion point for this source
    index = pd.DatetimeIndex(pd.to_datetime(sec.dates, format="%Y%m%d"), name="date")
    frame = pd.DataFrame(values, index=index, columns=sec.header)
    return frame, int(sentinel_mask.sum())


def header_notes(text: str, max_lines: int = 12) -> list[str]:
    notes = []
    for line in text.splitlines()[:max_lines]:
        if line.startswith(",") or _DATA_ROW.match(line):
            break
        if line.strip():
            notes.append(line.strip())
    return notes


def parse_definitions(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(\d{1,2})\s+(\S+)\s+(.+?)\s*$", line)
        if m:
            out[m.group(2)] = m.group(3)
    return out


def load_french_panel(cfg: DataConfig) -> ReturnPanel:
    fr = cfg.french
    files = []
    blobs: dict[str, bytes] = {}
    for role, url in (("industries_zip", fr.industries_url), ("factors_zip", fr.factors_url)):
        path = cfg.cache_dir / _filename_from_url(url)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run `rpo fetch --config <config>` or download {url} manually "
                f"into {cfg.cache_dir}/ (the same parser is used)."
            )
        files.append(verify_cached_file(path, url, role))
        blobs[role] = path.read_bytes()
    def_path = cfg.cache_dir / _filename_from_url(fr.definitions_url)
    names = dict(FRENCH30_NAMES)
    def_check = "definitions file not cached; using transcribed dictionary"
    if def_path.exists():
        files.append(verify_cached_file(def_path, fr.definitions_url, "definitions_zip"))
        _, dtext = read_single_csv_from_zip(def_path.read_bytes())
        parsed = parse_definitions(dtext)
        mismatch = {k for k in names if parsed.get(k, "").strip() != names[k]}
        def_check = (
            "matches provider Siccodes30.txt" if not mismatch else f"differs for {sorted(mismatch)}"
        )

    member, itext = read_single_csv_from_zip(blobs["industries_zip"])
    sections = parse_french_sections(itext)
    sec = select_section(sections, fr.section)
    returns, n_sent = section_to_decimal_frame(sec, member)
    fmember, ftext = read_single_csv_from_zip(blobs["factors_zip"])
    fsecs = [s for s in parse_french_sections(ftext) if fr.rf_column in s.header]
    if len(fsecs) != 1:
        raise DataValidationError(
            f"{fmember}: expected one daily table with an {fr.rf_column} column"
        )
    factors, n_sent_f = section_to_decimal_frame(fsecs[0], fmember)
    rf = factors[fr.rf_column].rename("rf")

    files[0].update(
        {
            "member": member,
            "selected_section": fr.section,
            "sections_in_file": [s.title or "<unnamed>" for s in sections],
            "units_in_file": "percent per day",
            "units_after_parse": "decimal simple return per day",
            "provider_notes": header_notes(itext),
            "sentinels_converted": n_sent,
        }
    )
    files[1].update(
        {
            "member": fmember,
            "selected_column": fr.rf_column,
            "units_in_file": "percent per day",
            "units_after_parse": "decimal simple return per day",
            "provider_notes": header_notes(ftext),
            "sentinels_converted": n_sent_f,
        }
    )
    provenance = {
        "dataset": "Kenneth French Data Library: 30 Industry Portfolios (daily, value-weighted, including distributions); RF from Fama/French 3 Factors (daily)",
        "landing_page": fr.landing_page,
        "description_page": fr.description_page,
        "universe_note": UNIVERSE_NOTE,
        "vintage_note": VINTAGE_NOTE,
        "definitions_check": def_check,
        "files": files,
        "redistribution": "Raw provider files are cached locally and excluded from version control; reproduce them with `rpo fetch` and verify against the recorded SHA-256 values.",
    }
    return finalize_panel(returns, rf, names, provenance, cfg, synthetic=False)


# --------------------------------------------------------------------------------------
# Generic CSV adapter
# --------------------------------------------------------------------------------------

#: Ambiguity thresholds for declared units (daily data). Documented in docs/data_dictionary.md.
DECIMAL_MEDIAN_ABS_MAX = 0.02
PERCENT_MEDIAN_ABS_MIN = 0.05


def _read_dated_csv(path: Path, date_format: str | None, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]
    if "date" not in df.columns:
        raise DataValidationError(f"{label}: a 'date' column is required; found {list(df.columns)}")
    if len(set(df.columns)) != len(df.columns):
        raise DataValidationError(f"{label}: duplicate column names")
    try:
        dates = pd.to_datetime(df["date"].str.strip(), format=date_format or "ISO8601")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"{label}: unparseable dates ({exc})") from exc
    idx = pd.DatetimeIndex(dates, name="date")
    if idx.has_duplicates:
        dup = idx[idx.duplicated()].strftime("%Y-%m-%d").tolist()[:5]
        raise DataValidationError(f"{label}: duplicate dates, e.g. {dup}")
    if not idx.is_monotonic_increasing:
        raise DataValidationError(
            f"{label}: dates are not strictly increasing (sort the file explicitly)"
        )
    body = df.drop(columns=["date"])
    out = pd.DataFrame(index=idx)
    for col in body.columns:
        vals = pd.to_numeric(body[col].str.strip(), errors="coerce")
        bad = body[col].notna() & body[col].str.strip().ne("") & vals.isna()
        if bad.any():
            raise DataValidationError(f"{label}: non-numeric values in column {col!r}")
        out[col] = vals.to_numpy(dtype=float)
    return out


def _check_units(values: np.ndarray, units: str, label: str) -> None:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        raise DataValidationError(f"{label}: no finite values")
    med = float(np.median(finite))
    if units == "decimal":
        if np.nanmax(finite) >= 1.0 or med > DECIMAL_MEDIAN_ABS_MAX:
            raise DataValidationError(
                f"{label}: declared units 'decimal' but values look like percentages "
                f"(median |r|={med:.4g}, max |r|={np.nanmax(finite):.4g}). Declare units: percent "
                "if that is correct; values are never rescaled implicitly."
            )
    elif med < PERCENT_MEDIAN_ABS_MIN:
        raise DataValidationError(
            f"{label}: declared units 'percent' but values look like decimals (median |r|={med:.4g})."
        )


def load_csv_panel(src: CsvSource, cfg: DataConfig) -> ReturnPanel:
    rets = _read_dated_csv(src.returns_path, src.date_format, "returns CSV")
    if rets.shape[1] == 0:
        raise DataValidationError("returns CSV: no asset columns")
    rf_df = _read_dated_csv(src.rf_path, src.date_format, "risk-free CSV")
    if src.rf_column not in rf_df.columns:
        raise DataValidationError(f"risk-free CSV: column {src.rf_column!r} not found")
    _check_units(rets.to_numpy(), src.units, "returns CSV")
    scale = 100.0 if src.units == "percent" else 1.0
    rets = rets / scale  # single conversion point for this source
    rf = (rf_df[src.rf_column] / scale).rename("rf")
    files = []
    for role, path in (("returns_csv", src.returns_path), ("rf_csv", src.rf_path)):
        blob = Path(path).read_bytes()
        files.append(
            {
                "role": role,
                "path": str(path),
                "bytes": len(blob),
                "sha256": sha256_bytes(blob),
                "units_in_file": src.units,
                "frequency": src.frequency,
            }
        )
    provenance = {
        "dataset": "User-supplied CSV return panel",
        "files": files,
        "universe_note": "User-supplied assets.",
    }
    names = {c: c for c in rets.columns}
    return finalize_panel(rets, rf, names, provenance, cfg, synthetic=False)


# --------------------------------------------------------------------------------------
# Synthetic fixture
# --------------------------------------------------------------------------------------


def make_synthetic_panel(src: SyntheticSource, cfg: DataConfig) -> ReturnPanel:
    """Deterministic one-factor Gaussian panel. Clearly labeled SYNTHETIC everywhere."""
    rng = np.random.default_rng(src.seed)
    dates = pd.bdate_range(src.start, src.end, name="date")
    n, t = src.n_assets, len(dates)
    beta = np.linspace(0.6, 1.4, n)
    idio = np.linspace(0.006, 0.016, n)
    alpha = np.linspace(-0.0001, 0.0002, n)
    market = 0.0003 + 0.010 * rng.standard_normal(t)
    eps = rng.standard_normal((t, n)) * idio
    rets = alpha + np.outer(market, beta) + eps
    cols = [f"SYN{k + 1:02d}" for k in range(n)]
    returns = pd.DataFrame(rets, index=dates, columns=cols)
    rf = pd.Series(0.00008 + 0.00002 * np.sin(np.arange(t) / 250.0), index=dates, name="rf")
    names = {c: f"Synthetic asset {k + 1} (beta {beta[k]:.2f})" for k, c in enumerate(cols)}
    provenance = {
        "dataset": "SYNTHETIC one-factor Gaussian demonstration panel (not market data)",
        "generator": {
            "seed": src.seed,
            "n_assets": n,
            "start": str(src.start),
            "end": str(src.end),
        },
        "files": [],
        "universe_note": "SYNTHETIC assets generated by portfolio_engine.data.make_synthetic_panel.",
    }
    return finalize_panel(returns, rf, names, provenance, cfg, synthetic=True)


# --------------------------------------------------------------------------------------
# Shared validation
# --------------------------------------------------------------------------------------


def finalize_panel(
    returns: pd.DataFrame,
    rf: pd.Series,
    names: dict[str, str],
    provenance: dict[str, Any],
    cfg: DataConfig,
    synthetic: bool,
) -> ReturnPanel:
    checks: list[dict[str, Any]] = []

    def add(check: str, ok: bool, value: Any, detail: str = "", level: str = "fail") -> None:
        status = "info" if level == "info" else ("pass" if ok else level)
        checks.append({"check": check, "status": status, "value": value, "detail": detail})

    idx = pd.DatetimeIndex(returns.index)
    add("asset_dates_unique", not idx.has_duplicates, int(idx.duplicated().sum()))
    add("asset_dates_increasing", idx.is_monotonic_increasing, bool(idx.is_monotonic_increasing))
    cols = [str(c) for c in returns.columns]
    add("asset_ids_unique", len(set(cols)) == len(cols), len(cols))
    if cfg.expected_assets is not None:
        add(
            "expected_column_count",
            len(cols) == cfg.expected_assets,
            len(cols),
            f"expected {cfg.expected_assets}",
        )
    hard_fail = [c for c in checks if c["status"] == "fail"]
    if hard_fail:
        raise DataValidationError(f"data validation failed: {hard_fail}")

    start, end = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    after_end = int((idx > end).sum())
    before_start = int((idx < start).sum())
    returns = returns.loc[(idx >= start) & (idx <= end)].copy()
    add(
        "excluded_after_end",
        True,
        after_end,
        f"observations after {cfg.end} excluded",
        level="info",
    )
    add(
        "excluded_before_start",
        True,
        before_start,
        f"observations before {cfg.start} excluded",
        level="info",
    )

    rf = rf.loc[(rf.index >= start) & (rf.index <= end)]
    if rf.index.has_duplicates:
        raise DataValidationError("risk-free series has duplicate dates")
    missing_rf = returns.index.difference(rf.index)
    extra_rf = rf.index.difference(returns.index)
    add(
        "asset_dates_without_rf",
        len(missing_rf) == 0,
        len(missing_rf),
        "asset dates lacking an RF observation are excluded" if len(missing_rf) else "",
        level="warn",
    )
    add("rf_dates_without_assets", True, len(extra_rf), "RF-only dates ignored", level="info")
    returns = returns.loc[returns.index.isin(rf.index)]
    rf = rf.reindex(returns.index)
    if not np.all(np.isfinite(rf.to_numpy())):
        raise DataValidationError(
            "risk-free series contains missing or non-finite values on asset dates"
        )

    vals = returns.to_numpy(dtype=float)
    n_missing = int(np.isnan(vals).sum())
    n_inf = int(np.isinf(vals).sum())
    add("non_finite_values", n_inf == 0, n_inf)
    if n_inf:
        raise DataValidationError("returns contain infinite values")
    if n_missing:
        rows_missing = returns.index[np.isnan(vals).any(axis=1)]
        if cfg.missing_policy == "error":
            raise DataValidationError(
                f"returns contain {n_missing} missing values on {len(rows_missing)} dates "
                f"(first {rows_missing[0].date()}); returns are never forward-filled or zero-filled. "
                "Set data.missing_policy: drop_dates to exclude those dates explicitly."
            )
        returns = returns.drop(index=rows_missing)
        rf = rf.drop(index=rows_missing)
        add(
            "dates_dropped_for_missing",
            True,
            len(rows_missing),
            "complete-case dates only",
            level="warn",
        )
    add("missing_values", n_missing == 0, n_missing, level="warn")
    below = int((returns.to_numpy() <= -1.0).sum())
    add("returns_at_or_below_minus_100pct", below == 0, below)
    if below:
        raise DataValidationError("impossible simple returns at or below -100% present")
    thr = cfg.large_return_threshold
    stacked = pd.Series(returns.stack())
    big = stacked[stacked.abs() > thr]
    large = pd.DataFrame(
        {
            "date": big.index.get_level_values(0),
            "asset": big.index.get_level_values(1),
            "return": big.to_numpy(),
        }
    )
    add(
        "large_abs_returns",
        True,
        len(large),
        f"|r| > {thr:.0%}; diagnostics only, not deleted",
        level="info",
    )
    add("rf_min", True, float(rf.min()), "decimal per day", level="info")
    add("rf_max", True, float(rf.max()), "decimal per day", level="info")
    if len(returns) == 0:
        raise DataValidationError("no observations remain inside the configured date range")
    add("first_date", True, str(returns.index[0].date()), level="info")
    add("last_date", True, str(returns.index[-1].date()), level="info")
    add("n_dates", True, len(returns), level="info")

    summary = pd.DataFrame(
        {
            "asset": cols,
            "name": [names.get(c, c) for c in cols],
            "first_date": [returns[c].first_valid_index() for c in cols],
            "last_date": [returns[c].last_valid_index() for c in cols],
            "n_obs": [int(returns[c].notna().sum()) for c in cols],
            "mean_daily": [float(returns[c].mean()) for c in cols],
            "std_daily": [float(returns[c].std(ddof=1)) for c in cols],
            "min_daily": [float(returns[c].min()) for c in cols],
            "max_daily": [float(returns[c].max()) for c in cols],
        }
    )
    provenance = dict(provenance)
    provenance["coverage"] = {
        "first_date": str(returns.index[0].date()),
        "last_date": str(returns.index[-1].date()),
        "n_dates": len(returns),
        "n_assets": len(cols),
        "configured_start": str(cfg.start),
        "configured_end": str(cfg.end),
    }
    returns.index.name = "date"
    rf.index.name = "date"
    return ReturnPanel(
        returns=returns.astype(float),
        rf=rf.astype(float).rename("rf"),
        asset_names={c: names.get(c, c) for c in cols},
        provenance=provenance,
        quality=pd.DataFrame(checks),
        asset_summary=summary,
        large_returns=large,
        synthetic=synthetic,
    )


def load_panel(cfg: Config) -> ReturnPanel:
    d = cfg.data
    if d.source == "french":
        return load_french_panel(d)
    if d.source == "csv":
        assert d.csv is not None
        return load_csv_panel(d.csv, d)
    return make_synthetic_panel(d.synthetic or SyntheticSource(), d)
