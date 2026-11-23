"""Run directories, atomic writes, manifests, and typed table I/O.

Writes go to a temporary sibling file followed by ``os.replace`` (atomic on POSIX and on
Windows when the destination is not held open), so an interrupted run can never leave a
half-written table that looks complete. The manifest records per-experiment status
(``pending``, ``running``, ``completed``, ``failed``, ``partial``, ``skipped``).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = "1.0"
EXPERIMENT_STATES = ("pending", "running", "completed", "failed", "partial", "skipped")


def utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _tmp_path(path: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    return Path(name)


def atomic_write_bytes(path: Path, blob: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        tmp.write_bytes(blob)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(Path(path), text.encode("utf-8"))


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(Path(path), json.dumps(obj, indent=2, default=_json_default, allow_nan=True))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o: Any) -> Any:
    if isinstance(o, (dt.date, dt.datetime, pd.Timestamp)):
        return o.isoformat()
    if isinstance(o, Path):
        return o.as_posix()
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        df.to_parquet(tmp, index=False, engine="pyarrow")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_csv(path: Path, df: pd.DataFrame) -> None:
    atomic_write_text(Path(path), df.to_csv(index=False, lineterminator="\n"))


def code_identifier(repo_root: Path | None = None) -> dict[str, Any]:
    """Git commit and dirty flag when available; never fails."""
    root = repo_root or Path(__file__).resolve().parents[2]
    info: dict[str, Any] = {"git_commit": None, "git_dirty": None, "note": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if commit.returncode != 0:
            info["note"] = "git repository or commit unavailable"
            return info
        info["git_commit"] = commit.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        info["git_dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        info["note"] = "git executable unavailable"
    return info


def dependency_versions() -> dict[str, str | None]:
    pkgs = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "cvxpy",
        "clarabel",
        "osqp",
        "scs",
        "pyarrow",
        "pydantic",
        "typer",
        "streamlit",
        "plotly",
        "matplotlib",
        "reportlab",
    ]
    out: dict[str, str | None] = {}
    for p in pkgs:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
    }


def make_run_id(run_name: str, fingerprint: str) -> str:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{run_name}_{fingerprint[:8]}"


@dataclass
class RunPaths:
    root: Path

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def presentation(self) -> Path:
        return self.root / "presentation"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def design_freeze(self) -> Path:
        return self.root / "design_freeze.json"

    def ensure(self) -> RunPaths:
        for p in (self.root, self.tables, self.figures, self.presentation, self.checkpoints):
            p.mkdir(parents=True, exist_ok=True)
        return self


class Manifest:
    """Mutable manifest persisted atomically after every update."""

    def __init__(self, paths: RunPaths, data: dict[str, Any]):
        self.paths = paths
        self.data = data

    @classmethod
    def create(cls, paths: RunPaths, **fields: Any) -> Manifest:
        data = {
            "schema_version": SCHEMA_VERSION,
            "run_status": "running",
            "created_utc": utc_now_iso(),
            "experiments": {},
            "tables": {},
        }
        data.update(fields)
        m = cls(paths, data)
        m.save()
        return m

    @classmethod
    def load(cls, paths: RunPaths) -> Manifest:
        return cls(paths, read_json(paths.manifest))

    def save(self) -> None:
        self.data["updated_utc"] = utc_now_iso()
        write_json(self.paths.manifest, self.data)

    def set_experiment(self, name: str, status: str, **info: Any) -> None:
        if status not in EXPERIMENT_STATES:
            raise ValueError(status)
        entry = self.data["experiments"].setdefault(name, {})
        entry.update({"status": status, "updated_utc": utc_now_iso(), **info})
        self.save()

    def register_table(self, name: str, df: pd.DataFrame, filename: str, description: str) -> None:
        self.data["tables"][name] = {
            "file": f"tables/{filename}",
            "rows": len(df),
            "description": description,
            "columns": {str(c): str(t) for c, t in df.dtypes.items()},
        }
        self.save()


def save_table(
    manifest: Manifest,
    name: str,
    df: pd.DataFrame,
    description: str,
    csv: bool = False,
    parquet: bool = True,
) -> None:
    """Save a table as Parquet (typed) and/or CSV (human-readable) and register its schema."""
    tables = manifest.paths.tables
    fname = None
    if parquet:
        fname = f"{name}.parquet"
        write_parquet(tables / fname, df)
    if csv:
        cname = f"{name}.csv"
        write_csv(tables / cname, df)
        fname = fname or cname
    assert fname is not None
    manifest.register_table(name, df, fname, description)


def find_run(output_dir: Path, run: str | None) -> RunPaths:
    """Resolve a run id (or 'latest') to its directory."""
    output_dir = Path(output_dir)
    if run and Path(run).is_dir() and (Path(run) / "manifest.json").exists():
        return RunPaths(Path(run))
    if run and run != "latest":
        p = output_dir / run
        if not (p / "manifest.json").exists():
            raise FileNotFoundError(f"run {run!r} not found under {output_dir}")
        return RunPaths(p)
    runs = sorted(d for d in output_dir.glob("*") if (d / "manifest.json").exists())
    if not runs:
        raise FileNotFoundError(f"no saved runs under {output_dir}")
    return RunPaths(runs[-1])


def load_table(paths: RunPaths, name: str) -> pd.DataFrame:
    pq = paths.tables / f"{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = paths.tables / f"{name}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"table {name!r} not found in {paths.tables}")
