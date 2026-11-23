"""Command-line interface: ``rpo``.

Output uses plain ASCII so it renders in macOS, Linux, and Windows consoles alike.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from .config import Config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Regularized Portfolio Optimization Engine: 1/N vs. Markowitz on academic industry return portfolios.",
)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="YAML configuration file.")]


def _load(config: Path) -> Config:
    from .config import load_config

    return load_config(config)


@app.command()
def fetch(
    config: ConfigOpt = Path("configs/default.yaml"),
    force: Annotated[bool, typer.Option(help="Re-download even if cached.")] = False,
) -> None:
    """Download and cache the official Kenneth French files (original bytes + SHA-256)."""
    from .data import fetch_french

    cfg = _load(config)
    if cfg.data.source != "french":
        typer.echo(f"data.source is {cfg.data.source!r}; nothing to fetch.")
        raise typer.Exit(0)
    for prov in fetch_french(cfg.data, force=force):
        path = (cfg.data.cache_dir / prov["filename"]).resolve()
        typer.echo(
            f"{prov['role']:<16} {prov['retrieval']:<11} sha256={prov['sha256'][:16]}...  {path}"
        )


@app.command()
def validate(config: ConfigOpt = Path("configs/default.yaml")) -> None:
    """Validate configuration and data (no optimization)."""
    from .constraints import FeasibleSet
    from .data import load_panel
    from .experiments import validation_summary

    cfg = _load(config)
    panel = load_panel(cfg)
    FeasibleSet(len(panel.assets), cfg.portfolio.max_weight)
    typer.echo(validation_summary(panel))
    typer.echo("configuration and data: OK")


@app.command()
def reproduce(
    config: ConfigOpt = Path("configs/default.yaml"),
    preset: Annotated[str | None, typer.Option(help="core or full (default from config).")] = None,
    no_reports: Annotated[
        bool, typer.Option("--no-reports", help="Skip report generation.")
    ] = False,
) -> None:
    """Run the frozen primary comparison, sweeps, bootstrap, stress, simulation, and reports."""
    from .experiments import reproduce as _rep

    if preset not in (None, "core", "full"):
        raise typer.BadParameter("preset must be 'core' or 'full'")
    cfg = _load(config)
    paths = _rep(cfg, preset, with_reports=not no_reports)
    typer.echo(f"saved run: {paths.root.resolve()}")


@app.command()
def demo(
    output_dir: Annotated[Path, typer.Option(help="Where to save the synthetic run.")] = Path(
        "outputs"
    ),
    no_reports: Annotated[bool, typer.Option("--no-reports")] = False,
) -> None:
    """Fast offline demonstration on a SYNTHETIC panel (no download)."""
    from .config import load_config
    from .experiments import reproduce as _rep

    cfg_path = Path("configs") / "smoke.yaml"
    if not cfg_path.exists():
        raise typer.BadParameter(f"{cfg_path} not found; run `rpo demo` from the repository root")
    cfg = load_config(cfg_path).model_copy(update={"output_dir": output_dir})
    typer.echo(
        "*** SYNTHETIC DEMO: results below are generated from simulated returns, not market data ***"
    )
    paths = _rep(cfg, "core", with_reports=not no_reports)
    typer.echo(f"saved SYNTHETIC run: {paths.root.resolve()}")


@app.command()
def report(
    run: Annotated[str, typer.Option(help="Run id, run directory, or 'latest'.")] = "latest",
    output_dir: Annotated[Path, typer.Option()] = Path("outputs"),
) -> None:
    """Regenerate figures, report.html, technical_note.pdf, and presentation assets from saved results."""
    from .artifacts import find_run
    from .reporting import generate_all

    paths = find_run(output_dir, run)
    generate_all(paths)
    typer.echo(f"reports regenerated in {paths.root.resolve()}")


@app.command()
def dashboard(
    run: Annotated[str, typer.Option(help="Run id, run directory, or 'latest'.")] = "latest",
    output_dir: Annotated[Path, typer.Option()] = Path("outputs"),
    port: Annotated[int, typer.Option()] = 8501,
    headless: Annotated[bool, typer.Option(help="Do not open a browser.")] = False,
) -> None:
    """Launch the Streamlit dashboard on a saved run."""
    from .artifacts import find_run

    paths = find_run(output_dir, run)
    app_file = Path(__file__).resolve().parents[2] / "app" / "dashboard.py"
    if not app_file.exists():
        raise typer.BadParameter(
            f"dashboard script not found at {app_file}; run from a source checkout"
        )
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_file),
        "--server.port",
        str(port),
        "--server.headless",
        "true" if headless else "false",
        "--browser.gatherUsageStats",
        "false",
        "--theme.base",
        "light",
        "--theme.primaryColor",
        "#30638E",
        "--",
        "--run",
        str(paths.root.resolve()),
    ]
    typer.echo(f"dashboard for run {paths.root.name} at http://localhost:{port}")
    raise typer.Exit(subprocess.call(cmd))


@app.command()
def optimize(
    returns_csv: Annotated[
        Path, typer.Argument(help="CSV with 'date' + one decimal simple-return column per asset.")
    ],
    as_of: Annotated[str, typer.Option(help="As-of signal date (YYYY-MM-DD).")],
    units: Annotated[str, typer.Option(help="decimal or percent (required, never inferred).")],
    weights_csv: Annotated[
        Path | None, typer.Option(help="CSV with columns asset,weight (current risky weights).")
    ] = None,
    method: Annotated[str, typer.Option()] = "COST_AWARE_REG_MV",
    lookback: Annotated[int, typer.Option()] = 252,
    gamma: Annotated[float, typer.Option()] = 5.0,
    rho: Annotated[float, typer.Option()] = 0.75,
    max_weight: Annotated[float, typer.Option()] = 0.10,
    cost_bps: Annotated[float, typer.Option()] = 10.0,
    out: Annotated[Path | None, typer.Option(help="Optional CSV for the target weights.")] = None,
) -> None:
    """Propose a target for a user-supplied daily return panel (hypothetical weights only)."""
    import pandas as pd

    from .data import _check_units, _read_dated_csv
    from .optimizer import optimize_user_panel
    from .strategies import STRATEGIES

    if units not in ("decimal", "percent"):
        raise typer.BadParameter("units must be 'decimal' or 'percent'")
    if method not in STRATEGIES:
        raise typer.BadParameter(f"method must be one of {list(STRATEGIES)}")
    df = _read_dated_csv(returns_csv, None, "returns CSV")
    _check_units(df.to_numpy(), units, "returns CSV")
    if units == "percent":
        df = df / 100.0
    cur = None
    if weights_csv is not None:
        wdf = pd.read_csv(weights_csv)
        cur = wdf.set_index("asset")["weight"]
    target, info = optimize_user_panel(
        df, as_of, cur, method, lookback, gamma, 21, rho, max_weight, cost_bps
    )
    for k, v in info.items():
        typer.echo(f"{k}: {v}")
    typer.echo(target.round(6).to_string())
    if out is not None:
        target.rename_axis("asset").reset_index().to_csv(out, index=False)
        typer.echo(f"saved: {out.resolve()}")


@app.command("list-runs")
def list_runs(output_dir: Annotated[Path, typer.Option()] = Path("outputs")) -> None:
    """List saved runs and their status."""
    from .artifacts import read_json

    for d in sorted(p for p in output_dir.glob("*") if (p / "manifest.json").exists()):
        m = read_json(d / "manifest.json")
        typer.echo(
            f"{d.name}  status={m.get('run_status')}  preset={m.get('preset')}  synthetic={m.get('synthetic')}"
        )


if __name__ == "__main__":
    app()
