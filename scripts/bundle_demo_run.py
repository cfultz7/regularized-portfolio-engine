"""Copy a trimmed, dashboard-ready subset of a saved run into demo_runs/ for hosted demos.

Keeps only the derived tables the dashboard reads (no raw provider return series) and stores
the large notional-reallocation table as Parquet. Cross-platform: standard library + pandas.

    uv run python scripts/bundle_demo_run.py --run <run_id>
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEEP_FILES = [
    "manifest.json",
    "design_freeze.json",
    "findings.json",
    "report.html",
    "technical_note.pdf",
]
KEEP_TABLES = [
    "daily_results.parquet",
    "weights.parquet",
    "rebalance_events.parquet",
    "estimation_diagnostics.parquet",
    "estimation_vectors.parquet",
    "covariance_matrices.parquet",
    "stress_bands.parquet",
    "metrics.csv",
    "sensitivity_results.csv",
    "bootstrap_comparisons.csv",
    "stress_summary.csv",
    "frontier_snapshot.csv",
    "simulation_summary.csv",
    "simulation_truth.csv",
    "data_quality.csv",
    "asset_summary.csv",
    "validation_summary.md",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id under --output-dir")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--dest", default=str(ROOT / "demo_runs"))
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    import pandas as pd

    from portfolio_engine.artifacts import read_json

    src = Path(args.output_dir) / args.run
    if read_json(src / "manifest.json").get("run_status") != "completed":
        raise SystemExit(f"{src} is not a completed run")
    dest_root = Path(args.dest)
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest = dest_root / args.run
    (dest / "tables").mkdir(parents=True)
    for name in KEEP_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)
    for name in KEEP_TABLES:
        if (src / "tables" / name).exists():
            shutil.copy2(src / "tables" / name, dest / "tables" / name)
    realloc = pd.read_csv(src / "tables" / "notional_reallocations.csv")
    realloc.to_parquet(dest / "tables" / "notional_reallocations.parquet", index=False)
    (dest_root / "README.md").write_text(
        "# Bundled demo run\n\n"
        f"Trimmed copy of headline run `{args.run}` for the hosted dashboard, made by "
        "`scripts/bundle_demo_run.py`. Derived results only (strategy NAVs, weights, estimates, "
        "metrics); no raw provider return files. Regenerate the full run with "
        "`rpo reproduce --config configs/default.yaml --preset core`.\n",
        encoding="utf-8",
    )
    size = sum(p.stat().st_size for p in dest_root.rglob("*") if p.is_file())
    print(f"bundled {args.run} -> {dest} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
