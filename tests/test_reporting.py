"""Reports and dashboard consume saved artifacts and agree with exported tables (SYNTHETIC run)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from portfolio_engine.artifacts import read_json
from portfolio_engine.diagnostics import RunData
from portfolio_engine.experiments import reproduce
from portfolio_engine.reporting import facts, generate_all, metrics_display

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reported_run(tmp_path_factory, smoke_config):
    cfg = smoke_config.model_copy(update={"output_dir": tmp_path_factory.mktemp("rep")})
    paths = reproduce(cfg, "core", log=lambda m: None, run_id="rep_run", with_reports=False)
    generate_all(paths, log=lambda m: None)
    return paths


def test_report_numbers_match_exported_metrics(reported_run):
    run = RunData(reported_run)
    csv = pd.read_csv(reported_run.tables / "metrics.csv")
    f = facts(run)
    for sid, vals in f["metrics"].items():
        row = csv[(csv["strategy"] == sid) & (csv["split"] == run.headline)].iloc[0]
        assert vals["sharpe"] == pytest.approx(row["sharpe"], rel=1e-12)
        assert vals["net_cagr"] == pytest.approx(row["net_cagr"], rel=1e-12)
    html = (reported_run.root / "report.html").read_text(encoding="utf-8")
    disp = metrics_display(run, run.headline)
    for sid in disp.index:
        assert disp.loc[sid, "Sharpe"] in html
    assert "SYNTHETIC" in html
    saved = read_json(reported_run.root / "findings.json")
    assert saved["facts"]["metrics"]["EQW"]["sharpe"] == pytest.approx(
        f["metrics"]["EQW"]["sharpe"]
    )


def test_report_outputs_exist_and_pdf_length(reported_run):
    import pypdfium2 as pdfium

    root = reported_run.root
    for p in (
        "report.html",
        "technical_note.pdf",
        "presentation/01_overview.png",
    ):
        assert (root / p).exists(), p
    n = len(pdfium.PdfDocument(str(root / "technical_note.pdf")))
    assert 4 <= n <= 6
    # Only figures are generated for sharing; no text drafts.
    assert sorted(x.suffix for x in (root / "presentation").iterdir()) == [".png"] * 4
    from PIL import Image

    with Image.open(root / "presentation" / "01_overview.png") as im:
        assert im.size == (1600, 1000)


def test_dashboard_renders_saved_run_without_exceptions(reported_run, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("RPO_RUN", str(reported_run.root))
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 5
    assert any("SYNTHETIC" in m.value for m in at.markdown)
    # Switching split and scenario re-slices saved data without errors.
    at.sidebar.selectbox[1].set_value("validation").run()
    at.sidebar.selectbox[2].set_value("cost_bps=25").run()
    assert not at.exception, [e.value for e in at.exception]


def test_dashboard_reports_missing_runs_gracefully(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RPO_RUN", raising=False)
    monkeypatch.setenv("RPO_INCLUDE_BUNDLED", "0")
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=60).run()
    assert not at.exception
    assert any("No saved runs" in e.value for e in at.error)
    assert os.getcwd() == str(tmp_path)
