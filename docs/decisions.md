# Decisions and specification notes

Recorded resolutions of points the specification left open, and unavoidable changes.

1. **Python and packaging.** Python 3.12.16 managed by uv 0.13.9; `uv.lock` is the tested
   lockfile (universal: it resolves for macOS, Linux, and Windows). `requires-python` is pinned
   to 3.12.
2. **Paths.** Configuration paths are relative to the working directory (run from the repo
   root) and handled with `pathlib`; no OS-specific separators or shell calls in the engine.
   CLI output is ASCII so Windows consoles render it under default code pages.
3. **Dashboard location.** `app/dashboard.py` lives at the repository root as specified, so
   `rpo dashboard` requires a source checkout (editable install, the default with `uv sync`).
   The dashboard is read-only: it never re-optimizes. Changing the date range re-anchors metrics
   on saved daily rows. The spec's "explicitly triggered new calculation with its own run ID" is
   met by the CLI (`rpo reproduce`), not by dashboard buttons.
4. **Dashboard theme.** Light theme is forced at launch (`--theme.base light`); no dark theme.
5. **Month-end signal at the panel end.** The final date counts as a signal only if it is the
   last weekday of its month, so a panel ending mid-month does not create a fake month-end.
6. **Covariance factorization.** The quadratic form uses an eigen factor of the (repaired)
   covariance so CVXPY sees a PSD form by construction and the problem stays DPP-compliant.
7. **Objective scaling.** The whole objective is multiplied by a positive constant
   (`OBJECTIVE_SCALE = 100`; minimum variance additionally by `(5/2)*21`) to keep solver
   tolerances meaningful at daily-variance scales. The argmin is unchanged; reported objective
   values use the unscaled formula.
8. **Solution cleanup.** Accepted solutions (residual <= 1e-6) are clipped to the box and
   renormalized so targets sum to exactly one; the largest adjustment is recorded.
9. **Forecast risk for EQW.** EQW uses no estimates, but its forecast volatility diagnostic is
   evaluated with the same signal-date Ledoit-Wolf matrix used for the other methods.
10. **Allocation caching.** Targets of path-independent methods (all but COST_AWARE_REG_MV)
    depend only on signal-date estimates and parameters, so they are cached across sweep
    scenarios that share them. The cost-aware method is always re-solved because it depends on
    the drifted weights.
11. **rho sweep scope.** Applied to MEAN_COV_SHRINK_MV only, as specified; the rho = 1 cell can
    be compared with LW_MINVAR.
12. **Stress view.** Uses the latest *executed* allocation (signal 2025-11-28, execution
    2025-12-01 in the primary run), not the pending 2025-12-31 proposal.
13. **Covariance-shift experiment.** Optional; not implemented. No regime-switching model is
    claimed.
14. **Type checking.** mypy checks all engine modules. `reporting.py` (plotting/PDF layout) is
    excluded because matplotlib/ReportLab stub typing produces noise without checking numerics;
    it is covered by tests that compare report numbers with exported tables.
15. **PDF renderer for tests.** `pypdfium2` (dev dependency, wheels for macOS/Windows/Linux)
    counts pages and renders them for visual inspection.
16. **Data vintage observed.** The downloaded files state they were created from the 202609
    CRSP database and contain data through 2026-09-30; 2026 rows are excluded.
17. **Git.** The Xcode Command Line Tools (and therefore git) were not installed when the
    engine was first built, so runs created before git was available record
    `code.git_commit = null` with a note. The headline runs were regenerated after the first
    commits; all metrics were identical to the pre-git runs.
18. **Processed-data cache.** Parsing the cached ZIPs takes about 0.3 s per run, so no processed
    copy is written; `data/processed/` and `data.processed_dir` are reserved. The raw bytes stay
    the single source, verified by SHA-256 on every load.
19. **Units ambiguity thresholds.** Generic CSVs declared `decimal` are rejected if any |r| >= 1
    or the median |r| > 0.02. CSVs declared `percent` are rejected if the median |r| < 0.05. The
    second threshold was raised from 0.005 after an end-to-end `rpo optimize` check showed that
    ordinary decimal equity data (median |r| about 0.008) slipped through.
