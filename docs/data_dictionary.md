# Data dictionary

Every run writes `outputs/<run_id>/`. Typed tables are Parquet; human-readable tables are CSV.
`manifest.json` lists each table with its row count and column dtypes (`tables` key), plus the
schema version (`schema_version`, currently `1.0`). Dates are trading dates; returns are decimal
simple returns; dollar values are hypothetical NAV dollars.

## Run-level files

| File | Content |
|---|---|
| `manifest.json` | run id, status (`running`/`completed`/`partial`/`failed`), per-experiment status (`pending`/`running`/`completed`/`failed`/`skipped`), config, strategy definitions, data provenance and hashes, dependency versions, environment, seeds, stage wall times, report inventory |
| `design_freeze.json` | written (exclusive create, never overwritten) after data validation and before any backtest: config, strategies, objective, code id, data hashes |
| `findings.json` | numbers and generated sentences used by the report and share images |
| `report.html`, `technical_note.pdf` | generated from saved tables only |
| `figures/*.png`, `presentation/*` | generated figures and 1600x1000 share images |
| `checkpoints/` | per-cell simulation checkpoints (resume support) |

## Tables (`tables/`)

**data_quality.csv**: `check, status (pass/warn/fail/info), value, detail`.
**asset_summary.csv**: per asset `first_date, last_date, n_obs, mean_daily, std_daily, min_daily, max_daily`.
**large_returns.csv**: `date, asset, return` with |return| above the configured threshold (diagnostic).
**provenance.json**: URLs, retrieval time, SHA-256, selected section, units, provider header notes.

**daily_results.parquet**: one row per strategy and date from the anchor signal.
`date, strategy, scenario, nav_pre, nav_post, nav_before_fees, gross_return, net_return, rf, fee,
gross_traded, turnover_gross, turnover_half, cash_weight, hhi, effective_n, max_weight,
drift_cap_exceedances, is_anchor, is_signal, is_execution, is_initial_deployment, event_status`.
`gross_return` = pretrade NAV / previous post-event NAV - 1; `net_return` uses post-fee NAV.

**weights.parquet** (long): `date, signal_date, strategy, asset, weight_type, value, scenario` with
`weight_type` in `signal_holding` (drifted at signal close), `target` (frozen),
`exec_pretrade`, `exec_posttrade`.

**rebalance_events.parquet / .csv**: one row per strategy and signal. Dates and status
(`executed`, `pending`, `failed`); `solver, solver_status, objective, solve_time_s,
residual_budget/lower/upper/max, cleanup_adjustment, solver_attempts (JSON)`;
`nav_signal, signal_cash_weight, target_cap_hits, feasibility_violation`;
planned `planned_trade_fraction, planned_half_turnover, planned_cost_fraction` (signal-date
estimate from `w_signal`); realized `realized_trade_fraction, realized_half_turnover,
realized_cost_fraction, fee, nav_pre_exec, nav_post_exec, cost_fraction_p,
exec_drift_from_signal`; risk `forecast_vol_lw_ann, forecast_vol_sample_ann, realized_vol_ann,
holding_days` (ex post, complete intervals only); `target_hhi, target_effective_n,
target_max_weight, is_initial_deployment`.

**notional_reallocations.csv**: `interpretation, signal_date, execution_date, strategy, asset,
pretrade_dollars, posttrade_dollars, change_dollars`. Hypothetical reallocations between academic
research portfolios, not orders or share quantities.

**estimation_diagnostics.parquet**: per signal `lookback, window_start, window_end, n_obs,
lw_alpha (covariance shrinkage), rho (fixed mean shrinkage), mu_sample_xs_mean,
mu_sample_xs_std, mu_shrunk_xs_std, trace_sample, min/max_eig_sample, cond_sample,
min/max_eig_lw, cond_lw, sample_repaired, lw_repaired, avg_correlation, notes`.

**estimation_vectors.parquet**: per signal and asset `rank, mu_sample, mu_shrunk,
eig_sample_desc, eig_lw_desc` (eigenvalues sorted descending, indexed by `rank`).

**covariance_matrices.parquet**: per signal `kind (sample/lw), asset` and one column per asset.

**metrics.csv**: per `strategy, scenario, split`: anchor date/NAV, `n_days, end_nav,
total_return, net_cagr, gross_cagr_before_fees, ann_vol, sharpe, sharpe_undefined_reason,
sharpe_gross_before_fees, rf_cagr, max_drawdown, mdd_peak_date, mdd_trough_date,
mdd_recovery_date, avg_hhi, avg_effective_n, avg_max_weight, n_executions,
recurring_half_turnover_ann, recurring_gross_turnover_ann, initial_deployment_in_split,
initial_deployment_fees, total_fees, recurring_fees, fees_pct_of_anchor_nav,
drift_cap_exceedance_days, n_signals, target_cap_hits, avg_target_cap_hits, solver_failures,
feasibility_violations, accepted_inaccurate`.

**sensitivity_results.csv**: `sweep, parameter, value, is_primary` + metrics columns, for every
prespecified cell and split. **sensitivity_daily.parquet**: daily NAV and returns per scenario.

**bootstrap_comparisons.csv**: `split, strategy, reference, statistic, observed_difference,
boot_mean, boot_se, ci_low, ci_high, ci_level, interval_excludes_zero, share_boot_diff_positive,
block_length, replications, seed, n_days, analysis (prespecified / exploratory_block_length)`.

**stress_summary.csv / stress_bands.parquet**: terminal-wealth and max-drawdown quantiles per
strategy; per-day 5/50/95% wealth bands; settings and history used.

**frontier_snapshot.csv**: estimated ex ante frontier points and strategy targets at one signal.

**simulation_replications.parquet / simulation_summary.csv / simulation_truth.csv**
(SYNTHETIC): per replication and method `regret, pop_ann_return, pop_ann_vol, hhi,
effective_n, l1_distance_to_oracle, cov_rel_error_sample, cov_rel_error_lw, lw_alpha, w00..`;
summaries by `case, n_assets, n_obs, method`; population moment summaries.

## Generic CSV input

`returns.csv`: column `date` (ISO `YYYY-MM-DD` or `date_format`) plus one column per asset of
**decimal simple daily returns** (or percent, if declared). `rf.csv`: `date` and the RF column.
Units must be declared (`units: decimal | percent`, `frequency: daily`). Values are never rescaled
implicitly: with `units: decimal`, any |r| >= 1 or a median |r| above 0.02 is rejected as
ambiguous; with `units: percent`, a median |r| below 0.05 (percent units) is rejected. Dates must be unique and
strictly increasing.
