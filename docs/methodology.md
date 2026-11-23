# Methodology

This document defines every quantity the engine computes. Code references are to
`src/portfolio_engine/`.

## 1. Question and scope

How do estimation error, portfolio constraints, and turnover change the practical performance of
mean-variance optimization relative to monthly equal weighting? The empirical part evaluates one
prespecified design on the 30 Kenneth French value-weighted industry portfolios. It is a narrower
comparison than DeMiguel, Garlappi and Uppal (2009), not a replication, and it makes no claim of a
new return signal. A finding that 1/N remains competitive is a valid outcome.

## 2. Data (`data.py`)

* Assets: *30 Industry Portfolios, daily, value-weighted, returns including distributions*. These
  are academic research portfolios, not individual stocks or tradable ETFs.
* Risk-free: `RF` column of *Fama/French 3 Factors, daily*.
* The industry file contains a value-weighted and an equal-weighted daily table. The parser splits
  the file at header rows (lines beginning with a comma), names each table by the line immediately
  above it, and selects exactly the table titled `Average Value Weighted Returns -- Daily`. A
  missing or duplicated title is an error.
* The documented sentinels `-99.99` and `-999` are converted to missing **before** any conversion.
  Percentages are divided by 100 exactly once (`section_to_decimal_frame`).
* Sample: 1985-01-01 to 2025-12-31 (1985-1989 is estimation warmup). Later rows (2026) are
  excluded and counted in `data_quality.csv`.
* RF is aligned by trading date. Asset dates without RF are excluded and reported. Returns are
  never forward-filled or zero-filled; any missing asset return is an error unless
  `missing_policy: drop_dates` explicitly drops those dates.
* Validation: increasing unique dates, unique asset codes, expected column count, finite values,
  no simple return at or below -100%. Returns with |r| > 25% are listed as diagnostics only.
* Vintage: the provider rebuilds full histories at each update and CRSP revisions can change past
  values, so this is a current-vintage dataset, not a point-in-time reconstruction. From the
  January 2025 release the provider builds US research returns from CRSP's CIZ (Flat File Format
  2.0) files instead of the discontinued legacy FIZ files; the provider highlights the change in
  how monthly returns are compounded. This project uses daily returns and does not quantify the
  transition's effect on its sample.

## 3. Estimation (`estimates.py`)

For the T x N window `X` ending at the signal close (T = 252 by default):

* `mu = mean(X, axis=0)` (arithmetic daily means).
* `S = Xc' Xc / T`, `Xc = X - mu` (divisor **T**, the normalization of
  `sklearn.covariance.empirical_covariance`, which `LedoitWolf` shrinks).
* Ledoit-Wolf (2004): `Sigma_LW = (1 - alpha) S + alpha (tr(S)/N) I` via
  `LedoitWolf(assume_centered=False)`; `alpha` (`lw_alpha`) is recorded at every signal. The
  target is a scaled identity, not the constant-correlation target of "Honey, I Shrunk the Sample
  Covariance Matrix".
* Mean shrinkage: `mu_shrunk = (1 - rho) mu + rho mean(mu) 1`, fixed `rho = 0.75`. This is a
  transparent fixed rule, not a James-Stein or Bayesian estimator. With `rho = 1` all assets share
  one forecast, so for a fully invested portfolio `mu' w` is constant and, absent a cost penalty,
  the mean-variance problem reduces to minimum variance with the same covariance (tested).
* Numerical repair vs regularization: matrices are symmetrized; eigenvalues in
  `[-1e-10 * max_eig, 0)` are treated as roundoff and set to zero (logged as `*_repaired`); more
  negative eigenvalues raise `CovarianceError`. No ridge is ever added. Condition numbers are
  reported, `inf` for singular matrices.

## 4. Allocation (`optimizer.py`, `strategies.py`)

Common feasible set for every method: `sum(w) = 1`, `0 <= w <= 0.10`. The engine rejects
configurations with `N * max_weight < 1`.

```
minimize  (gamma/2) * H * w' Sigma w  -  H * mu' w  +  kappa * ||w - w_signal||_1
```

gamma = 5, H = 21. kappa = 0 except for COST_AWARE_REG_MV, where kappa is the one-way cost rate
(0.001 at 10 bps). Daily moments times H approximate H-day mean and variance (no serial-correlation
model); the cost term is charged once. Because the feasible set is long-only and fully invested,
`||w||_1 = 1` is constant, so an L1 penalty on *weights* would do nothing; the penalty acts on
*changes* in weights. No L2 weight penalty is added.

| ID | Mean | Covariance | Plans costs |
|---|---|---|---|
| EQW | - | - | no (1/N at each signal, drifts between) |
| SAMPLE_MV | sample | sample | no |
| COV_SHRINK_MV | sample | Ledoit-Wolf | no |
| MEAN_COV_SHRINK_MV | shrunk (rho) | Ledoit-Wolf | no |
| COST_AWARE_REG_MV | shrunk (rho) | Ledoit-Wolf | **yes** |
| LW_MINVAR | none (min variance) | Ledoit-Wolf | no |

All six pay identical realized costs in the ledger.

Solver details: CVXPY with CLARABEL, falling back to OSQP then SCS. The quadratic form is written
as `||F' w||^2` with `Sigma = F F'` from the eigendecomposition (PSD by construction, DPP
compliant so the problem is compiled once and re-solved). A positive constant scales the objective
for numerical conditioning without changing the argmin; reported objective values are recomputed
from the formula above. A solution is accepted only if it is finite and its budget, lower-bound,
and cap residuals are all <= 1e-6; `optimal_inaccurate` statuses pass through the same check.
Accepted solutions have sub-tolerance noise removed (clip to bounds, renormalize; the adjustment
is recorded as `cleanup_adjustment`). If every solver fails, the event is logged as `failed`, the
existing holdings are kept, and the run manifest records unresolved failures.

## 5. Timing and accounting (`backtest.py`, `accounting.py`)

Each daily row is the close(t-1) to close(t) return. On each date:

1. Held risky dollars grow by asset returns; cash grows by RF.
2. Pretrade NAV and drifted weights are computed.
3. A target frozen at an earlier signal executes at today's close.
4. On a month-end signal, estimates use data through today and the target is frozen from today's
   drifted weights `w_signal`, queued for the next trading close.
5. Post-event holdings and returns are recorded.

Three-date example (tested): signal at the 1989-12-29 close; execution at the 1990-01-02 close
(the 1990-01-02 return accrues to the previous holdings, cash here); the new weights first earn
the 1990-01-03 return. The final 2025-12-31 signal has no later trading date in the sample and is
recorded as `pending`. Execution at the next close is a simulation convention; close-to-close data
do not show fills.

Self-financing trade with one-way cost c, pretrade NAV `V_pre`, pretrade weights `b`, target `w`:
solve `p + c * sum|p w_i - b_i| = 1` for p in [0, 1] (Brent's method; the function is strictly
increasing with slope >= 1 - c). Then `V_post = p V_pre`, posttrade holdings `V_post w`, traded
notional `V_post w - V_pre b`, fee `c * sum|traded| = V_pre - V_post`. Both identities are checked
at every execution. The first purchase from cash (b = 0) gives `p = 1/(1+c)`.

Turnover: gross traded notional / pretrade NAV, and half of that (**half-gross turnover**, a
convention). Costs are charged on the full bought-plus-sold notional. Initial deployment is
reported separately from recurring rebalancing. Holdings are not liquidated at the end.

Two different cost comparisons:

* *Fee deduction on the recorded path*: before-fee wealth compounds the actual policy's daily
  gross returns (`nav_before_fees`); net wealth is the reconciled NAV.
* *Cost-scenario sensitivity*: rerun the policies under 0/5/10/25 bps, which changes the
  cost-aware method's decisions as well as the fees.

## 6. Metrics (`metrics.py`)

* Split anchor: the post-event NAV at the last close before the split. Return, CAGR and drawdown
  are measured from it; the drawdown peak resets there. State is never reset between splits.
* CAGR: `(NAV_end / NAV_anchor)^(1/years) - 1`, `years = calendar days / 365.25`.
* Volatility: `sqrt(252) * std(net, ddof=1)`.
* Sharpe: `sqrt(252) * mean(net - rf) / std(net - rf, ddof=1)` with daily RF. Undefined values are
  reported as missing with a reason. Square-root-of-time scaling assumes no serial dependence.
* Concentration: HHI `sum(w^2)` and effective N `1/HHI` on daily drifted weights (invested days).
* Recurring half-gross turnover per year = sum over non-initial executions / (days / 252).
* Target cap hits (targets at the cap), solver failures, feasibility violations (always 0 for an
  accepted solution) and drift-cap exceedance days (held weights drifting above the cap; a
  separate, expected phenomenon).

## 7. Robustness and uncertainty (`experiments.py`)

One factor at a time around the frozen design, every cell reported: cost {0, 5, 10, 25} bps (all
methods rerun); lookback {126, 252, 504} (all methods); rho {0, 0.5, 0.75, 1} for
MEAN_COV_SHRINK_MV; gamma {1, 5, 10} for SAMPLE_MV and MEAN_COV_SHRINK_MV.

Paired moving-block bootstrap vs EQW: aligned daily rows of every strategy and RF are resampled
together in blocks of 20 days, 1,000 replications, seed 20090301, 95% percentile intervals for
`Sharpe(strategy) - Sharpe(EQW)`. The bootstrap is conditional on observed returns: the engine is
not retrained per replication. An interval containing zero means the difference is not resolved;
no multiplicity adjustment is made. The full preset adds block lengths 10 and 40, labeled
exploratory.

Historical stress view: the latest executed allocation's posttrade weights, 1,260 trading days of
returns through its signal date, 63-day horizon, 10-day moving blocks, 1,000 paths, weights drift
with no trades or fees. Outputs are scenario summaries, not calibrated probabilities.

Ex ante frontier snapshot at the latest signal: estimated and in-sample to that window; never an
out-of-sample result.

## 8. Controlled Monte Carlo (`simulation.py`, SYNTHETIC)

`R = 0.7 I + 0.3 11'`. Case 1: `mu_i = 0.00025`, `Sigma = 0.012^2 R`. Case 2: same mean,
`sigma = linspace(0.008, 0.020, N)`, `Sigma = diag(sigma) R diag(sigma)`. Case 3: case-2
covariance, `mu = linspace(0.00005, 0.00045, N)`. Daily units. Cap `3/N`, zero costs, gamma 5,
H 21, rho 0.75. Seeds: `SeedSequence([1729, case, N, T, rep])`, so cells and replications are
independent of preset. Each replication draws one dataset shared by all estimators. The ORACLE
solves the same constrained problem with the true moments. Regret = `U(oracle) - U(method)` with
`U(w) = H mu'w - (gamma/2) H w'Sigma w` under the true moments. COST_AWARE_REG_MV is omitted
because it is identical to MEAN_COV_SHRINK_MV at zero cost. Core: N = 30, T in {126, 252, 504},
100 replications. Full: N in {10, 30, 60}, 200 replications. The optional covariance-shift
experiment is **not implemented**.
