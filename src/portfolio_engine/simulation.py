"""Controlled Monte Carlo experiments isolating estimation error. ALL OUTPUT IS SYNTHETIC.

Stationary multivariate Gaussian daily returns with fixed, recorded population moments
(daily arithmetic scales). With ``R = 0.7 I + 0.3 11'``:

* case 1 ``equal_exchangeable``: ``mu_i = 0.00025``, ``Sigma = 0.012**2 R``
  (population 1/N is optimal within the common feasible set);
* case 2 ``equal_mean_hetero_risk``: same mean, ``sigma = linspace(0.008, 0.020, N)``,
  ``Sigma = diag(sigma) R diag(sigma)``;
* case 3 ``hetero_mean_risk``: case-2 covariance, ``mu = linspace(0.00005, 0.00045, N)``.

These are controlled assumptions, not a claim that empirical returns are Gaussian. Each
replication draws one dataset used by every estimator (common random numbers). Generators
come from ``numpy.random.SeedSequence([base_seed, case, N, T, rep])`` so adding cells or
replications never changes existing ones. Costs are zero, so COST_AWARE_REG_MV is identical to
MEAN_COV_SHRINK_MV and is omitted. The ORACLE solves the same constrained mean-variance problem
with the true inputs. The cap is ``3 / N`` (10% at N = 30).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import Manifest, save_table, write_parquet
from .config import Config
from .constraints import FeasibleSet
from .estimates import estimate_moments, psd_factor
from .optimizer import Optimizer
from .strategies import STRATEGIES, AllocationParams, propose

CASES = {1: "equal_exchangeable", 2: "equal_mean_hetero_risk", 3: "hetero_mean_risk"}
SIM_METHODS = ("EQW", "SAMPLE_MV", "COV_SHRINK_MV", "MEAN_COV_SHRINK_MV", "LW_MINVAR")


def population(case: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    R = 0.7 * np.eye(n) + 0.3 * np.ones((n, n))
    if case == 1:
        return np.full(n, 0.00025), 0.012**2 * R
    sig = np.linspace(0.008, 0.020, n)
    cov = np.diag(sig) @ R @ np.diag(sig)
    if case == 2:
        return np.full(n, 0.00025), cov
    if case == 3:
        return np.linspace(0.00005, 0.00045, n), cov
    raise ValueError(case)


def rng_for(base_seed: int, case: int, n: int, t: int, rep: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([base_seed, case, n, t, rep]))


def utility(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, gamma: float, horizon: int) -> float:
    return float(horizon * mu @ w - 0.5 * gamma * horizon * w @ sigma @ w)


def run_cell(
    cfg: Config, case: int, n: int, t: int, reps: int, optimizer: Optimizer
) -> pd.DataFrame:
    sc = cfg.experiments.simulation
    cap = sc.cap_multiple / n
    FeasibleSet(n, cap)
    mu, sigma = population(case, n)
    chol = np.linalg.cholesky(sigma)
    true_f = psd_factor(sigma)
    params = AllocationParams(
        gamma=sc.gamma, horizon=sc.horizon, rho=sc.rho, max_weight=cap, cost_rate=0.0
    )
    oracle = optimizer.mean_variance(true_f.factor, mu, sc.gamma, sc.horizon, cap)
    if not oracle.ok:
        raise RuntimeError(f"oracle failed for case {case}, N={n}")
    w_or = oracle.weights
    assert w_or is not None
    u_or = utility(w_or, mu, sigma, sc.gamma, sc.horizon)
    sig_norm = np.linalg.norm(sigma)
    rows = []
    zeros = np.zeros(n)

    def row(
        rep: int,
        method: str,
        w: np.ndarray | None,
        status: str,
        cov_err_s: float,
        cov_err_lw: float,
        alpha: float,
    ) -> dict:
        base = {
            "case": case,
            "case_name": CASES[case],
            "n_assets": n,
            "n_obs": t,
            "rep": rep,
            "method": method,
            "max_weight": cap,
            "status": status,
            "cov_rel_error_sample": cov_err_s,
            "cov_rel_error_lw": cov_err_lw,
            "lw_alpha": alpha,
        }
        if w is None:
            return base
        base.update(
            {
                "regret": u_or - utility(w, mu, sigma, sc.gamma, sc.horizon),
                "pop_ann_return": 252.0 * float(mu @ w),
                "pop_ann_vol": float(np.sqrt(252.0 * w @ sigma @ w)),
                "hhi": float(w @ w),
                "effective_n": float(1.0 / (w @ w)),
                "l1_distance_to_oracle": float(np.abs(w - w_or).sum()),
                **{f"w{i:02d}": float(w[i]) for i in range(n)},
            }
        )
        return base

    rows.append(row(-1, "ORACLE", w_or, oracle.status, np.nan, np.nan, np.nan))
    for rep in range(reps):
        rng = rng_for(sc.base_seed, case, n, t, rep)
        x = mu + rng.standard_normal((t, n)) @ chol.T
        est = estimate_moments(x, cfg.solver.psd_repair_rel_tol)
        e_s = float(np.linalg.norm(est.sample.matrix - sigma) / sig_norm)
        e_lw = float(np.linalg.norm(est.lw.matrix - sigma) / sig_norm)
        for m in SIM_METHODS:
            res = propose(STRATEGIES[m], est, zeros, params, optimizer)
            rows.append(row(rep, m, res.weights, res.status, e_s, e_lw, est.lw_alpha))
    return pd.DataFrame(rows)


def summarize(reps: pd.DataFrame) -> pd.DataFrame:
    out = []
    wcols = [c for c in reps.columns if c.startswith("w") and c[1:].isdigit()]
    for (case, n, t, m), g in reps.groupby(["case", "n_assets", "n_obs", "method"]):
        ok = g.dropna(subset=["regret"])
        wc = [c for c in wcols if c in ok and ok[c].notna().any()]
        disp = float(ok[wc].std(ddof=1).mean()) if len(ok) > 1 and wc else np.nan
        out.append(
            {
                "case": case,
                "case_name": g["case_name"].iloc[0],
                "n_assets": n,
                "n_obs": t,
                "method": m,
                "replications": int((g["rep"] >= 0).sum()),
                "failures": int(g["regret"].isna().sum()),
                "mean_regret": ok["regret"].mean(),
                "median_regret": ok["regret"].median(),
                "p90_regret": ok["regret"].quantile(0.9),
                "mean_pop_ann_vol": ok["pop_ann_vol"].mean(),
                "mean_pop_ann_return": ok["pop_ann_return"].mean(),
                "mean_hhi": ok["hhi"].mean(),
                "mean_effective_n": ok["effective_n"].mean(),
                "mean_l1_to_oracle": ok["l1_distance_to_oracle"].mean(),
                "weight_dispersion": disp,
                "mean_cov_rel_error_sample": g["cov_rel_error_sample"].mean(),
                "mean_cov_rel_error_lw": g["cov_rel_error_lw"].mean(),
                "mean_lw_alpha": g["lw_alpha"].mean(),
                "max_weight": g["max_weight"].iloc[0],
            }
        )
    return pd.DataFrame(out)


def run_simulation(
    cfg: Config,
    preset: str,
    manifest: Manifest,
    optimizer: Optimizer,
    log: Callable[[str], None] = print,
) -> dict:
    sc = cfg.experiments.simulation
    pre = sc.full if preset == "full" else sc.core
    ckpt: Path = manifest.paths.checkpoints
    frames = []
    cells = [(c, n, t) for n in pre.n_assets for c in CASES for t in pre.lengths]
    t0 = time.perf_counter()
    for k, (case, n, t) in enumerate(cells, 1):
        f = ckpt / f"sim_case{case}_n{n}_t{t}_r{pre.replications}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            src = "checkpoint"
        else:
            c0 = time.perf_counter()
            df = run_cell(cfg, case, n, t, pre.replications, optimizer)
            write_parquet(f, df)
            src = f"{time.perf_counter() - c0:.1f}s"
        frames.append(df)
        log(
            f"  simulation cell {k}/{len(cells)} case={case} N={n} T={t} reps={pre.replications} ({src})"
        )
    reps = pd.concat(frames, ignore_index=True)
    summary = summarize(reps)
    summary.insert(0, "label", "SYNTHETIC Monte Carlo (not historical evidence)")
    save_table(
        manifest,
        "simulation_replications",
        reps,
        "SYNTHETIC: per-replication population metrics and weights for each method and the oracle.",
    )
    save_table(
        manifest,
        "simulation_summary",
        summary,
        "SYNTHETIC: Monte Carlo summary by case, N, T, and method (regret vs oracle, risk, concentration, dispersion).",
        csv=True,
        parquet=False,
    )
    truth = []
    for n in pre.n_assets:
        for case in CASES:
            mu, sigma = population(case, n)
            truth.append(
                {
                    "case": case,
                    "case_name": CASES[case],
                    "n_assets": n,
                    "max_weight": sc.cap_multiple / n,
                    "mu_min": mu.min(),
                    "mu_max": mu.max(),
                    "vol_min": float(np.sqrt(np.diag(sigma)).min()),
                    "vol_max": float(np.sqrt(np.diag(sigma)).max()),
                    "offdiag_corr": 0.3,
                }
            )
    save_table(
        manifest,
        "simulation_truth",
        pd.DataFrame(truth),
        "SYNTHETIC: population moments per case and N (equations in simulation.py).",
        csv=True,
        parquet=False,
    )
    return {
        "preset": preset,
        "cells": len(cells),
        "replications_per_cell": pre.replications,
        "n_assets": list(pre.n_assets),
        "lengths": list(pre.lengths),
        "base_seed": sc.base_seed,
        "cap_rule": f"{sc.cap_multiple}/N",
        "cost_aware_note": "COST_AWARE_REG_MV omitted: identical to MEAN_COV_SHRINK_MV at zero cost",
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
