"""The six allocation methods. IDs are stable and used in every table, export, and plot."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .estimates import MomentEstimates
from .optimizer import Optimizer, SolveResult


@dataclass(frozen=True)
class StrategySpec:
    id: str
    name: str
    objective: str  # "equal" | "mv" | "minvar"
    mean_model: str | None  # "sample" | "shrunk" | None
    cov_model: str | None  # "sample" | "lw" | None
    cost_aware: bool
    description: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def uses_estimates(self) -> bool:
        return self.objective != "equal"

    @property
    def path_dependent(self) -> bool:
        """True when the target depends on the current holdings (only via the cost penalty)."""
        return self.cost_aware


STRATEGIES: dict[str, StrategySpec] = {
    s.id: s
    for s in (
        StrategySpec(
            "EQW",
            "Monthly 1/N",
            "equal",
            None,
            None,
            False,
            "Target 1/N in every asset at each monthly signal; weights drift between executions; pays realized costs.",
        ),
        StrategySpec(
            "SAMPLE_MV",
            "Sample Markowitz",
            "mv",
            "sample",
            "sample",
            False,
            "Constrained mean-variance with trailing sample means and empirical covariance; no planned-cost penalty.",
        ),
        StrategySpec(
            "COV_SHRINK_MV",
            "Covariance shrinkage MV",
            "mv",
            "sample",
            "lw",
            False,
            "Sample means; Ledoit-Wolf (scaled-identity target) covariance; no planned-cost penalty.",
        ),
        StrategySpec(
            "MEAN_COV_SHRINK_MV",
            "Mean + covariance shrinkage MV",
            "mv",
            "shrunk",
            "lw",
            False,
            "Means shrunk toward their cross-sectional average with fixed rho; Ledoit-Wolf covariance; no planned-cost penalty.",
        ),
        StrategySpec(
            "COST_AWARE_REG_MV",
            "Regularized cost-aware MV",
            "mv",
            "shrunk",
            "lw",
            True,
            "Same forecasts as MEAN_COV_SHRINK_MV plus an L1 planned-trading-cost penalty kappa*||w - w_signal||_1 with kappa = one-way cost rate. The only method that anticipates costs.",
        ),
        StrategySpec(
            "LW_MINVAR",
            "Shrinkage minimum variance",
            "minvar",
            None,
            "lw",
            False,
            "Minimum forecast variance with Ledoit-Wolf covariance; no return forecast and no cost penalty.",
        ),
    )
}
STRATEGY_IDS: tuple[str, ...] = tuple(STRATEGIES)
STRATEGY_COLORS: dict[str, str] = {
    "EQW": "#4D4D4D",
    "SAMPLE_MV": "#D1495B",
    "COV_SHRINK_MV": "#EDAE49",
    "MEAN_COV_SHRINK_MV": "#00798C",
    "COST_AWARE_REG_MV": "#30638E",
    "LW_MINVAR": "#7B5EA7",
    "ORACLE": "#2A9D8F",
    "RF": "#9E9E9E",
}


@dataclass(frozen=True)
class AllocationParams:
    gamma: float
    horizon: int
    rho: float
    max_weight: float
    cost_rate: float


def propose(
    spec: StrategySpec,
    est: MomentEstimates | None,
    w_signal: np.ndarray,
    params: AllocationParams,
    optimizer: Optimizer,
) -> SolveResult:
    """Form a frozen target from signal-date information only."""
    n = len(w_signal)
    if spec.objective == "equal":
        w = np.full(n, 1.0 / n)
        if w.max() > params.max_weight + optimizer.tol:
            return SolveResult(
                None,
                "failed",
                None,
                None,
                None,
                0.0,
                {"max": np.inf},
                attempts=[{"solver": "analytic", "outcome": "1/N violates cap"}],
            )
        return SolveResult(
            w,
            "analytic",
            "analytic",
            "n/a",
            None,
            0.0,
            {"budget": 0.0, "lower": 0.0, "upper": 0.0, "max": 0.0},
        )
    assert est is not None
    cov = est.cov(spec.cov_model or "lw")
    if spec.objective == "minvar":
        return optimizer.min_variance(cov.factor, params.max_weight)
    mu = est.mean(spec.mean_model or "sample", params.rho)
    kappa = params.cost_rate if spec.cost_aware else 0.0
    return optimizer.mean_variance(
        cov.factor, mu, params.gamma, params.horizon, params.max_weight, kappa, w_signal
    )
