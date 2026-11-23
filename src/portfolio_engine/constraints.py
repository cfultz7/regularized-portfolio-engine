"""Common feasible set: fully invested, long only, per-asset target cap.

Caps bind *target* weights at execution. Later market drift can push a held weight above the
cap; that is reported as a drift exceedance, never as a solver feasibility violation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class InfeasibleConstraintError(ValueError):
    """Raised when the configured constraint set admits no portfolio."""


@dataclass(frozen=True)
class FeasibleSet:
    n_assets: int
    max_weight: float
    tol: float = 1e-6

    def __post_init__(self) -> None:
        if self.n_assets < 1:
            raise InfeasibleConstraintError("at least one asset is required")
        if not 0 < self.max_weight <= 1:
            raise InfeasibleConstraintError("max_weight must lie in (0, 1]")
        if self.n_assets * self.max_weight < 1.0 - 1e-12:
            raise InfeasibleConstraintError(
                f"infeasible caps: N * max_weight = {self.n_assets} * {self.max_weight} < 1; "
                "a fully invested long-only portfolio cannot satisfy them"
            )

    def residuals(self, w: np.ndarray) -> dict[str, float]:
        w = np.asarray(w, dtype=float)
        if w.shape != (self.n_assets,) or not np.all(np.isfinite(w)):
            return {"budget": np.inf, "lower": np.inf, "upper": np.inf, "max": np.inf}
        r = {
            "budget": float(abs(w.sum() - 1.0)),
            "lower": float(max(0.0, -w.min())),
            "upper": float(max(0.0, (w - self.max_weight).max())),
        }
        r["max"] = max(r.values())
        return r

    def is_feasible(self, w: np.ndarray) -> bool:
        return self.residuals(w)["max"] <= self.tol

    def clean(self, w: np.ndarray) -> tuple[np.ndarray, float]:
        """Remove sub-tolerance numerical noise from an accepted solution.

        Only called after residuals pass the tolerance. Returns the cleaned vector (exactly
        summing to one) and the largest absolute adjustment applied.
        """
        w = np.asarray(w, dtype=float)
        c = np.clip(w, 0.0, self.max_weight)
        c = c / c.sum()
        return c, float(np.max(np.abs(c - w)))

    def cap_hits(self, w: np.ndarray) -> int:
        return int(np.sum(np.asarray(w) >= self.max_weight - self.tol))

    def drift_exceedances(self, w: np.ndarray) -> int:
        return int(np.sum(np.asarray(w) > self.max_weight + self.tol))
