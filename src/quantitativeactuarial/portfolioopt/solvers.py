"""Shared numerical solvers for portfolio optimization."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import minimize


def normalize_weights(weights: np.ndarray, min_weight: float, max_weight: float) -> np.ndarray:
    """Clip weights to bounds and renormalize them to sum to one."""
    clipped = np.clip(np.asarray(weights, dtype=float), min_weight, max_weight)
    total = clipped.sum()
    if total != 0:
        return clipped / total
    return clipped


def solve_slsqp_weights(
    objective: Callable[[np.ndarray], float],
    n_assets: int,
    min_weight: float,
    max_weight: float,
    bounds: Sequence[tuple[float, float]] | None = None,
    constraints: Sequence[dict] | tuple[dict, ...] | None = None,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Solve a constrained portfolio-weight problem with SLSQP.

    The solver uses the package convention that weights sum to one.  If SciPy
    fails to converge, it returns the feasible initial point rather than raising,
    matching the historical behavior of ``PortfolioOptimizer``.
    """
    if n_assets == 1:
        return np.array([1.0])
    if x0 is None:
        x0 = np.repeat(1.0 / n_assets, n_assets)
    if bounds is None:
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    if constraints is None:
        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    weights = result.x if result.success else x0
    return normalize_weights(weights, min_weight, max_weight)


__all__ = ["normalize_weights", "solve_slsqp_weights"]
