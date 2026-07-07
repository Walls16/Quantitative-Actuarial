"""Market-risk VaR and CVaR primitives."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def parametric_var(
    annual_return: float,
    annual_volatility: float,
    portfolio_value: float,
    confidence_level: float,
    horizon_days: int | float,
) -> tuple[float, float, float, float]:
    """Return normal parametric VaR plus z-score, horizon return, and horizon volatility."""
    t = horizon_days / 252.0
    period_return = annual_return * t
    period_volatility = annual_volatility * np.sqrt(t)
    z_score = norm.ppf(confidence_level)
    var_amount = portfolio_value * (z_score * period_volatility - period_return)
    return max(var_amount, 0), z_score, period_return, period_volatility


def monte_carlo_var_cvar(
    annual_return: float,
    annual_volatility: float,
    portfolio_value: float,
    confidence_level: float,
    horizon_days: int | float,
    simulations: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate VaR and CVaR by simulating normally distributed horizon returns."""
    t = horizon_days / 252.0
    period_return = annual_return * t
    period_volatility = annual_volatility * np.sqrt(t)

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(simulations)
    simulated_returns = period_return + period_volatility * shocks

    alpha = 1.0 - confidence_level
    q_alpha = np.percentile(simulated_returns, alpha * 100)

    tail = simulated_returns[simulated_returns <= q_alpha]
    cvar_alpha = tail.mean() if len(tail) > 0 else q_alpha

    return max(-q_alpha * portfolio_value, 0), max(-cvar_alpha * portfolio_value, 0)


__all__ = ["parametric_var", "monte_carlo_var_cvar"]
