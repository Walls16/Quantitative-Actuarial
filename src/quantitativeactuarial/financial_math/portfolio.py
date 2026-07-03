"""Portfolio analytics and optimization helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import scipy.optimize as opt


def portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """Compute portfolio expected return."""
    return float(np.asarray(weights, dtype=float) @ np.asarray(expected_returns, dtype=float))


def portfolio_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Compute portfolio volatility from a covariance matrix."""
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    return float(np.sqrt(w @ cov @ w))


def portfolio_sharpe(return_: float, volatility: float, risk_free_rate: float = 0.0) -> float:
    """Compute Sharpe ratio."""
    if volatility == 0:
        return 0.0
    return float((return_ - risk_free_rate) / volatility)


def equal_weight_portfolio(asset_names: list[str], expected_returns: np.ndarray, covariance: np.ndarray, risk_free_rate: float) -> tuple[float, float, float, dict[str, float]]:
    """Compute return, volatility, Sharpe ratio, and weights for a 1/N portfolio."""
    n_assets = len(asset_names)
    weights = np.ones(n_assets) / n_assets
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, covariance)
    sharpe = portfolio_sharpe(ret, vol, risk_free_rate)
    return ret, vol, sharpe, {name: float(weight) for name, weight in zip(asset_names, weights)}


def risk_parity_objective(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Squared distance between actual and equal risk contributions."""
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    sigma = portfolio_volatility(w, cov)
    if sigma == 0:
        return 0.0
    marginal = cov @ w / sigma
    contribution = w * marginal
    target = sigma / len(w)
    return float(np.sum((contribution - target) ** 2))


def risk_parity_portfolio(asset_names: list[str], expected_returns: np.ndarray, covariance: np.ndarray, risk_free_rate: float) -> tuple[float, float, float, dict[str, float]]:
    """Optimize a long-only risk-parity portfolio."""
    n_assets = len(asset_names)
    w0 = np.ones(n_assets) / n_assets
    result = opt.minimize(
        risk_parity_objective,
        w0,
        args=(np.asarray(covariance, dtype=float),),
        method="SLSQP",
        bounds=[(1e-4, 1.0)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    weights = np.abs(result.x) / np.sum(np.abs(result.x))
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, covariance)
    sharpe = portfolio_sharpe(ret, vol, risk_free_rate)
    return ret, vol, sharpe, {name: float(weight) for name, weight in zip(asset_names, weights)}


def mvsk_neg_utility(weights: np.ndarray, daily_returns: np.ndarray, lambda2: float = 1.0, lambda3: float = 0.5, lambda4: float = 0.5) -> float:
    """Negative mean-variance-skewness-kurtosis utility for minimization."""
    w = np.asarray(weights, dtype=float)
    port_returns = np.asarray(daily_returns, dtype=float) @ w
    mean = float(np.mean(port_returns))
    variance = float(np.var(port_returns))
    std = float(np.std(port_returns)) + 1e-10
    skewness = float(np.mean(((port_returns - mean) / std) ** 3))
    kurtosis = float(np.mean(((port_returns - mean) / std) ** 4))
    utility = mean - lambda2 * variance + lambda3 * skewness - lambda4 * kurtosis
    return float(-utility)


def mvsk_portfolio(asset_names: list[str], expected_returns: np.ndarray, covariance: np.ndarray, daily_returns: np.ndarray, risk_free_rate: float) -> tuple[float, float, float, dict[str, float]]:
    """Optimize a long-only MVSK utility portfolio."""
    n_assets = len(asset_names)
    w0 = np.ones(n_assets) / n_assets
    result = opt.minimize(
        mvsk_neg_utility,
        w0,
        args=(np.asarray(daily_returns, dtype=float),),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    weights = np.abs(result.x) / np.sum(np.abs(result.x))
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, covariance)
    sharpe = portfolio_sharpe(ret, vol, risk_free_rate)
    return ret, vol, sharpe, {name: float(weight) for name, weight in zip(asset_names, weights)}


def monte_carlo_portfolio_cloud(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float,
    n_simulations: int = 2500,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate random long-only portfolio returns, volatility, and Sharpe ratios."""
    rng = np.random.default_rng(seed)
    expected = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    weights = rng.dirichlet(np.ones(len(expected)), size=n_simulations)
    returns = weights @ expected
    volatilities = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    sharpes = (returns - risk_free_rate) / volatilities
    return weights, returns, volatilities, sharpes


def evaluate_custom_portfolio(
    prices: pd.DataFrame,
    weights_by_asset: dict[str, float],
    expected_return_fn: Callable[[pd.DataFrame], pd.Series],
    covariance_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> tuple[float, float, np.ndarray, list[str]]:
    """Evaluate a user-provided portfolio using injected return/covariance estimators."""
    mu = expected_return_fn(prices)
    covariance = covariance_fn(prices)
    weights = np.array([weights_by_asset.get(column, 0.0) for column in prices.columns])
    total_weight = weights.sum()
    if total_weight > 0:
        weights = weights / total_weight
    ret = portfolio_return(weights, mu.values)
    vol = portfolio_volatility(weights, covariance.values)
    return ret, vol, weights, list(prices.columns)


__all__ = [
    "portfolio_return",
    "portfolio_volatility",
    "portfolio_sharpe",
    "equal_weight_portfolio",
    "risk_parity_objective",
    "risk_parity_portfolio",
    "mvsk_neg_utility",
    "mvsk_portfolio",
    "monte_carlo_portfolio_cloud",
    "evaluate_custom_portfolio",
]
