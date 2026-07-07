"""Implied-volatility inversion routines."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from ..validation import validate_option_type


def _bsm_price(
    S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str
) -> float:
    if T <= 0:
        return float(max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0))
    if sigma <= 0:
        forward_intrinsic = S * np.exp(-q * T) - K * np.exp(-r * T)
        return float(
            max(forward_intrinsic, 0.0) if option_type == "call" else max(-forward_intrinsic, 0.0)
        )
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def implied_volatility_black_scholes(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """Invert the Black-Scholes-Merton formula by Brent root finding."""
    validate_option_type(option_type)

    def objective(sigma: float) -> float:
        return _bsm_price(S, K, T, r, sigma, q, option_type) - market_price

    low = objective(1e-6)
    high = objective(10.0)
    if np.isnan(low) or np.isnan(high) or low * high > 0:
        return float(np.nan)
    return float(brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter))


def _bachelier_price(
    S: float, K: float, T: float, r: float, sigma_n: float, q: float, option_type: str
) -> float:
    if T <= 0:
        return float(max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0))
    F = S * np.exp((r - q) * T)
    vol = sigma_n * np.sqrt(T)
    if vol <= 0:
        intrinsic = F - K
        return float(
            np.exp(-r * T)
            * (max(intrinsic, 0.0) if option_type == "call" else max(-intrinsic, 0.0))
        )
    d = (F - K) / vol
    if option_type == "call":
        return float(np.exp(-r * T) * ((F - K) * norm.cdf(d) + vol * norm.pdf(d)))
    return float(np.exp(-r * T) * ((K - F) * norm.cdf(-d) + vol * norm.pdf(d)))


def implied_normal_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    tol: float = 1e-8,
) -> float:
    """Invert the Bachelier normal-volatility formula."""
    validate_option_type(option_type)

    def objective(sigma_n: float) -> float:
        return _bachelier_price(S, K, T, r, sigma_n, q, option_type) - market_price

    low = objective(1e-6)
    upper = max(abs(S) * 10.0, abs(K) * 10.0, 1.0)
    high = objective(upper)
    if np.isnan(low) or np.isnan(high) or low * high > 0:
        return float(np.nan)
    return float(brentq(objective, 1e-6, upper, xtol=tol))
