"""Vanilla Black-Scholes-Merton, Black-76, and Greeks."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bsm_option_prices(
    model_type: str, S: float, K: float, T: float, r: float, sigma: float, extra: float = 0.0
) -> tuple[float, float, float, float]:
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0

    d1 = 0.0
    d2 = 0.0
    call = 0.0
    put = 0.0

    if model_type == "Simple":
        d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    elif model_type == "Income":
        S_adj = S - extra
        if S_adj <= 0:
            S_adj = 0.0001
        d1 = (np.log(S_adj / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = S_adj * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S_adj * norm.cdf(-d1)

    elif model_type in ("Yield", "Currency"):
        d1 = (np.log(S / K) + (r - extra + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = S * np.exp(-extra * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-extra * T) * norm.cdf(-d1)

    elif model_type == "Futures":
        d1 = (np.log(S / K) + ((sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = np.exp(-r * T) * (S * norm.cdf(d1) - K * norm.cdf(d2))
        put = np.exp(-r * T) * (K * norm.cdf(-d2) - S * norm.cdf(-d1))

    elif model_type == "Costs":
        S_adj = S + extra
        d1 = (np.log(S_adj / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = S_adj * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S_adj * norm.cdf(-d1)

    return call, put, d1, d2


def bsm_greeks(
    model_type: str, S: float, K: float, T: float, r: float, sigma: float, extra: float = 0.0
) -> tuple[float, float, float, float, float, float, float, float]:
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    q_yield = 0.0
    S_adj = S

    if model_type == "Income":
        S_adj = S - extra
        if S_adj <= 0:
            S_adj = 0.0001
    elif model_type == "Costs":
        S_adj = S + extra
    elif model_type in ("Yield", "Currency"):
        q_yield = extra
    elif model_type == "Futures":
        q_yield = r

    d1 = (np.log(S_adj / K) + (r - q_yield + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    N_neg_d1 = norm.cdf(-d1)
    N_neg_d2 = norm.cdf(-d2)
    nd1 = norm.pdf(d1)

    delta_call = np.exp(-q_yield * T) * Nd1
    delta_put = np.exp(-q_yield * T) * (Nd1 - 1)
    gamma = np.exp(-q_yield * T) * nd1 / (S_adj * sigma * np.sqrt(T))
    vega = S_adj * np.exp(-q_yield * T) * nd1 * np.sqrt(T) / 100

    termino_comun = -(S_adj * np.exp(-q_yield * T) * nd1 * sigma) / (2 * np.sqrt(T))
    theta_call = (
        termino_comun - r * K * np.exp(-r * T) * Nd2 + q_yield * S_adj * np.exp(-q_yield * T) * Nd1
    ) / 365
    theta_put = (
        termino_comun
        + r * K * np.exp(-r * T) * N_neg_d2
        - q_yield * S_adj * np.exp(-q_yield * T) * N_neg_d1
    ) / 365

    rho_call = K * T * np.exp(-r * T) * Nd2 / 100
    rho_put = -K * T * np.exp(-r * T) * N_neg_d2 / 100

    return delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put


def black_scholes(
    S: float, K: float, r: float, sigma: float, T: float, is_call: bool = True, q: float = 0.0
) -> float:
    """Return a scalar Black-Scholes-Merton option premium with continuous yield."""
    call, put, _, _ = bsm_option_prices("Yield", S, K, T, r, sigma, extra=q)
    return call if is_call else put


def black_76(F0: float, K: float, r: float, sigma: float, T: float, is_call: bool = True) -> float:
    """Return a Black-76 option premium on a futures or forward price."""
    call, put, _, _ = bsm_option_prices("Futures", F0, K, T, r, sigma)
    return call if is_call else put


def bsm_d1_d2(
    S: float, K: float, r: float, q: float, sigma: float, T: float
) -> tuple[float, float]:
    """Return Black-Scholes-Merton ``d1`` and ``d2`` with continuous yield."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(d1), float(d1 - sigma * np.sqrt(T))


def calculate_greeks(
    S: float, K: float, r: float, sigma: float, T: float, is_call: bool = True, q: float = 0.0
) -> dict[str, float]:
    """Return Black-Scholes-Merton Greeks with continuous yield."""
    dc, dp, gamma, vega, tc, tp, rc, rp = bsm_greeks("Yield", S, K, T, r, sigma, extra=q)
    delta = dc if is_call else dp
    theta = tc if is_call else tp
    rho = rc if is_call else rp
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


def second_order_greeks(
    S: float, K: float, r: float, q: float, sigma: float, T: float, is_call: bool = True
) -> dict[str, float]:
    """Compute selected second-order BSM Greeks: vanna, charm, color, and raw vega."""
    if T <= 1e-6 or sigma <= 1e-6:
        return {"vanna": 0.0, "vomma": 0.0, "charm": 0.0, "speed": 0.0, "color_val": 0.0}
    d1, d2 = bsm_d1_d2(S, K, r, q, sigma, T)
    pdf_d1 = norm.pdf(d1)
    sqrt_t = np.sqrt(T)
    discount_yield = np.exp(-q * T)
    vanna = -discount_yield * pdf_d1 * d2 / sigma
    vega_raw = S * discount_yield * pdf_d1 * sqrt_t
    vomma = vega_raw * d1 * d2 / sigma * 0.01
    if is_call:
        charm = q * discount_yield * norm.cdf(d1) - discount_yield * pdf_d1 * (
            2 * (r - q) * T - d2 * sigma * sqrt_t
        ) / (2 * T * sigma * sqrt_t)
    else:
        charm = -q * discount_yield * norm.cdf(-d1) - discount_yield * pdf_d1 * (
            2 * (r - q) * T - d2 * sigma * sqrt_t
        ) / (2 * T * sigma * sqrt_t)
    charm = charm / 365.0
    gamma = discount_yield * pdf_d1 / (S * sigma * sqrt_t)
    speed = -gamma / S * (d1 / (sigma * sqrt_t) + 1)
    color = (
        -discount_yield
        * pdf_d1
        / (2 * S * T * sigma * sqrt_t)
        * (2 * q * T + 1 + (2 * (r - q) * T - d2 * sigma * sqrt_t) * d1 / (sigma * sqrt_t))
        / 365.0
    )
    return {
        "vanna": float(vanna),
        "vomma": float(vomma),
        "charm": float(charm),
        "speed": float(speed),
        "color_val": float(color),
    }


def implied_volatility_bsm(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    is_call: bool = True,
    q: float = 0.0,
    lower: float = 1e-6,
    upper: float = 10.0,
) -> float:
    """Solve BSM implied volatility with Brent's method."""

    def objective(sigma: float) -> float:
        return black_scholes(S, K, r, sigma, T, is_call, q) - market_price

    return float(brentq(objective, lower, upper, xtol=1e-8, maxiter=200))


__all__ = [
    "bsm_option_prices",
    "bsm_greeks",
    "black_scholes",
    "black_76",
    "bsm_d1_d2",
    "calculate_greeks",
    "second_order_greeks",
    "implied_volatility_bsm",
]
