"""Monte Carlo option pricing routines."""

from __future__ import annotations

import numpy as np

from .models.black_scholes import BSMEngine
from .validation import validate_option_type


def monte_carlo_bsm(
    S,
    K,
    T,
    r,
    sigma,
    q=0.0,
    option_type="call",
    n_paths=50_000,
    seed=42,
    antithetic=True,
    control_variate=True,
):
    """Monte Carlo pricing of European options under GBM with variance reduction."""
    validate_option_type(option_type)
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
        if len(z) < n_paths:
            z = np.concatenate([z, rng.standard_normal(1)])
    else:
        z = rng.standard_normal(n_paths)
    n_effective = len(z)
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0.0)
        bsm_price = BSMEngine(S, K, T, r, sigma, q).call_price()
    else:
        payoffs = np.maximum(K - ST, 0.0)
        bsm_price = BSMEngine(S, K, T, r, sigma, q).put_price()
    discounted_payoff = np.exp(-r * T) * payoffs
    if control_variate:
        control = np.exp(-r * T) * ST
        control_expectation = S * np.exp(-q * T)
        control_variance = np.var(control, ddof=1)
        beta_cv = (
            np.cov(discounted_payoff, control, ddof=1)[0, 1] / control_variance
            if control_variance > 0
            else 0.0
        )
        estimator = discounted_payoff - beta_cv * (control - control_expectation)
    else:
        estimator = discounted_payoff
    price = float(np.mean(estimator))
    std_err = float(np.std(estimator, ddof=1) / np.sqrt(n_effective))
    z95 = 1.96
    return {
        "price": price,
        "std_error": std_err,
        "ci_95_lo": float(price - z95 * std_err),
        "ci_95_hi": float(price + z95 * std_err),
        "n_paths": n_paths,
        "bsm_price": float(bsm_price),
        "error_vs_bsm": float(abs(price - bsm_price)),
    }
