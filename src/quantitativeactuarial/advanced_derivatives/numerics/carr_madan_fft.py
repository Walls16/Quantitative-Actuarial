"""Carr-Madan FFT option-pricing routine."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import interp1d

from ..validation import validate_option_type


def carr_madan_call_price(
    characteristic_function: Callable[[complex | np.ndarray], complex | np.ndarray],
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
    *,
    alpha: float = 1.5,
    n_grid: int = 4096,
    eta: float = 0.25,
) -> float:
    """Price a European call from a log-price characteristic function."""
    lam = 2 * np.pi / (n_grid * eta)
    b = np.pi / eta
    k_grid = -b + lam * np.arange(n_grid)
    v_grid = eta * np.arange(n_grid)
    cf_vals = characteristic_function(v_grid - (alpha + 1) * 1j)
    denom = alpha**2 + alpha - v_grid**2 + 1j * (2 * alpha + 1) * v_grid
    psi_vals = np.exp(-rate * maturity) * cf_vals / denom
    simpson = (eta / 3) * np.array([3 + (-1) ** j - (1 if j == 0 else 0) for j in range(n_grid)])
    fft_input = np.exp(1j * b * v_grid) * psi_vals * simpson
    prices_raw = np.real(np.fft.fft(fft_input)) * np.exp(-alpha * k_grid) / np.pi
    interpolator = interp1d(k_grid, prices_raw, kind="cubic", fill_value="extrapolate")
    call = float(interpolator(np.log(strike)))
    intrinsic_floor = max(
        spot * np.exp(-dividend_yield * maturity) - strike * np.exp(-rate * maturity), 0.0
    )
    return max(call, intrinsic_floor)


def carr_madan_option_price(
    characteristic_function: Callable[[complex | np.ndarray], complex | np.ndarray],
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    *,
    alpha: float = 1.5,
    n_grid: int = 4096,
    eta: float = 0.25,
) -> float:
    """Price a European call or put through Carr-Madan and put-call parity."""
    validate_option_type(option_type)
    call = carr_madan_call_price(
        characteristic_function,
        spot,
        strike,
        maturity,
        rate,
        dividend_yield,
        alpha=alpha,
        n_grid=n_grid,
        eta=eta,
    )
    if option_type == "call":
        return call
    return call - spot * np.exp(-dividend_yield * maturity) + strike * np.exp(-rate * maturity)
