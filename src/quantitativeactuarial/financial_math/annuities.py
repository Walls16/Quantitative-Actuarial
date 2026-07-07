"""Annuities, perpetuities, gradients, and period solvers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import root_scalar

from .rates import convert_nominal_frequency


def effective_annuity_future_value(
    payment: float, period_rate: float, periods: float, due: bool = False
) -> float:
    """Future value of an ordinary or annuity-due payment stream."""
    R = payment
    i_m = period_rate
    n_m = periods
    anticipada = due
    if i_m == 0:
        return R * n_m
    factor = ((1 + i_m) ** n_m - 1) / i_m
    if anticipada:
        factor *= 1 + i_m
    return R * factor


def nominal_annuity_future_value(
    payment: float,
    nominal_rate: float,
    compounding_frequency: int | float,
    payment_frequency: int | float,
    years: float,
) -> float:
    """Future value of ordinary payments made at a payment frequency under a nominal rate."""
    i_p = convert_nominal_frequency(nominal_rate, compounding_frequency, payment_frequency)
    R = payment
    p = payment_frequency
    n = years
    n_p = n * p
    return effective_annuity_future_value(R, i_p, n_p, due=False)


def continuous_annuity_future_value(annual_flow: float, delta: float, years: float) -> float:
    """Future value of a continuous payment flow under a continuous force of interest."""
    R_anual = annual_flow
    n = years
    if delta == 0:
        return R_anual * n
    return R_anual * (np.exp(delta * n) - 1) / delta


def effective_annuity_present_value(
    payment: float, period_rate: float, periods: float, due: bool = False
) -> float:
    """Present value of an ordinary or annuity-due payment stream."""
    R = payment
    i_m = period_rate
    n_m = periods
    anticipada = due
    if i_m == 0:
        return R * n_m
    factor = (1 - (1 + i_m) ** (-n_m)) / i_m
    if anticipada:
        factor *= 1 + i_m
    return R * factor


def nominal_annuity_present_value(
    payment: float,
    nominal_rate: float,
    compounding_frequency: int | float,
    payment_frequency: int | float,
    years: float,
) -> float:
    """Present value of ordinary payments made at a payment frequency under a nominal rate."""
    i_p = convert_nominal_frequency(nominal_rate, compounding_frequency, payment_frequency)
    n_p = years * payment_frequency
    return effective_annuity_present_value(payment, i_p, n_p, due=False)


def continuous_annuity_present_value(annual_flow: float, delta: float, years: float) -> float:
    """Present value of a continuous payment flow under a continuous force of interest."""
    R_anual = annual_flow
    n = years
    if delta == 0:
        return R_anual * n
    return R_anual * (1 - np.exp(-delta * n)) / delta


def perpetuity_present_value(payment: float, rate: float) -> float:
    """Present value of a level perpetuity."""
    if rate == 0:
        return 0
    return payment / rate


def geometric_gradient_future_value(R1: float, i_m: float, q_m: float, n_m: float) -> float:
    """Future value of a geometric-gradient payment stream."""
    if i_m == q_m:
        return n_m * R1 * ((1 + i_m) ** (n_m - 1))
    numerador = ((1 + i_m) ** n_m) - ((1 + q_m) ** n_m)
    denominador = i_m - q_m
    return R1 * (numerador / denominador)


def geometric_gradient_present_value(R1: float, i: float, q: float, n: float) -> float:
    """Present value of a geometric-gradient payment stream."""
    if i == q:
        return R1 * n / (1 + i)
    return R1 * (1 - ((1 + q) / (1 + i)) ** n) / (i - q)


def periods_for_annuity_future_value(
    future_value: float, payment: float, period_rate: float
) -> float:
    """Solve analytically for the number of periods in a level-annuity future value."""
    VF = future_value
    R = payment
    i_m = period_rate
    if i_m == 0:
        return VF / R
    val = (VF * i_m / R) + 1
    if val <= 0:
        return np.nan
    return np.log(val) / np.log(1 + i_m)


def periods_for_annuity_present_value(
    present_value: float, payment: float, period_rate: float
) -> float:
    """Solve analytically for the number of periods in a level-annuity present value."""
    VP = present_value
    R = payment
    i_m = period_rate
    if i_m == 0:
        return VP / R
    val = 1 - (VP * i_m / R)
    if val <= 0:
        return np.nan
    return -np.log(val) / np.log(1 + i_m)


def periods_for_geometric_gradient_future_value(
    future_value: float, R1: float, i_m: float, q_m: float
) -> float:
    """Solve numerically for the number of periods in a geometric-gradient future value."""
    VF = future_value
    if i_m == q_m:
        f = lambda n: n * R1 * ((1 + i_m) ** (n - 1)) - VF
    else:
        f = lambda n: R1 * (((1 + i_m) ** n - (1 + q_m) ** n) / (i_m - q_m)) - VF
    try:
        res = root_scalar(f, bracket=[0.0001, 2000], method="brentq")
        return res.root
    except Exception:
        return np.nan


def periods_for_geometric_gradient_present_value(
    present_value: float, R1: float, i_m: float, q_m: float
) -> float:
    """Solve numerically for the number of periods in a geometric-gradient present value."""
    VP = present_value
    if i_m == q_m:
        return VP * (1 + i_m) / R1
    else:
        f = lambda n: R1 * (1 - ((1 + q_m) / (1 + i_m)) ** n) / (i_m - q_m) - VP
    try:
        res = root_scalar(f, bracket=[0.0001, 2000], method="brentq")
        return res.root
    except Exception:
        return np.nan


def arithmetic_gradient_present_value(R1: float, G: float, i_m: float, n_m: float) -> float:
    """Present value of an increasing or decreasing arithmetic-gradient annuity."""
    if i_m == 0:
        return R1 * n_m + G * n_m * (n_m - 1) / 2
    an = (1 - (1 + i_m) ** (-n_m)) / i_m
    return R1 * an + (G / i_m) * (an - n_m * (1 + i_m) ** (-n_m))


def arithmetic_gradient_future_value(R1: float, G: float, i_m: float, n_m: float) -> float:
    """Future value of an increasing or decreasing arithmetic-gradient annuity."""
    if i_m == 0:
        return R1 * n_m + G * n_m * (n_m - 1) / 2
    sn = ((1 + i_m) ** n_m - 1) / i_m
    return R1 * sn + (G / i_m) * (sn - n_m)


def periods_for_arithmetic_gradient_future_value(
    future_value: float, R1: float, G: float, i_m: float
) -> float:
    """Solve numerically for the number of periods in an arithmetic-gradient future value."""
    VF = future_value
    f = lambda n: arithmetic_gradient_future_value(R1, G, i_m, n) - VF
    try:
        res = root_scalar(f, bracket=[0.0001, 2000], method="brentq")
        return res.root
    except Exception:
        return np.nan


def periods_for_arithmetic_gradient_present_value(
    present_value: float, R1: float, G: float, i_m: float
) -> float:
    """Solve numerically for the number of periods in an arithmetic-gradient present value."""
    VP = present_value
    f = lambda n: arithmetic_gradient_present_value(R1, G, i_m, n) - VP
    try:
        res = root_scalar(f, bracket=[0.0001, 2000], method="brentq")
        return res.root
    except Exception:
        return np.nan


__all__ = [
    "effective_annuity_future_value",
    "nominal_annuity_future_value",
    "continuous_annuity_future_value",
    "effective_annuity_present_value",
    "nominal_annuity_present_value",
    "continuous_annuity_present_value",
    "perpetuity_present_value",
    "geometric_gradient_future_value",
    "geometric_gradient_present_value",
    "periods_for_annuity_future_value",
    "periods_for_annuity_present_value",
    "periods_for_geometric_gradient_future_value",
    "periods_for_geometric_gradient_present_value",
    "arithmetic_gradient_present_value",
    "arithmetic_gradient_future_value",
    "periods_for_arithmetic_gradient_future_value",
    "periods_for_arithmetic_gradient_present_value",
]
