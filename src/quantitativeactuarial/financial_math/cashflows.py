"""Discounting helpers for dividend and irregular cash-flow streams."""

from __future__ import annotations

import numpy as np


def present_value_of_dividends(
    dividend_amount: float,
    payments_per_year: int,
    rate: float,
    total_years: float,
    compounding: str = "Continuous",
) -> float:
    """Discount a level dividend stream paid at a fixed frequency."""
    vp_total = 0
    dt = 1 / payments_per_year
    num_pagos = int(total_years * payments_per_year)

    for k in range(1, num_pagos + 1):
        t_pago = k * dt
        if compounding in {"Continuous", "Continua"}:
            vp_total += dividend_amount * np.exp(-rate * t_pago)
        else:
            vp_total += dividend_amount / ((1 + rate) ** t_pago)
    return vp_total


def present_value_of_irregular_cashflows(
    amounts: list[float] | np.ndarray,
    times_years: list[float] | np.ndarray,
    rate: float,
    compounding: str = "Continuous",
) -> float:
    """Discount irregular cash flows at specified times in years."""
    vp_total = 0.0
    for amount, t in zip(amounts, times_years):
        if compounding in {"Continuous", "Continua"}:
            vp_total += amount * np.exp(-rate * t)
        else:
            vp_total += amount / ((1 + rate) ** t)
    return vp_total


__all__ = ["present_value_of_dividends", "present_value_of_irregular_cashflows"]
