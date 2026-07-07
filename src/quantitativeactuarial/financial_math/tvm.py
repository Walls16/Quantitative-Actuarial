"""Time-value-of-money primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd


def future_value(principal: float, rate: float, periods: float) -> float:
    """Return the future value under discrete compounding."""
    return principal * (1 + rate) ** periods


def continuous_future_value(principal: float, delta: float, periods: float) -> float:
    """Return the future value under continuous compounding."""
    return principal * np.exp(delta * periods)


def present_value(future_amount: float, rate: float, periods: float) -> float:
    """Return the present value under discrete discounting."""
    return future_amount / (1 + rate) ** periods


def continuous_present_value(future_amount: float, delta: float, periods: float) -> float:
    """Return the present value under continuous discounting."""
    return future_amount * np.exp(-delta * periods)


def number_of_periods(principal: float, future_amount: float, rate: float) -> float:
    """Solve the number of periods implied by a principal, future amount, and rate."""
    if principal == 0 or rate <= 0:
        return 0
    return np.log(future_amount / principal) / np.log(1 + rate)


def rate_of_return(principal: float, future_amount: float, periods: float) -> float:
    """Solve the discrete rate of return implied by two cash amounts and time."""
    if principal == 0 or periods == 0:
        return 0
    return (future_amount / principal) ** (1 / periods) - 1


def decompose_periods(periods: float) -> pd.DataFrame:
    """Break a fractional year count into years, months, days, hours, minutes, and seconds."""
    years = int(periods)
    frac_years = periods - years

    raw_months = frac_years * 12
    months = int(raw_months)
    frac_months = raw_months - months

    raw_days = frac_months * (365 / 12)
    days = int(raw_days)
    frac_days = raw_days - days

    raw_hours = frac_days * 24
    hours = int(raw_hours)
    frac_hours = raw_hours - hours

    raw_minutes = frac_hours * 60
    minutes = int(raw_minutes)
    frac_minutes = raw_minutes - minutes

    raw_seconds = frac_minutes * 60
    seconds = int(raw_seconds)

    return pd.DataFrame(
        [
            {
                "Years": years,
                "Months": months,
                "Days": days,
                "Hours": hours,
                "Minutes": minutes,
                "Seconds": seconds,
            }
        ]
    )


__all__ = [
    "future_value",
    "continuous_future_value",
    "present_value",
    "continuous_present_value",
    "number_of_periods",
    "rate_of_return",
    "decompose_periods",
]
