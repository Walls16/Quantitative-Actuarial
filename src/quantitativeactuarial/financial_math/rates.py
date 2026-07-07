"""Interest-rate convention and reinvestment-table functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def nominal_to_effective_rate(nominal_rate: float, compounds_per_year: int | float) -> float:
    """Convert a nominal annual rate compounded ``m`` times into an effective annual rate."""
    if compounds_per_year <= 0:
        raise ValueError(f"`compounds_per_year` must be positive; received {compounds_per_year}.")
    return (1 + nominal_rate / compounds_per_year) ** compounds_per_year - 1


def effective_to_nominal_rate(effective_rate: float, compounds_per_year: int | float) -> float:
    """Convert an effective annual rate into a nominal annual rate compounded ``m`` times."""
    if compounds_per_year <= 0:
        raise ValueError(f"`compounds_per_year` must be positive; received {compounds_per_year}.")
    return compounds_per_year * ((1 + effective_rate) ** (1 / compounds_per_year) - 1)


def nominal_to_continuous_rate(nominal_rate: float, compounds_per_year: int | float) -> float:
    """Convert a nominal annual rate compounded ``m`` times into a continuous force of interest."""
    if compounds_per_year <= 0:
        raise ValueError(f"`compounds_per_year` must be positive; received {compounds_per_year}.")
    return compounds_per_year * np.log(1 + nominal_rate / compounds_per_year)


def continuous_to_effective_rate(delta: float) -> float:
    """Convert a continuous force of interest into an effective annual rate."""
    return np.exp(delta) - 1


def continuous_to_nominal_rate(delta: float, compounds_per_year: int | float) -> float:
    """Convert a continuous force of interest into a nominal annual rate compounded ``m`` times."""
    if compounds_per_year <= 0:
        raise ValueError(f"`compounds_per_year` must be positive; received {compounds_per_year}.")
    return compounds_per_year * (np.exp(delta / compounds_per_year) - 1)


def convert_nominal_frequency(
    nominal_rate: float, from_frequency: int | float, to_frequency: int | float
) -> float:
    """Convert a nominal annual rate from one compounding frequency to another."""
    if from_frequency <= 0 or to_frequency <= 0:
        raise ValueError(
            "`from_frequency` and `to_frequency` must be positive; "
            f"received from_frequency={from_frequency}, to_frequency={to_frequency}."
        )
    period_effective_rate = (
        (1 + nominal_rate / from_frequency) ** (from_frequency / to_frequency)
    ) - 1
    return period_effective_rate * to_frequency


def reinvestment_table(principal: float, nominal_rate: float, years: float) -> pd.DataFrame:
    """Return accumulated value and return for common reinvestment frequencies."""
    periods = [
        ("Every 4 years", 0.25),
        ("Every 2 years", 0.5),
        ("Annual", 1),
        ("Semiannual", 2),
        ("Quarterly", 4),
        ("Monthly", 12),
        ("Weekly", 52),
        ("Daily", 365),
        ("Hourly", 8760),
        ("Every minute", 525600),
        ("Every second", 31536000),
    ]

    rows = []
    for name, frequency in periods:
        accumulated_amount = principal * ((1 + nominal_rate / frequency) ** (frequency * years))
        cumulative_return = (accumulated_amount / principal) - 1
        rows.append(
            {
                "Reinvestment period": name,
                "m = Times per year": str(frequency),
                "Accumulated amount": accumulated_amount,
                "Cumulative return": cumulative_return,
            }
        )

    continuous_amount = principal * np.exp(nominal_rate * years)
    continuous_return = (continuous_amount / principal) - 1
    rows.append(
        {
            "Reinvestment period": "Continuous",
            "m = Times per year": "infinity",
            "Accumulated amount": continuous_amount,
            "Cumulative return": continuous_return,
        }
    )

    return pd.DataFrame(rows)


__all__ = [
    "nominal_to_effective_rate",
    "effective_to_nominal_rate",
    "nominal_to_continuous_rate",
    "continuous_to_effective_rate",
    "continuous_to_nominal_rate",
    "convert_nominal_frequency",
    "reinvestment_table",
]
