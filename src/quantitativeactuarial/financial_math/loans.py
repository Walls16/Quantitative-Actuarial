"""Loan amortization schedules."""

from __future__ import annotations

import pandas as pd


def amortization_schedule(principal: float, period_rate: float, periods: int) -> pd.DataFrame:
    """Return a fixed-payment amortization schedule without a period-zero row."""
    if period_rate < 0:
        raise ValueError("period_rate must be >= 0")
    period_count = round(periods)

    if period_rate == 0:
        payment = principal / period_count
    else:
        payment = principal * (period_rate / (1 - (1 + period_rate) ** (-period_count)))

    balance = principal
    rows = []

    for period in range(1, period_count + 1):
        opening_balance = balance
        interest = opening_balance * period_rate
        amortization = payment - interest

        if period == period_count:
            amortization = opening_balance
            balance = 0.0
        else:
            balance -= amortization
            if abs(balance) < 0.01:
                balance = 0.0

        rows.append(
            {
                "Period": period,
                "Opening balance": opening_balance,
                "Interest": interest,
                "Amortization": amortization,
                "Outstanding balance": balance,
            }
        )

    return pd.DataFrame(rows)


__all__ = ["amortization_schedule"]
