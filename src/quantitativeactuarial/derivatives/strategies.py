"""Option payoff-leg and strategy-profile utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def option_payoff_leg(
    option_type: str, position: int, terminal_prices: np.ndarray, strike: float, premium: float
) -> np.ndarray:
    """Return the payoff of one option leg net of premium."""
    if option_type == "call":
        payoff = np.maximum(terminal_prices - strike, 0)
    elif option_type == "put":
        payoff = np.maximum(strike - terminal_prices, 0)
    else:
        payoff = np.zeros_like(terminal_prices)
    return position * payoff - (position * premium)


def strategy_profile(spot: float, legs: list[dict], points: int = 500) -> pd.DataFrame:
    """Return the net payoff profile for an option strategy over terminal prices."""
    terminal_prices = np.linspace(spot * 0.5, spot * 1.5, points)
    net_payoff = np.zeros_like(terminal_prices)
    data = {"S_T": terminal_prices}

    for leg in legs:
        option_type = leg["option_type"]
        position = leg["position"]
        strike = leg["strike"]
        premium = leg["premium"]
        payoff = option_payoff_leg(option_type, position, terminal_prices, strike, premium)
        net_payoff += payoff
        label = f"{'Long' if position == 1 else 'Short'} {option_type.capitalize()} K={strike}"
        data[label] = payoff

    data["Net Payoff"] = net_payoff
    return pd.DataFrame(data)


__all__ = ["option_payoff_leg", "strategy_profile"]
