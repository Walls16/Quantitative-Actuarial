"""Forward, futures-style carry, FX forward, and FRA pricing."""

from __future__ import annotations

import numpy as np


def _is_continuous(compounding: str) -> bool:
    return compounding.strip().lower().startswith("continuous")


def _growth_factor(rate: float, maturity: float, compounding: str) -> float:
    if _is_continuous(compounding):
        return float(np.exp(rate * maturity))
    return float((1 + rate) ** maturity)


def _discount_factor(rate: float, maturity: float, compounding: str) -> float:
    return 1.0 / _growth_factor(rate, maturity, compounding)


def forward_price(
    spot: float,
    rate: float,
    maturity: float,
    carry: float = 0.0,
    compounding: str = "Continuous",
) -> float:
    """Price a forward with a financing rate and income yield."""
    if _is_continuous(compounding):
        return float(spot * np.exp((rate - carry) * maturity))
    return float(
        spot
        * _growth_factor(rate, maturity, compounding)
        / _growth_factor(carry, maturity, compounding)
    )


def forward_price_with_yield(
    spot: float, rate: float, yield_rate: float, maturity: float, compounding: str = "Continuous"
) -> float:
    """Return a forward price with a continuous or discrete income yield."""
    return forward_price(spot, rate, maturity, carry=yield_rate, compounding=compounding)


def forward_contract_value(
    spot: float,
    delivery_price: float,
    rate: float,
    yield_rate: float,
    time_to_maturity: float,
    position: str = "Long",
    compounding: str = "Continuous",
) -> float:
    """Return the value of a live forward contract for a long or short position."""
    long_value = live_forward_value(
        spot, delivery_price, rate, yield_rate, time_to_maturity, compounding=compounding
    )
    return long_value if position == "Long" else -long_value


def simple_forward_price(
    spot: float, rate: float, maturity: float, compounding: str = "Continuous"
) -> float:
    """Return the no-income forward price."""
    return forward_price(spot, rate, maturity, compounding=compounding)


def forward_price_with_continuous_dividend(
    spot: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
    compounding: str = "Continuous",
) -> float:
    """Return a forward price with continuous dividend yield or foreign rate."""
    return forward_price(spot, rate, maturity, carry=dividend_yield, compounding=compounding)


def forward_price_with_discrete_dividends(
    spot: float,
    rate: float,
    maturity: float,
    present_value_dividends: float,
    compounding: str = "Continuous",
) -> float:
    """Return a forward price after subtracting the present value of discrete dividends."""
    return float((spot - present_value_dividends) * _growth_factor(rate, maturity, compounding))


def commodity_forward_price(
    spot: float, rate: float, storage_cost: float, maturity: float, compounding: str = "Continuous"
) -> float:
    """Return a commodity forward price with storage cost."""
    if _is_continuous(compounding):
        return float(spot * np.exp((rate + storage_cost) * maturity))
    return float(
        spot
        * _growth_factor(rate, maturity, compounding)
        * _growth_factor(storage_cost, maturity, compounding)
    )


def fx_forward_price(
    spot: float,
    domestic_rate: float,
    foreign_rate: float,
    maturity: float,
    compounding: str = "Continuous",
) -> float:
    """Return an FX forward price from covered interest-rate parity."""
    return forward_price(spot, domestic_rate, maturity, carry=foreign_rate, compounding=compounding)


def live_forward_value(
    spot: float,
    delivery_price: float,
    rate: float,
    yield_rate: float,
    time_to_maturity: float,
    compounding: str = "Continuous",
) -> float:
    """Return the market value of a long forward before maturity."""
    return float(
        spot * _discount_factor(yield_rate, time_to_maturity, compounding)
        - delivery_price * _discount_factor(rate, time_to_maturity, compounding)
    )


def fra(
    r1: float,
    r2: float,
    t1: float,
    t2: float,
    notional: float,
    fixed_rate: float,
    compounding: str = "Continuous",
) -> tuple[float, float]:
    """Return implied forward rate and FRA value for a fixed-rate receiver."""
    tau = t2 - t1
    if _is_continuous(compounding):
        forward_rate = (r2 * t2 - r1 * t1) / tau
    else:
        forward_rate = (
            _growth_factor(r2, t2, compounding) / _growth_factor(r1, t1, compounding)
        ) ** (1 / tau) - 1
    value = notional * (forward_rate - fixed_rate) * tau * _discount_factor(r2, t2, compounding)
    return float(forward_rate), float(value)


__all__ = [
    "forward_price",
    "forward_price_with_yield",
    "forward_contract_value",
    "simple_forward_price",
    "forward_price_with_continuous_dividend",
    "forward_price_with_discrete_dividends",
    "commodity_forward_price",
    "fx_forward_price",
    "live_forward_value",
    "fra",
]
