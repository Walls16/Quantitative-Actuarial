"""Equity valuation primitives."""

from __future__ import annotations


def gordon_shapiro_valuation(
    next_dividend: float, required_return: float, growth_rate: float
) -> float:
    """Value an equity share with the Gordon-Shapiro constant-growth dividend model."""
    if required_return <= growth_rate:
        return None
    return next_dividend / (required_return - growth_rate)


def required_equity_return(next_dividend: float, current_price: float, growth_rate: float) -> float:
    """Return the implied required equity return from dividend yield plus growth."""
    if current_price <= 0:
        return None
    return (next_dividend / current_price) + growth_rate


def multiples_valuation(value_metric: float, target_multiple: float) -> float:
    """Value an asset or company by multiplying a metric by a target multiple."""
    return value_metric * target_multiple


__all__ = ["gordon_shapiro_valuation", "required_equity_return", "multiples_valuation"]
