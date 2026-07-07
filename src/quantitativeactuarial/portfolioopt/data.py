"""Pure data transformations for portfolio optimization.

This module intentionally contains no market-data download logic.  It only
normalizes user-provided tickers and transforms already-loaded price panels
into simple daily return matrices.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a price matrix into simple daily returns.

    Parameters
    ----------
    prices:
        Historical prices with dates on the index and one column per asset.

    Returns
    -------
    pandas.DataFrame
        Daily simple returns, with rows containing missing values removed.
    """
    if prices.empty:
        return prices.copy()
    returns = prices.pct_change().dropna(how="all")
    return returns.dropna()


def validate_tickers(tickers: Iterable[str]) -> list[str]:
    """Normalize a ticker iterable by trimming, uppercasing, and deduplicating.

    Parameters
    ----------
    tickers:
        Iterable of raw ticker strings.

    Returns
    -------
    list[str]
        Unique tickers in first-seen order.
    """
    cleaned: list[str] = []
    for ticker in tickers:
        normalized = ticker.strip().upper()
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


__all__ = ["get_returns", "validate_tickers"]
