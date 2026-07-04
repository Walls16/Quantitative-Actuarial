"""Historical and implied-volatility analytics."""

from __future__ import annotations

import numpy as np


def realized_vol(prices, window=21, annualize=252):
    """
    Compute rolling realized (historical) volatility from price series.

    Parameters
    ----------
    prices   : array-like of closing prices
    window   : rolling window in days (default 21 = ~1 month)
    annualize: trading days per year for annualization

    Returns
    -------
    rv : np.array of annualized realized vols (same length as prices, NaN for first window)
    """
    prices = np.array(prices, dtype=float)
    log_ret = np.log(prices[1:] / prices[:-1])
    rv = np.full(len(prices), np.nan)
    for i in range(window, len(log_ret) + 1):
        rv[i] = np.std(log_ret[i - window : i], ddof=1) * np.sqrt(annualize)
    return rv


def iv_rv_spread(iv_series, rv_series):
    """
    Compute IV - RV spread (variance risk premium proxy).
    Positive spread = market paying premium for vol insurance.
    """
    iv = np.array(iv_series, dtype=float)
    rv = np.array(rv_series, dtype=float)
    spread = iv - rv
    return {
        "spread": spread,
        "mean_spread": np.nanmean(spread),
        "std_spread": np.nanstd(spread),
        "pct_positive": np.nanmean(spread > 0) * 100,
    }
