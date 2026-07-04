"""Streamlit-side market data services for the Optima portfolio page."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_prices(
    tickers: tuple[str, ...],
    start: str | dt.date,
    end: str | dt.date,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance for app consumption."""
    clean_tickers = tuple(t.strip().upper() for t in tickers if t.strip())
    if not clean_tickers:
        return pd.DataFrame()

    raw = yf.download(
        list(clean_tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        closes = {}
        for ticker in clean_tickers:
            try:
                closes[ticker] = raw[ticker]["Close"]
            except (KeyError, TypeError):
                continue
        prices = pd.DataFrame(closes)
    else:
        close_col = "Close" if "Close" in raw.columns else raw.columns[0]
        prices = raw[[close_col]].copy()
        prices.columns = clean_tickers

    return prices.dropna(how="all").sort_index()


@st.cache_data(show_spinner=False, ttl=30)
def get_live_quotes(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Download recent closes to display live-ish quote changes in Streamlit."""
    clean_tickers = tuple(t.strip().upper() for t in tickers if t.strip())
    if not clean_tickers:
        return pd.DataFrame()

    data = yf.download(
        list(clean_tickers),
        period="5d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    rows: list[dict[str, float | str]] = []
    for ticker in clean_tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                series = data[ticker]["Close"].dropna()
            else:
                series = data["Close"].dropna()

            if len(series) >= 2:
                last_price = float(series.iloc[-1])
                prev_price = float(series.iloc[-2])
            elif len(series) == 1:
                last_price = float(series.iloc[-1])
                prev_price = last_price
            else:
                continue

            change = last_price - prev_price
            pct_change = (change / prev_price) * 100 if prev_price != 0 else 0.0
            rows.append(
                {
                    "Ticker": ticker,
                    "Precio": last_price,
                    "Cambio": change,
                    "Cambio %": pct_change,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)
