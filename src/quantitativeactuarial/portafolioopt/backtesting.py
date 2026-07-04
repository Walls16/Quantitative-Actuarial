"""
backtesting.py
================
Motor de backtesting "walk-forward" con rebalanceo dinámico.

En cada fecha de rebalanceo (mensual, trimestral o anual) se recalculan los
pesos óptimos usando ÚNICAMENTE datos pasados (ventana de lookback), evitando
sesgo de look-ahead. Entre fechas de rebalanceo los activos evolucionan según
sus propios retornos (drift natural de buy & hold) hasta el siguiente ajuste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .optimization import PortfolioOptimizer

FREQ_MAP = {
    "Mensual": "M",
    "Trimestral": "Q",
    "Anual": "Y",
}


def _get_rebalance_dates(dates: pd.DatetimeIndex, freq_label: str) -> list:
    freq = FREQ_MAP.get(freq_label, "M")
    periods = dates.to_period(freq)
    date_series = pd.Series(dates, index=dates)
    return date_series.groupby(periods).first().tolist()


def run_backtest(
    prices: pd.DataFrame,
    strategy_name: str,
    rebalance_freq: str = "Mensual",
    lookback_days: int = 252,
    rf: float = 0.0,
    allow_short: bool = False,
    initial_capital: float = 10_000.0,
    cvar_alpha: float = 0.05,
    target_vol: float = 0.10,
) -> dict | None:
    """Ejecuta un backtest con rebalanceo periódico.

    Returns
    -------
    dict con:
        - 'equity_curve': pd.Series (valor del portafolio)
        - 'returns': pd.Series (retornos diarios del portafolio)
        - 'weights_history': pd.DataFrame (pesos en cada fecha de rebalanceo)
        - 'benchmark_equity': pd.Series (equal-weight buy & hold)
        - 'benchmark_returns': pd.Series
        - 'rebalance_dates': lista de fechas usadas para rebalanceo
    Devuelve None si no hay suficientes datos históricos.
    """
    returns = prices.pct_change().dropna()
    dates = returns.index
    n_assets = prices.shape[1]

    rebalance_dates = _get_rebalance_dates(dates, rebalance_freq)

    valid_rebalance_dates = []
    for d in rebalance_dates:
        idx = dates.get_loc(d)
        if idx >= lookback_days:
            valid_rebalance_dates.append(d)

    if not valid_rebalance_dates:
        return None

    first_date = valid_rebalance_dates[0]
    start_idx = dates.get_loc(first_date)
    sim_dates = dates[start_idx:]
    rebalance_set = set(valid_rebalance_dates)

    asset_values = None
    portfolio_values = []
    weights_history = {}
    current_weights = None

    extra_kwargs = {}
    if strategy_name in ("Mínimo CVaR", "Mínimo CDaR"):
        extra_kwargs = {"alpha": cvar_alpha}
    elif strategy_name.startswith("Volatilidad Objetivo"):
        extra_kwargs = {"target_vol": target_vol}

    for date in sim_dates:
        idx = dates.get_loc(date)

        if date in rebalance_set:
            total_value = asset_values.sum() if asset_values is not None else initial_capital
            lookback_returns = returns.iloc[idx - lookback_days : idx]

            opt = PortfolioOptimizer(lookback_returns, rf=rf, allow_short=allow_short)
            try:
                current_weights = opt.optimize(strategy_name, **extra_kwargs)
            except (ValueError, np.linalg.LinAlgError):
                current_weights = np.repeat(1.0 / n_assets, n_assets)

            asset_values = total_value * current_weights
            weights_history[date] = current_weights

        asset_values = asset_values * (1 + returns.loc[date].values)
        portfolio_values.append(asset_values.sum())

    equity_curve = pd.Series(portfolio_values, index=sim_dates, name="Portafolio")

    base_date = dates[start_idx - 1] if start_idx > 0 else sim_dates[0]
    base_value = initial_capital
    full_values = pd.concat([pd.Series([base_value], index=[base_date]), equity_curve])
    portfolio_returns = full_values.pct_change().dropna()
    portfolio_returns.name = "Portafolio"

    # --- Benchmark: equal-weight buy & hold sobre el mismo periodo ---
    bench_returns_daily = returns.loc[sim_dates].mean(axis=1)
    bench_full = pd.concat(
        [
            pd.Series([base_value], index=[base_date]),
            base_value * (1 + bench_returns_daily).cumprod(),
        ]
    )
    benchmark_equity = bench_full.iloc[1:]
    benchmark_equity.name = "Equal Weight (Buy & Hold)"
    benchmark_returns = bench_full.pct_change().dropna()
    benchmark_returns.name = "Equal Weight (Buy & Hold)"

    weights_df = pd.DataFrame(weights_history, index=prices.columns).T

    return {
        "equity_curve": equity_curve,
        "returns": portfolio_returns,
        "weights_history": weights_df,
        "benchmark_equity": benchmark_equity,
        "benchmark_returns": benchmark_returns,
        "rebalance_dates": valid_rebalance_dates,
    }
