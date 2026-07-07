"""
risk_metrics.py
================
Funciones para calcular métricas de riesgo y rendimiento sobre series de
retornos de un portafolio: Sharpe, Sortino, VaR, CVaR, Max Drawdown, Calmar,
volatilidad anualizada, retorno anualizado, etc.

Convención: los retornos son simples (no logarítmicos) y diarios.
TRADING_DAYS = 252 se usa para anualizar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Helpers básicos
# ---------------------------------------------------------------------------
def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Calcula la serie de retornos diarios de un portafolio."""
    weights = np.asarray(weights, dtype=float)
    return returns.dot(weights)


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Curva de capital acumulado (base 1.0)."""
    return (1 + returns).cumprod()


# ---------------------------------------------------------------------------
# Retorno y volatilidad
# ---------------------------------------------------------------------------
def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    if len(returns) == 0:
        return 0.0
    total_return = (1 + returns).prod()
    n_years = len(returns) / periods
    if n_years <= 0:
        return 0.0
    return total_return ** (1 / n_years) - 1


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    if len(returns) < 2:
        return 0.0
    return returns.std(ddof=1) * np.sqrt(periods)


# ---------------------------------------------------------------------------
# Ratios ajustados por riesgo
# ---------------------------------------------------------------------------
def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    vol = annualized_volatility(returns, periods)
    if vol == 0:
        return 0.0
    ret = annualized_return(returns, periods)
    return (ret - rf) / vol


def downside_deviation(
    returns: pd.Series, rf_daily: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    downside = returns[returns < rf_daily] - rf_daily
    if len(downside) == 0:
        return 0.0
    return np.sqrt((downside**2).mean()) * np.sqrt(periods)


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    rf_daily = (1 + rf) ** (1 / periods) - 1
    dd = downside_deviation(returns, rf_daily, periods)
    if dd == 0:
        return 0.0
    ret = annualized_return(returns, periods)
    return (ret - rf) / dd


# ---------------------------------------------------------------------------
# Drawdown y Calmar
# ---------------------------------------------------------------------------
def drawdown_series(returns: pd.Series) -> pd.Series:
    """Serie de drawdown (negativo o cero) a partir de retornos."""
    cum = cumulative_returns(returns)
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    return dd


def max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    dd = drawdown_series(returns)
    return dd.min()  # valor negativo


def calmar_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    ret = annualized_return(returns, periods)
    return ret / abs(mdd)


def conditional_drawdown_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    """CDaR: promedio del peor `alpha` de los drawdowns observados."""
    dd = drawdown_series(returns)
    if len(dd) == 0:
        return 0.0
    losses = -dd  # convertir a positivo
    var_dd = np.quantile(losses, 1 - alpha)
    tail = losses[losses >= var_dd]
    if len(tail) == 0:
        return var_dd
    return tail.mean()


# ---------------------------------------------------------------------------
# Value at Risk / Conditional VaR
# ---------------------------------------------------------------------------
def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """VaR histórico diario (valor negativo = pérdida)."""
    if len(returns) == 0:
        return 0.0
    return np.quantile(returns, alpha)


def cvar_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """CVaR (Expected Shortfall) histórico diario (valor negativo = pérdida)."""
    if len(returns) == 0:
        return 0.0
    var = var_historical(returns, alpha)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return tail.mean()


# ---------------------------------------------------------------------------
# Resumen completo de métricas
# ---------------------------------------------------------------------------
def compute_all_metrics(
    returns: pd.Series,
    rf: float = 0.0,
    periods: int = TRADING_DAYS,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Devuelve un diccionario con todas las métricas relevantes."""
    return {
        "Retorno Anualizado": annualized_return(returns, periods),
        "Volatilidad Anualizada": annualized_volatility(returns, periods),
        "Sharpe Ratio": sharpe_ratio(returns, rf, periods),
        "Sortino Ratio": sortino_ratio(returns, rf, periods),
        "Max Drawdown": max_drawdown(returns),
        "Calmar Ratio": calmar_ratio(returns, periods),
        f"VaR {int((1 - alpha) * 100)}% (diario)": var_historical(returns, alpha),
        f"CVaR {int((1 - alpha) * 100)}% (diario)": cvar_historical(returns, alpha),
    }
