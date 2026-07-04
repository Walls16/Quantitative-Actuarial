"""
efficient_frontier.py
======================
Calcula la Frontera Eficiente analítica de Markowitz (mínima varianza para
cada nivel de retorno objetivo) y genera una nube de portafolios aleatorios
mediante simulación Monte Carlo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def _portfolio_perf(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0)))
    return ret, vol


def markowitz_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Calcula la frontera eficiente: para una grilla de retornos objetivo
    entre el retorno del portafolio de mínima varianza y el retorno máximo
    de un solo activo, encuentra el portafolio de mínima varianza.

    Returns
    -------
    DataFrame con columnas: Retorno, Volatilidad, Sharpe (rf=0) y los pesos
    de cada activo.
    """
    mu = mean_returns.values
    cov = cov_matrix.values
    n = len(mu)
    tickers = list(mean_returns.index)

    bounds = tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(n))
    x0 = np.repeat(1.0 / n, n)

    # Rango de retornos objetivo
    ret_min = mu.min()
    ret_max = mu.max()
    target_returns = np.linspace(ret_min, ret_max, n_points)

    rows = []
    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        )

        def variance(w):
            return w @ cov @ w

        result = minimize(
            variance,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            continue

        w = result.x
        ret, vol = _portfolio_perf(w, mu, cov)
        row = {"Retorno": ret, "Volatilidad": vol}
        row.update({t: wi for t, wi in zip(tickers, w)})
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Sharpe"] = df["Retorno"] / df["Volatilidad"].replace(0, np.nan)
    return df.sort_values("Volatilidad").reset_index(drop=True)


def monte_carlo_portfolios(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_portfolios: int = 3000,
    rf: float = 0.0,
    allow_short: bool = False,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Genera `n_portfolios` portafolios con pesos aleatorios y calcula su
    retorno, volatilidad y Sharpe ratio.

    Returns
    -------
    DataFrame con columnas: Retorno, Volatilidad, Sharpe y pesos por activo.
    """
    rng = np.random.default_rng(seed)
    mu = mean_returns.values
    cov = cov_matrix.values
    n = len(mu)
    tickers = list(mean_returns.index)

    if allow_short:
        raw = rng.normal(size=(n_portfolios, n))
        weights = raw / np.abs(raw).sum(axis=1, keepdims=True)
    else:
        # Distribución de Dirichlet -> pesos positivos que suman 1
        weights = rng.dirichlet(np.ones(n), size=n_portfolios)

    rets = weights @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    vols = np.clip(vols, 1e-12, None)
    sharpes = (rets - rf) / vols

    df = pd.DataFrame(weights, columns=tickers)
    df.insert(0, "Sharpe", sharpes)
    df.insert(0, "Volatilidad", vols)
    df.insert(0, "Retorno", rets)
    return df
