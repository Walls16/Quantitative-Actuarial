"""Application-only market-data services.

These routines intentionally live outside the core package because they perform
network I/O through Yahoo Finance and therefore are not mathematically pure.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf
from quantitativeactuarial.financial_math import (
    evaluate_custom_portfolio_from_prices,
    log_return_volatility,
    optimize_portfolio_strategies,
)


def optimizacion_portafolios(
    tickers_list: Any, start_date: Any, end_date: Any, r_f: Any = 0.05
) -> Any:
    """
    Calcula 5 estrategias de ponderación de portafolios:
      1. Markowitz — Máximo Sharpe Ratio
      2. Markowitz — Mínima Varianza Global
      3. 1/N — Equiponderación
      4. Paridad de Riesgo (Risk Parity)
      5. MVSK — Media-Varianza-Asimetría-Curtosis (via scipy)
    Retorna data, mu, S, dict_resultados, nube_grafica
    """
    data = _descargar_y_limpiar(tickers_list, start_date, end_date)
    mu, S, resultados, nube_grafica = optimize_portfolio_strategies(data, r_f)

    return data, mu, S, resultados, nube_grafica


def optimizacion_markowitz(
    tickers_list: Any, start_date: Any, end_date: Any, r_f: Any = 0.05
) -> Any:
    data, mu, S, res, nube = optimizacion_portafolios(tickers_list, start_date, end_date, r_f)
    return (data, mu, S, res["Máx. Sharpe"], res["Mín. Varianza"], nube)


def obtener_datos_subyacente(ticker_symbol: Any) -> Any:
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1y")

        if hist.empty or len(hist) < 20:
            return None, None

        spot_price = hist["Close"].iloc[-1]

        volatilidad_hist = log_return_volatility(hist["Close"])

        return spot_price, volatilidad_hist

    except Exception:
        return None, None


def evaluar_portafolio_personalizado(
    tickers_list: Any, dict_pesos: Any, start_date: Any, end_date: Any
) -> Any:
    data = _descargar_y_limpiar(tickers_list, start_date, end_date)

    rend_p, vol_p, pesos_array, columns = evaluate_custom_portfolio_from_prices(data, dict_pesos)

    return data, rend_p, vol_p, pesos_array, columns


def _descargar_y_limpiar(tickers_list: Any, start_date: Any, end_date: Any) -> Any:
    """
    Descarga y limpia datos de Yahoo Finance.
    Centralizado para evitar llamadas HTTP duplicadas entre métodos.
    """
    raw_data = yf.download(tickers_list, start=start_date, end=end_date, progress=False)

    if "Adj Close" in raw_data:
        data = raw_data["Adj Close"]
    elif "Close" in raw_data:
        data = raw_data["Close"]
    else:
        data = raw_data

    data = data.ffill().dropna()

    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers_list[0])

    return data
