"""Application-only market-data services.

These routines intentionally live outside the core package because they perform
network I/O through Yahoo Finance and therefore are not mathematically pure.
"""



from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from quantitativeactuarial.financial_math import (
    equal_weight_portfolio,
    evaluate_custom_portfolio,
    monte_carlo_portfolio_cloud,
    mvsk_portfolio,
    risk_parity_portfolio,
)

def optimizacion_portafolios(tickers_list: Any, start_date: Any, end_date: Any, r_f: Any = 0.05) -> Any:
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
    mu   = expected_returns.mean_historical_return(data)
    S    = risk_models.sample_cov(data)

    retornos_diarios = data.pct_change().dropna()

    # ── 1. Máximo Sharpe ─────────────────────────────────────────────────
    ef_s = EfficientFrontier(mu, S)
    ef_s.max_sharpe(risk_free_rate=r_f)
    pesos_sharpe = ef_s.clean_weights()
    ret_s, vol_s, sharpe_s = ef_s.portfolio_performance(verbose=False, risk_free_rate=r_f)

    # ── 2. Mínima Varianza Global ─────────────────────────────────────────
    ef_m = EfficientFrontier(mu, S)
    ef_m.min_volatility()
    pesos_min = ef_m.clean_weights()
    ret_m, vol_m, sharpe_m = ef_m.portfolio_performance(verbose=False, risk_free_rate=r_f)

    # ── 3. 1/N Equiponderación ────────────────────────────────────────────
    ret_eq, vol_eq, sharpe_eq, pesos_eq = equal_weight_portfolio(
        list(data.columns), mu.values, S.values, r_f
    )

    # ── 4. Paridad de Riesgo ──────────────────────────────────────────────
    ret_rp, vol_rp, sharpe_rp, pesos_rp = risk_parity_portfolio(
        list(data.columns), mu.values, S.values, r_f
    )

    # ── 5. MVSK (Media-Varianza-Asimetría-Curtosis) ───────────────────────
    ret_mv, vol_mv, sharpe_mv, pesos_mvsk = mvsk_portfolio(
        list(data.columns), mu.values, S.values, retornos_diarios.values, r_f
    )

    # ── Nube Monte Carlo ──────────────────────────────────────────────────
    _, ret_sim, vol_sim, sharpe_sim = monte_carlo_portfolio_cloud(mu.values, S.values, r_f, n_simulations=2500)
    nube_grafica = (ret_sim, vol_sim, sharpe_sim)

    # ── Empaquetar resultados ─────────────────────────────────────────────
    resultados = {
        "Máx. Sharpe":       (ret_s,  vol_s,  sharpe_s,  pesos_sharpe),
        "Mín. Varianza":     (ret_m,  vol_m,  sharpe_m,  pesos_min),
        "1/N Equiponderado": (ret_eq, vol_eq, sharpe_eq, pesos_eq),
        "Paridad de Riesgo": (ret_rp, vol_rp, sharpe_rp, pesos_rp),
        "MVSK":              (ret_mv, vol_mv, sharpe_mv, pesos_mvsk),
    }

    return data, mu, S, resultados, nube_grafica



def optimizacion_markowitz(tickers_list: Any, start_date: Any, end_date: Any, r_f: Any = 0.05) -> Any:
    data, mu, S, res, nube = optimizacion_portafolios(tickers_list, start_date, end_date, r_f)
    return (data, mu, S,
            res["Máx. Sharpe"],
            res["Mín. Varianza"],
            nube)



def obtener_datos_subyacente(ticker_symbol: Any) -> Any:
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1y")

        if hist.empty or len(hist) < 20:
            return None, None

        spot_price = hist['Close'].iloc[-1]

        # MEJORA: usar ddof=1 (desviación muestral) y limpiar infinitos por precios cero
        retornos_log = np.log(hist['Close'] / hist['Close'].shift(1))
        retornos_log = retornos_log.replace([np.inf, -np.inf], np.nan).dropna()
        volatilidad_hist = retornos_log.std(ddof=1) * np.sqrt(252)

        return spot_price, volatilidad_hist

    except Exception:
        return None, None



def evaluar_portafolio_personalizado(tickers_list: Any, dict_pesos: Any, start_date: Any, end_date: Any) -> Any:
    data = _descargar_y_limpiar(tickers_list, start_date, end_date)

    rend_p, vol_p, pesos_array, columns = evaluate_custom_portfolio(
        data, dict_pesos, expected_returns.mean_historical_return, risk_models.sample_cov
    )

    return data, rend_p, vol_p, pesos_array, columns



def _descargar_y_limpiar(tickers_list: Any, start_date: Any, end_date: Any) -> Any:
    """
    Descarga y limpia datos de Yahoo Finance.
    Centralizado para evitar llamadas HTTP duplicadas entre métodos.
    """
    raw_data = yf.download(tickers_list, start=start_date, end=end_date, progress=False)

    if 'Adj Close' in raw_data:
        data = raw_data['Adj Close']
    elif 'Close' in raw_data:
        data = raw_data['Close']
    else:
        data = raw_data

    data = data.ffill().dropna()

    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers_list[0])

    return data
