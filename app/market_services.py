"""Application-only market-data services.

These routines intentionally live outside the core package because they perform
network I/O through Yahoo Finance and therefore are not mathematically pure.
"""



from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize as opt
import yfinance as yf
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

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
    n_activos = len(data.columns)

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
    w_eq = np.ones(n_activos) / n_activos
    ret_eq  = float(w_eq @ mu.values)
    vol_eq  = float(np.sqrt(w_eq @ S.values @ w_eq))
    sharpe_eq = (ret_eq - r_f) / vol_eq
    pesos_eq = {t: float(w) for t, w in zip(data.columns, w_eq)}

    # ── 4. Paridad de Riesgo ──────────────────────────────────────────────
    def _risk_parity_objective(w, cov):
        w = np.array(w)
        sigma_p = np.sqrt(w @ cov @ w)
        mrc = cov @ w / sigma_p          # contribución marginal al riesgo
        rc  = w * mrc                    # contribución absoluta
        target = sigma_p / n_activos     # contribución objetivo (igual)
        return np.sum((rc - target) ** 2)

    w0 = np.ones(n_activos) / n_activos
    bounds_rp = [(1e-4, 1.0)] * n_activos
    cons_rp   = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    res_rp = opt.minimize(
        _risk_parity_objective, w0,
        args=(S.values,),
        method="SLSQP",
        bounds=bounds_rp,
        constraints=cons_rp,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    w_rp    = np.abs(res_rp.x) / np.sum(np.abs(res_rp.x))
    ret_rp  = float(w_rp @ mu.values)
    vol_rp  = float(np.sqrt(w_rp @ S.values @ w_rp))
    sharpe_rp = (ret_rp - r_f) / vol_rp
    pesos_rp = {t: float(w) for t, w in zip(data.columns, w_rp)}

    # ── 5. MVSK (Media-Varianza-Asimetría-Curtosis) ───────────────────────
    # Incorpora momentos de orden 3 y 4 para capturar colas y asimetrías.
    # Función objetivo: maximizar E[R] - λ₂σ² + λ₃skew - λ₄kurt
    # Pesos λ estándar usados en literatura (Harvey et al. 2010).
    lambda2, lambda3, lambda4 = 1.0, 0.5, 0.5

    daily_ret = retornos_diarios.values  # shape (T, N)
    mu_d = daily_ret.mean(axis=0)

    def _mvsk_neg_utility(w):
        w = np.array(w)
        p_ret   = daily_ret @ w
        mean_p  = float(np.mean(p_ret))
        var_p   = float(np.var(p_ret))
        std_p   = float(np.std(p_ret)) + 1e-10
        skew_p  = float(np.mean(((p_ret - mean_p) / std_p) ** 3))
        kurt_p  = float(np.mean(((p_ret - mean_p) / std_p) ** 4))
        utility = mean_p - lambda2 * var_p + lambda3 * skew_p - lambda4 * kurt_p
        return -utility  # negativo para minimizar

    res_mvsk = opt.minimize(
        _mvsk_neg_utility, w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_activos,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    w_mv    = np.abs(res_mvsk.x) / np.sum(np.abs(res_mvsk.x))
    ret_mv  = float(w_mv @ mu.values)
    vol_mv  = float(np.sqrt(w_mv @ S.values @ w_mv))
    sharpe_mv = (ret_mv - r_f) / vol_mv
    pesos_mvsk = {t: float(w) for t, w in zip(data.columns, w_mv)}

    # ── Nube Monte Carlo ──────────────────────────────────────────────────
    n_sim   = 2500
    pesos_rand = np.random.dirichlet(np.ones(n_activos), size=n_sim)
    ret_sim    = pesos_rand @ mu.values
    vol_sim    = np.sqrt(np.einsum('ij,jk,ik->i', pesos_rand, S.values, pesos_rand))
    sharpe_sim = (ret_sim - r_f) / vol_sim
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

    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)

    pesos_array = np.array([dict_pesos.get(c, 0) for c in data.columns])
    suma = pesos_array.sum()
    if suma > 0:
        pesos_array = pesos_array / suma

    rend_p = np.dot(pesos_array, mu)
    vol_p = np.sqrt(np.dot(pesos_array.T, np.dot(S, pesos_array)))

    return data, rend_p, vol_p, pesos_array, data.columns



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
