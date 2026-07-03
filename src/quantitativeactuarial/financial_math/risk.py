"""Market-risk VaR and CVaR primitives."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def calcular_var_parametrico(rend_anual: float, vol_anual: float, valor_portafolio: float, nivel_confianza: float, dias_horizonte: int | float) -> tuple[float, float, float, float]:
    t = dias_horizonte / 252.0
    rend_periodo = rend_anual * t
    vol_periodo = vol_anual * np.sqrt(t)
    z_score = norm.ppf(nivel_confianza)
    var_monto = valor_portafolio * (z_score * vol_periodo - rend_periodo)
    return max(var_monto, 0), z_score, rend_periodo, vol_periodo


def calcular_var_cvar_montecarlo(rend_anual: float, vol_anual: float, valor_portafolio: float, nivel_confianza: float, dias_horizonte: int | float, simulaciones: int = 10000, seed: int = 42) -> tuple[float, float]:
    t = dias_horizonte / 252.0
    rend_periodo = rend_anual * t
    vol_periodo = vol_anual * np.sqrt(t)

    rng = np.random.default_rng(seed)
    e = rng.standard_normal(simulaciones)
    simulacion_retornos = rend_periodo + vol_periodo * e

    # MEJORA: np.percentile es más limpio y evita errores de índice manual
    alpha = 1.0 - nivel_confianza
    q_alpha = np.percentile(simulacion_retornos, alpha * 100)

    # CVaR: promedio de retornos en la cola de pérdidas
    cola = simulacion_retornos[simulacion_retornos <= q_alpha]
    cvar_alpha = cola.mean() if len(cola) > 0 else q_alpha

    return max(-q_alpha * valor_portafolio, 0), max(-cvar_alpha * valor_portafolio, 0)

__all__ = ["calcular_var_parametrico", "calcular_var_cvar_montecarlo"]
