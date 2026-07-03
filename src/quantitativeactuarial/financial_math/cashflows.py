"""Discounting helpers for dividend and irregular cash-flow streams."""

from __future__ import annotations

import numpy as np


def calcular_vp_dividendos(monto_div: float, m_pagos: int, r: float, T_total: float, capitalizacion: str = "Continua") -> float:
    vp_total = 0
    dt = 1 / m_pagos
    num_pagos = int(T_total * m_pagos)

    for k in range(1, num_pagos + 1):
        t_pago = k * dt
        if capitalizacion == "Continua":
            vp_total += monto_div * np.exp(-r * t_pago)
        else:
            vp_total += monto_div / ((1 + r) ** t_pago)
    return vp_total


def calcular_vp_flujos_irregulares(montos: list[float] | np.ndarray, tiempos_anios: list[float] | np.ndarray, r: float, capitalizacion: str = "Continua") -> float:
    vp_total = 0.0
    for monto, t in zip(montos, tiempos_anios):
        if capitalizacion == "Continua":
            vp_total += monto * np.exp(-r * t)
        else:
            vp_total += monto / ((1 + r) ** t)
    return vp_total

__all__ = ["calcular_vp_dividendos", "calcular_vp_flujos_irregulares"]
