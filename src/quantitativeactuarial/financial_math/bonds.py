"""Deterministic bond valuation, yield, duration, and convexity."""

from __future__ import annotations

import numpy as np
from scipy.optimize import root_scalar


def precio_bono(
    F: float, r_m: float, C: float, i_m: float, n: int
) -> tuple[float, float, float, float]:
    cupon_Fr = F * r_m

    if i_m == 0:
        vp_cupones = cupon_Fr * n
    else:
        vp_cupones = cupon_Fr * ((1 - (1 + i_m) ** (-n)) / i_m)

    vp_redencion = C * (1 + i_m) ** (-n)
    precio_total = vp_cupones + vp_redencion
    return precio_total, cupon_Fr, vp_cupones, vp_redencion


def tasa_rendimiento_bono(P: float, F: float, r_m: float, C: float, n: int) -> float:
    def f(i):
        if i == 0:
            precio_calc = (F * r_m * n) + C
        else:
            precio_calc = precio_bono(F, r_m, C, i, n)[0]
        return precio_calc - P

    try:
        res = root_scalar(f, bracket=[-0.99, 10.0], method="brentq")
        return res.root
    except Exception:
        return np.nan


def riesgo_bono(
    F: float, r_periodo: float, C: float, i_periodo: float, n_periodos: int, m: int | float
) -> tuple[float, float, float]:
    cupon = F * r_periodo
    precio = 0.0
    sum_mac = 0.0
    sum_conv = 0.0

    for t in range(1, int(n_periodos) + 1):
        cf = cupon if t < n_periodos else cupon + C
        vp_cf = cf / ((1 + i_periodo) ** t)
        precio += vp_cf
        sum_mac += t * vp_cf
        sum_conv += t * (t + 1) * vp_cf

    mac_duration_periodos = sum_mac / precio
    mac_duration_anios = mac_duration_periodos / m
    mod_duration_anios = mac_duration_anios / (1 + i_periodo)
    convexity_anios = sum_conv / (precio * (m**2) * ((1 + i_periodo) ** 2))

    return mac_duration_anios, mod_duration_anios, convexity_anios


__all__ = ["precio_bono", "tasa_rendimiento_bono", "riesgo_bono"]
