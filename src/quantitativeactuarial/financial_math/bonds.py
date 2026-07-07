"""Deterministic bond valuation, yield, duration, and convexity."""

from __future__ import annotations

import numpy as np
from scipy.optimize import root_scalar


def bond_price(
    face_value: float,
    coupon_rate_per_period: float,
    redemption_value: float,
    yield_per_period: float,
    periods: int,
) -> tuple[float, float, float, float]:
    """Return total bond price, coupon, coupon present value, and redemption present value."""
    F = face_value
    r_m = coupon_rate_per_period
    C = redemption_value
    i_m = yield_per_period
    n = periods
    cupon_Fr = F * r_m

    if i_m == 0:
        vp_cupones = cupon_Fr * n
    else:
        vp_cupones = cupon_Fr * ((1 - (1 + i_m) ** (-n)) / i_m)

    vp_redencion = C * (1 + i_m) ** (-n)
    precio_total = vp_cupones + vp_redencion
    return precio_total, cupon_Fr, vp_cupones, vp_redencion


def bond_yield(
    price: float,
    face_value: float,
    coupon_rate_per_period: float,
    redemption_value: float,
    periods: int,
) -> float:
    """Solve the yield per period implied by a bond price."""
    P = price
    F = face_value
    r_m = coupon_rate_per_period
    C = redemption_value
    n = periods

    def f(i):
        if i == 0:
            precio_calc = (F * r_m * n) + C
        else:
            precio_calc = bond_price(F, r_m, C, i, n)[0]
        return precio_calc - P

    try:
        res = root_scalar(f, bracket=[-0.99, 10.0], method="brentq")
        return res.root
    except Exception:
        return np.nan


def bond_risk(
    face_value: float,
    coupon_rate_per_period: float,
    redemption_value: float,
    yield_per_period: float,
    periods: int,
    payments_per_year: int | float,
) -> tuple[float, float, float]:
    """Return Macaulay duration, modified duration, and convexity in years."""
    F = face_value
    r_periodo = coupon_rate_per_period
    C = redemption_value
    i_periodo = yield_per_period
    n_periodos = periods
    m = payments_per_year
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


__all__ = ["bond_price", "bond_yield", "bond_risk"]
