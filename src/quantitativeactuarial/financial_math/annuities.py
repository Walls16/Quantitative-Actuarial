"""Annuities, perpetuities, gradients, and period solvers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import root_scalar

from .rates import tasa_nominal_m_a_nominal_p


def vf_anualidad_efectiva(R: float, i_m: float, n_m: float, anticipada: bool = False) -> float:
    """Valor futuro de rentas vencidas o anticipadas constantes a tasa im."""
    if i_m == 0:
        return R * n_m
    factor = ((1 + i_m) ** n_m - 1) / i_m
    if anticipada:
        factor *= 1 + i_m
    return R * factor


def vf_anualidad_nominal(R: float, i_nom: float, m: int | float, p: int | float, n: float) -> float:
    """
    Valor futuro de rentas vencidas constantes realizadas p veces al año,
    durante n años a una tasa nominal anual i(m).
    """
    i_p = tasa_nominal_m_a_nominal_p(i_nom, m, p)
    n_p = n * p
    return vf_anualidad_efectiva(R, i_p, n_p, anticipada=False)


def vf_anualidad_continua(R_anual: float, delta: float, n: float) -> float:
    """
    Valor futuro de rentas constantes realizadas de manera instantánea,
    durante n años a una fuerza de interés delta.
    R_anual es la tasa de flujo anual.
    """
    if delta == 0:
        return R_anual * n
    return R_anual * (np.exp(delta * n) - 1) / delta


def vp_anualidad_efectiva(R: float, i_m: float, n_m: float, anticipada: bool = False) -> float:
    if i_m == 0:
        return R * n_m
    factor = (1 - (1 + i_m) ** (-n_m)) / i_m
    if anticipada:
        factor *= 1 + i_m
    return R * factor


def vp_anualidad_nominal(R: float, i_nom: float, m: int | float, p: int | float, n: float) -> float:
    i_p = tasa_nominal_m_a_nominal_p(i_nom, m, p)
    n_p = n * p
    return vp_anualidad_efectiva(R, i_p, n_p, anticipada=False)


def vp_anualidad_continua(R_anual: float, delta: float, n: float) -> float:
    if delta == 0:
        return R_anual * n
    return R_anual * (1 - np.exp(-delta * n)) / delta


def vp_perpetuidad(R: float, i: float) -> float:
    if i == 0:
        return 0
    return R / i


def vf_gradiente_geo(R1: float, i_m: float, q_m: float, n_m: float) -> float:
    if i_m == q_m:
        return n_m * R1 * ((1 + i_m) ** (n_m - 1))
    numerador = ((1 + i_m) ** n_m) - ((1 + q_m) ** n_m)
    denominador = i_m - q_m
    return R1 * (numerador / denominador)


def vp_gradiente_geo(R1: float, i: float, q: float, n: float) -> float:
    if i == q:
        return R1 * n / (1 + i)
    return R1 * (1 - ((1 + q) / (1 + i)) ** n) / (i - q)


def nper_anualidad_vf(VF: float, R: float, i_m: float) -> float:
    """Despeje analítico de n para Valor Futuro de anualidad constante."""
    if i_m == 0:
        return VF / R
    val = (VF * i_m / R) + 1
    if val <= 0:
        return np.nan
    return np.log(val) / np.log(1 + i_m)


def nper_anualidad_vp(VP: float, R: float, i_m: float) -> float:
    """Despeje analítico de n para Valor Presente de anualidad constante."""
    if i_m == 0:
        return VP / R
    val = 1 - (VP * i_m / R)
    if val <= 0:
        return np.nan
    return -np.log(val) / np.log(1 + i_m)


def nper_gradiente_geo_vf(VF: float, R1: float, i_m: float, q_m: float) -> float:
    """Uso de solver numérico para n en gradiente geométrico (VF)."""
    if i_m == q_m:
        f = lambda n: n * R1 * ((1 + i_m) ** (n - 1)) - VF
    else:
        f = lambda n: R1 * (((1 + i_m) ** n - (1 + q_m) ** n) / (i_m - q_m)) - VF
    try:
        res = root_scalar(f, bracket=[0.0001, 2000], method="brentq")
        return res.root
    except Exception:
        return np.nan


def nper_gradiente_geo_vp(VP: float, R1: float, i_m: float, q_m: float) -> float:
    """Uso de solver numérico para n en gradiente geométrico (VP)."""
    if i_m == q_m:
        return VP * (1 + i_m) / R1
    else:
        f = lambda n: R1 * (1 - ((1 + q_m) / (1 + i_m)) ** n) / (i_m - q_m) - VP
    try:
        res = root_scalar(f, bracket=[0.0001, 2000], method="brentq")
        return res.root
    except Exception:
        return np.nan


def vp_gradiente_aritmetico(R1: float, G: float, i_m: float, n_m: float) -> float:
    """Valor Presente de una Renta Creciente/Decreciente Aritmética."""
    if i_m == 0:
        return R1 * n_m + G * n_m * (n_m - 1) / 2
    an = (1 - (1 + i_m) ** (-n_m)) / i_m
    return R1 * an + (G / i_m) * (an - n_m * (1 + i_m) ** (-n_m))


def vf_gradiente_aritmetico(R1: float, G: float, i_m: float, n_m: float) -> float:
    """Valor Futuro de una Renta Creciente/Decreciente Aritmética."""
    if i_m == 0:
        return R1 * n_m + G * n_m * (n_m - 1) / 2
    sn = ((1 + i_m) ** n_m - 1) / i_m
    return R1 * sn + (G / i_m) * (sn - n_m)


def nper_gradiente_arit_vf(VF: float, R1: float, G: float, i_m: float) -> float:
    f = lambda n: vf_gradiente_aritmetico(R1, G, i_m, n) - VF
    try:
        res = root_scalar(f, bracket=[0.0001, 2000], method="brentq")
        return res.root
    except Exception:
        return np.nan


def nper_gradiente_arit_vp(VP: float, R1: float, G: float, i_m: float) -> float:
    f = lambda n: vp_gradiente_aritmetico(R1, G, i_m, n) - VP
    try:
        res = root_scalar(f, bracket=[0.0001, 2000], method="brentq")
        return res.root
    except Exception:
        return np.nan


__all__ = [
    "vf_anualidad_efectiva",
    "vf_anualidad_nominal",
    "vf_anualidad_continua",
    "vp_anualidad_efectiva",
    "vp_anualidad_nominal",
    "vp_anualidad_continua",
    "vp_perpetuidad",
    "vf_gradiente_geo",
    "vp_gradiente_geo",
    "nper_anualidad_vf",
    "nper_anualidad_vp",
    "nper_gradiente_geo_vf",
    "nper_gradiente_geo_vp",
    "vp_gradiente_aritmetico",
    "vf_gradiente_aritmetico",
    "nper_gradiente_arit_vf",
    "nper_gradiente_arit_vp",
]
