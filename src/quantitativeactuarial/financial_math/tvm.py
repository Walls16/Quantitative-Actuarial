"""Time-value-of-money primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd


def valor_futuro(C0: float, i: float, n: float) -> float:
    return C0 * (1 + i) ** n


def valor_futuro_continuo(C0: float, delta: float, n: float) -> float:
    return C0 * np.exp(delta * n)


def valor_presente(Cn: float, i: float, n: float) -> float:
    return Cn / (1 + i) ** n


def valor_presente_continuo(Cn: float, delta: float, n: float) -> float:
    return Cn * np.exp(-delta * n)


def numero_periodos(C0: float, Cn: float, i: float) -> float:
    if C0 == 0 or i <= 0:
        return 0
    return np.log(Cn / C0) / np.log(1 + i)


def tasa_rendimiento(C0: float, Cn: float, n: float) -> float:
    if C0 == 0 or n == 0:
        return 0
    return (Cn / C0) ** (1 / n) - 1


def desglosar_periodos(n: float) -> pd.DataFrame:
    anios = int(n)
    frac_anios = n - anios

    meses_raw = frac_anios * 12
    meses = int(meses_raw)
    frac_meses = meses_raw - meses

    dias_raw = frac_meses * (365 / 12)
    dias = int(dias_raw)
    frac_dias = dias_raw - dias

    horas_raw = frac_dias * 24
    horas = int(horas_raw)
    frac_horas = horas_raw - horas

    min_raw = frac_horas * 60
    minutos = int(min_raw)
    frac_min = min_raw - minutos

    seg_raw = frac_min * 60
    segundos = int(seg_raw)

    return pd.DataFrame([{
        "Años": anios,
        "Meses": meses,
        "Días": dias,
        "Horas": horas,
        "Minutos": minutos,
        "Segundos": segundos
    }])

__all__ = [
    "valor_futuro",
    "valor_futuro_continuo",
    "valor_presente",
    "valor_presente_continuo",
    "numero_periodos",
    "tasa_rendimiento",
    "desglosar_periodos",
]
