"""Interest-rate convention and reinvestment-table functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def tasa_nominal_a_efectiva(i_nom: float, m: int | float) -> float:
    if m <= 0:
        raise ValueError(f'`m` must be positive; received {m}.')
    return (1 + i_nom / m) ** m - 1

def tasa_efectiva_a_nominal(i_eff: float, m: int | float) -> float:
    if m <= 0:
        raise ValueError(f'`m` must be positive; received {m}.')
    return m * ((1 + i_eff) ** (1 / m) - 1)

def tasa_nominal_a_instantanea(i_nom: float, m: int | float) -> float:
    if m <= 0:
        raise ValueError(f'`m` must be positive; received {m}.')
    return m * np.log(1 + i_nom / m)

def tasa_instantanea_a_efectiva(delta: float) -> float:
    return np.exp(delta) - 1

def tasa_instantanea_a_nominal(delta: float, m: int | float) -> float:
    if m <= 0:
        raise ValueError(f'`m` must be positive; received {m}.')
    return m * (np.exp(delta / m) - 1)

def tasa_nominal_m_a_nominal_p(i_m: float, m: int | float, p: int | float) -> float:
    if m <= 0 or p <= 0:
        raise ValueError(f'`m` and `p` must be positive; received m={m}, p={p}.')
    tasa_efectiva_periodo = ((1 + i_m / m) ** (m / p)) - 1
    return tasa_efectiva_periodo * p

def generar_tabla_reinversion(C0: float, i_nom: float, n: float) -> pd.DataFrame:
    periodos = [
        ("Cada 4 años", 0.25), ("Cada 2 años", 0.5), ("Anual", 1),
        ("Semestral", 2), ("Trimestral", 4), ("Mensual", 12),
        ("Semanal", 52), ("Diaria", 365), ("Cada hora", 8760),
        ("Cada minuto", 525600), ("Cada segundo", 31536000)
    ]

    datos = []
    for nombre, m in periodos:
        monto = C0 * ((1 + i_nom / m) ** (m * n))
        rendimiento = (monto / C0) - 1
        datos.append({
            "Periodo de reinversión": nombre,
            "m = Veces al año": str(m),
            "Monto acumulado": monto,
            "Rendimiento Acumulado": rendimiento
        })

    monto_inst = C0 * np.exp(i_nom * n)
    rendimiento_inst = (monto_inst / C0) - 1
    datos.append({
        "Periodo de reinversión": "Instantánea",
        "m = Veces al año": "∞",
        "Monto acumulado": monto_inst,
        "Rendimiento Acumulado": rendimiento_inst
    })

    return pd.DataFrame(datos)


__all__ = [
    "tasa_nominal_a_efectiva",
    "tasa_efectiva_a_nominal",
    "tasa_nominal_a_instantanea",
    "tasa_instantanea_a_efectiva",
    "tasa_instantanea_a_nominal",
    "tasa_nominal_m_a_nominal_p",
    "generar_tabla_reinversion",
]
