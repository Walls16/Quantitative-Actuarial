"""Option payoff-leg and strategy-profile utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calcular_payoff_leg(
    tipo: str, posicion: int, S_T: np.ndarray, K: float, prima: float
) -> np.ndarray:
    """Payoff de una pata individual para estrategias."""
    if tipo == "call":
        payoff = np.maximum(S_T - K, 0)
    elif tipo == "put":
        payoff = np.maximum(K - S_T, 0)
    else:
        payoff = np.zeros_like(S_T)
    return posicion * payoff - (posicion * prima)


def perfil_estrategia(S_spot: float, patas: list[dict], puntos: int = 500) -> pd.DataFrame:
    """Calcula el perfil de payoff de una estrategia de opciones.

    Para una malla de precios terminales ``S_T``, suma los payoffs de cada
    pata:

    ``payoff_call = max(S_T - K, 0)``
    ``payoff_put = max(K - S_T, 0)``

    La prima se resta para posiciones largas y se suma para posiciones
    cortas mediante ``posicion * payoff - posicion * prima``.

    Parameters
    ----------
    S_spot : float
        Precio spot usado como centro de la malla.
    patas : list[dict]
        Patas con claves ``tipo`` ("call" o "put"), ``posicion`` (+1/-1),
        ``K`` y ``prima``.
    puntos : int
        Número de observaciones en la malla.

    Returns
    -------
    pandas.DataFrame
        Columnas ``S_T``, una columna por pata y ``Payoff Neto``.
    """
    S_T = np.linspace(S_spot * 0.5, S_spot * 1.5, puntos)
    payoff_total = np.zeros_like(S_T)
    data = {"S_T": S_T}
    for pata in patas:
        pp = calcular_payoff_leg(pata["tipo"], pata["posicion"], S_T, pata["K"], pata["prima"])
        payoff_total += pp
        lbl = f"{'Long' if pata['posicion'] == 1 else 'Short'} {pata['tipo'].capitalize()} K={pata['K']}"
        data[lbl] = pp
    data["Payoff Neto"] = payoff_total
    return pd.DataFrame(data)


__all__ = ["calcular_payoff_leg", "perfil_estrategia"]
