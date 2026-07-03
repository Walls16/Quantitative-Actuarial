"""Forward, futures-style carry, FX forward, and FRA pricing."""

from __future__ import annotations

import numpy as np


def forward_price(spot: float, rate: float, maturity: float, carry: float = 0.0) -> float:
    """Canonical continuous-carry forward price: F = S exp((r - carry)T)."""
    return spot * np.exp((rate - carry) * maturity)


def forward_calculo(S: float, r: float, delta: float, T: float, capitalizacion: str = "Continua") -> float:
    if capitalizacion == "Continua":
        return S * np.exp((r - delta) * T)
    else:
        return S * ((1 + r) ** T) / ((1 + delta) ** T)


def valor_forward_calculo(S: float, K: float, r: float, delta: float, T: float, posicion: str = "Larga", capitalizacion: str = "Continua") -> float:
    if capitalizacion == "Continua":
        val_largo = (S * np.exp(-delta * T)) - (K * np.exp(-r * T))
    else:
        val_largo = (S / (1 + delta) ** T) - (K / (1 + r) ** T)
    return val_largo if posicion == "Larga" else -val_largo


def precio_forward(S0: float, r: float, T: float) -> float:
    """Precio teórico de un forward sobre activo sin rendimientos."""
    return forward_price(S0, r, T)


def precio_forward_dividendo_continuo(S0: float, r: float, q: float, T: float) -> float:
    """Precio forward con dividendo continuo o tasa extranjera q."""
    return forward_price(S0, r, T, carry=q)


def precio_forward_dividendos_discretos(S0: float, r: float, T: float, I: float) -> float:
    """Precio forward descontando VP de dividendos discretos I."""
    return (S0 - I) * np.exp(r * T)


def precio_forward_commodity(S0: float, r: float, u: float, T: float) -> float:
    """Precio forward de commodity con costo de almacenamiento continuo u."""
    return forward_price(S0, r, T, carry=-u)


def precio_forward_divisa(S0: float, r_d: float, r_f: float, T: float) -> float:
    """Tipo de cambio forward (Paridad Cubierta de Tasas de Interés)."""
    return forward_price(S0, r_d, T, carry=r_f)


def valor_forward_en_vida(St: float, F0: float, r: float, q: float, tau: float) -> float:
    """Valor de mercado de un forward largo en t < T. tau = T - t."""
    return St * np.exp(-q * tau) - F0 * np.exp(-r * tau)


def fra(r1: float, r2: float, t1: float, t2: float, nocional: float, R_K: float) -> tuple[float, float]:
    """
    Forward Rate Agreement.
    Devuelve (tasa_forward_implicita, valor_fra).
    Convención: receptor de tasa fija (R_K) paga flotante (R_F).
    """
    tau = t2 - t1
    R_F = (r2 * t2 - r1 * t1) / tau
    valor = nocional * (R_F - R_K) * tau * np.exp(-r2 * t2)
    return R_F, valor

__all__ = [
    "forward_price",
    "forward_calculo",
    "valor_forward_calculo",
    "precio_forward",
    "precio_forward_dividendo_continuo",
    "precio_forward_dividendos_discretos",
    "precio_forward_commodity",
    "precio_forward_divisa",
    "valor_forward_en_vida",
    "fra",
]
