"""Forward, futures-style carry, FX forward, and FRA pricing."""

from __future__ import annotations

import numpy as np


def _is_continuous(capitalizacion: str) -> bool:
    return capitalizacion.strip().lower().startswith("continua")


def _growth_factor(rate: float, maturity: float, capitalizacion: str) -> float:
    if _is_continuous(capitalizacion):
        return float(np.exp(rate * maturity))
    return float((1 + rate) ** maturity)


def _discount_factor(rate: float, maturity: float, capitalizacion: str) -> float:
    return 1.0 / _growth_factor(rate, maturity, capitalizacion)


def forward_price(
    spot: float,
    rate: float,
    maturity: float,
    carry: float = 0.0,
    capitalizacion: str = "Continua",
) -> float:
    """
    Price a forward with a financing rate and income yield.

    For continuous compounding, the model is
    ``F = S exp((r - y)T)``. For discrete effective rates, the equivalent
    no-arbitrage relation is ``F = S (1 + r)^T / (1 + y)^T``.
    """
    if _is_continuous(capitalizacion):
        return float(spot * np.exp((rate - carry) * maturity))
    return float(spot * _growth_factor(rate, maturity, capitalizacion) / _growth_factor(carry, maturity, capitalizacion))


def forward_calculo(S: float, r: float, delta: float, T: float, capitalizacion: str = "Continua") -> float:
    return forward_price(S, r, T, carry=delta, capitalizacion=capitalizacion)


def valor_forward_calculo(S: float, K: float, r: float, delta: float, T: float, posicion: str = "Larga", capitalizacion: str = "Continua") -> float:
    val_largo = valor_forward_en_vida(S, K, r, delta, T, capitalizacion=capitalizacion)
    return val_largo if posicion == "Larga" else -val_largo


def precio_forward(S0: float, r: float, T: float, capitalizacion: str = "Continua") -> float:
    """Precio teórico de un forward sobre activo sin rendimientos."""
    return forward_price(S0, r, T, capitalizacion=capitalizacion)


def precio_forward_dividendo_continuo(S0: float, r: float, q: float, T: float, capitalizacion: str = "Continua") -> float:
    """Precio forward con dividendo continuo o tasa extranjera q."""
    return forward_price(S0, r, T, carry=q, capitalizacion=capitalizacion)


def precio_forward_dividendos_discretos(S0: float, r: float, T: float, I: float, capitalizacion: str = "Continua") -> float:
    """Precio forward descontando VP de dividendos discretos I."""
    return float((S0 - I) * _growth_factor(r, T, capitalizacion))


def precio_forward_commodity(S0: float, r: float, u: float, T: float, capitalizacion: str = "Continua") -> float:
    """Precio forward de commodity con costo de almacenamiento continuo u."""
    if _is_continuous(capitalizacion):
        return float(S0 * np.exp((r + u) * T))
    return float(S0 * _growth_factor(r, T, capitalizacion) * _growth_factor(u, T, capitalizacion))


def precio_forward_divisa(S0: float, r_d: float, r_f: float, T: float, capitalizacion: str = "Continua") -> float:
    """Tipo de cambio forward (Paridad Cubierta de Tasas de Interés)."""
    return forward_price(S0, r_d, T, carry=r_f, capitalizacion=capitalizacion)


def valor_forward_en_vida(St: float, F0: float, r: float, q: float, tau: float, capitalizacion: str = "Continua") -> float:
    """Valor de mercado de un forward largo en t < T. tau = T - t."""
    return float(St * _discount_factor(q, tau, capitalizacion) - F0 * _discount_factor(r, tau, capitalizacion))


def fra(r1: float, r2: float, t1: float, t2: float, nocional: float, R_K: float, capitalizacion: str = "Continua") -> tuple[float, float]:
    """
    Forward Rate Agreement.
    Devuelve (tasa_forward_implicita, valor_fra).
    Convención: receptor de tasa fija (R_K) paga flotante (R_F).
    """
    tau = t2 - t1
    if _is_continuous(capitalizacion):
        R_F = (r2 * t2 - r1 * t1) / tau
    else:
        R_F = (_growth_factor(r2, t2, capitalizacion) / _growth_factor(r1, t1, capitalizacion)) ** (1 / tau) - 1
    valor = nocional * (R_F - R_K) * tau * _discount_factor(r2, t2, capitalizacion)
    return float(R_F), float(valor)

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
