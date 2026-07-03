"""Pure derivatives-pricing functions.

This module provides function-level access to the analytical and numerical
option-pricing routines implemented by :class:`FinancialMathEngine`.  Functions
return primitive values, dictionaries, tuples of arrays, or pandas-compatible
objects; rendering is left to application layers.
"""

from __future__ import annotations

from typing import Literal

from .tasas import FinancialMathEngine

_ENGINE = FinancialMathEngine()


def black_scholes(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    es_call: bool = True,
    q: float = 0.0,
) -> float:
    """Price a European option under Black-Scholes-Merton with continuous yield.

    The model uses
    ``d1 = (ln(S/K) + (r - q + sigma^2/2)T) / (sigma sqrt(T))`` and
    ``d2 = d1 - sigma sqrt(T)``.  Calls are valued as
    ``S e^{-qT} N(d1) - K e^{-rT} N(d2)`` and puts by put-call parity.
    """

    return float(_ENGINE.black_scholes(S, K, r, sigma, T, es_call, q))


def black_76(F0: float, K: float, r: float, sigma: float, T: float, es_call: bool = True) -> float:
    """Price an option on a futures/forward price using Black's 1976 model."""

    return float(_ENGINE.black_76(F0, K, r, sigma, T, es_call))


def arbol_binomial_crr(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n: int,
    es_call: bool = True,
    american: bool = False,
    q: float = 0.0,
) -> tuple[float, object, object]:
    """Price an option with the Cox-Ross-Rubinstein binomial tree.

    The tree uses ``u = exp(sigma sqrt(dt))``, ``d = 1/u`` and the
    risk-neutral probability ``p = (exp((r-q)dt)-d)/(u-d)``.
    """

    return _ENGINE.arbol_binomial_crr(S, K, r, sigma, T, n, es_call, american, q)


def calcular_griegas(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    es_call: bool = True,
    q: float = 0.0,
) -> dict[str, float]:
    """Return BSM delta, gamma, theta, vega, and rho for a vanilla option."""

    return _ENGINE.calcular_griegas(S, K, r, sigma, T, es_call, q)


def precio_forward(S0: float, r: float, T: float) -> float:
    """Forward price for a non-income-paying asset: ``F0 = S0 exp(rT)``."""

    return float(_ENGINE.precio_forward(S0, r, T))


def precio_forward_dividendo_continuo(S0: float, r: float, q: float, T: float) -> float:
    """Forward price with continuous yield ``q``: ``F0 = S0 exp((r-q)T)``."""

    return float(_ENGINE.precio_forward_dividendo_continuo(S0, r, q, T))


def precio_forward_dividendos_discretos(S0: float, r: float, T: float, I: float) -> float:
    """Forward price with present value of discrete income ``I``."""

    return float(_ENGINE.precio_forward_dividendos_discretos(S0, r, T, I))


def precio_forward_commodity(S0: float, r: float, u: float, T: float) -> float:
    """Commodity forward with continuous storage cost ``u``."""

    return float(_ENGINE.precio_forward_commodity(S0, r, u, T))


def precio_forward_divisa(S0: float, r_d: float, r_f: float, T: float) -> float:
    """Covered interest parity FX forward: ``F0 = S0 exp((rd-rf)T)``."""

    return float(_ENGINE.precio_forward_divisa(S0, r_d, r_f, T))


def valor_forward_en_vida(St: float, F0: float, r: float, q: float, tau: float) -> float:
    """Value of a live long forward position at time ``t``."""

    return float(_ENGINE.valor_forward_en_vida(St, F0, r, q, tau))


def fra(r1: float, r2: float, t1: float, t2: float, nocional: float, R_K: float) -> tuple[float, float]:
    """Return implied forward rate and FRA value for the fixed-rate receiver."""

    return _ENGINE.fra(r1, r2, t1, t2, nocional, R_K)


def opcion_binaria(
    tipo_pago: Literal["cash", "asset"],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    tipo: Literal["call", "put"] = "call",
    Q: float = 1.0,
) -> float:
    """Price a cash-or-nothing or asset-or-nothing binary option."""

    if tipo_pago == "cash":
        return float(_ENGINE.opciones_cash_or_nothing(S, K, Q, T, r, sigma, q, tipo))
    if tipo_pago == "asset":
        return float(_ENGINE.opciones_asset_or_nothing(S, K, T, r, sigma, q, tipo))
    raise ValueError("tipo_pago must be 'cash' or 'asset'")
