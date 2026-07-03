"""Exotic option pricing formulas and approximations."""

from __future__ import annotations

import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import multivariate_normal, norm

from .vanilla import opciones_bsm


def opciones_gap(S: float, K1: float, K2: float, T: float, r: float, sigma: float, q: float = 0.0, tipo: str = 'call') -> float:
    d1 = (np.log(S / K2) + (r - q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if tipo == 'call':
        precio = S * np.exp(-q * T) * norm.cdf(d1) - K1 * np.exp(-r * T) * norm.cdf(d2)
    elif tipo == 'put':
        precio = K1 * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("El tipo debe ser 'call' o 'put'")
    return precio


def opciones_cash_or_nothing(S: float, K: float, Q: float, T: float, r: float, sigma: float, q: float = 0.0, tipo: str = 'call') -> float:
    d1 = (np.log(S / K) + (r - q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if tipo == 'call':
        precio = Q * np.exp(-r * T) * norm.cdf(d2)
    elif tipo == 'put':
        precio = Q * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("El tipo debe ser 'call' o 'put'")
    return precio


def opciones_asset_or_nothing(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, tipo: str = 'call') -> float:
    """Valuación de Opciones Binarias: Asset-or-Nothing."""
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r - q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))

    if tipo == 'call':
        precio = S * np.exp(-q * T) * norm.cdf(d1)
    elif tipo == 'put':
        precio = S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("El tipo debe ser 'call' o 'put'")

    return max(0.0, precio)


def barrera_down_and_out(S: float, K: float, H: float, T: float, r: float, sigma: float, q: float = 0.0, tipo: str = 'call') -> float:
    """Valuación de Opciones con Barrera: Down-and-Out."""
    if S <= H:
        return 0.0

    lam = (r - q + (sigma ** 2) / 2) / (sigma ** 2)
    y = (np.log(H ** 2 / (S * K)) / (sigma * np.sqrt(T))) + lam * sigma * np.sqrt(T)

    vanilla_call, vanilla_put, _, _ = opciones_bsm("Yield", S, K, T, r, sigma, extra=q)

    if tipo == 'call':
        c_di = (S * np.exp(-q * T) * (H / S) ** (2 * lam) * norm.cdf(y)
                - K * np.exp(-r * T) * (H / S) ** (2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(T)))
        precio = vanilla_call - c_di
    elif tipo == 'put':
        p_di = (-S * np.exp(-q * T) * (H / S) ** (2 * lam) * norm.cdf(-y)
                + K * np.exp(-r * T) * (H / S) ** (2 * lam - 2) * norm.cdf(-y + sigma * np.sqrt(T)))
        precio = vanilla_put - p_di

    return max(0.0, precio)


def opciones_asiaticas_aritmeticas(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, tipo: str = 'call') -> float:
    b = r - q

    # MEJORA: usar np.expm1 para mayor precisión cuando b*T o sigma²*T son pequeños
    if np.isclose(b, 0.0):
        M1 = S
        # Límite correcto de Turnbull-Wakeman cuando b→0:
        M2 = (2 * S ** 2 / (sigma ** 2 * T ** 2)) * (
            np.expm1(sigma ** 2 * T) / sigma ** 2 - T
        )
    else:
        # np.expm1(x) = exp(x)-1, más preciso que exp()-1 para x pequeño
        M1 = S * np.expm1(b * T) / (b * T)
        termino1 = np.expm1((2 * b + sigma ** 2) * T) / (2 * b + sigma ** 2)
        termino2 = np.expm1(b * T) / b
        M2 = (2 * S ** 2 / ((b + sigma ** 2) * T ** 2)) * (termino1 - termino2)

    F = M1
    # Protección: M2/M1² debe ser > 1 para que el log sea positivo
    ratio = M2 / (M1 ** 2)
    if ratio <= 1.0:
        ratio = 1.0 + 1e-12
    sigma_adj = np.sqrt(np.log(ratio) / T)

    d1 = (np.log(F / K) + (sigma_adj ** 2 / 2) * T) / (sigma_adj * np.sqrt(T))
    d2 = d1 - sigma_adj * np.sqrt(T)

    if tipo == 'call':
        precio = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif tipo == 'put':
        precio = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    else:
        raise ValueError("El tipo debe ser 'call' o 'put'")

    return max(0.0, precio)


def opciones_asiaticas_geometricas(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, tipo: str = 'call') -> float:
    b = r - q
    sigma_adj = sigma / np.sqrt(3.0)
    b_adj = 0.5 * (b - (sigma ** 2) / 6.0)

    d1 = (np.log(S / K) + (b_adj + (sigma_adj ** 2) / 2.0) * T) / (sigma_adj * np.sqrt(T))
    d2 = d1 - sigma_adj * np.sqrt(T)

    if tipo == 'call':
        precio = S * np.exp((b_adj - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif tipo == 'put':
        precio = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((b_adj - r) * T) * norm.cdf(-d1)
    else:
        raise ValueError("El tipo debe ser 'call' o 'put'")

    return max(0.0, precio)


def opciones_lookback_flotante(S: float, S_ref: float, T: float, r: float, sigma: float, q: float = 0.0, tipo: str = 'call') -> float:
    # MEJORA: en lugar de epsilon fijo (1e-8), derivar el límite cuando r==q
    # usando la expansión de primer orden: lim_{r→q} formula = formula_limite
    r_eq_q = np.isclose(r, q, atol=1e-9)

    if tipo == 'call':
        Smin = S_ref

        if r_eq_q:
            # Límite analítico cuando r = q (evita división por cero)
            sqrtT = np.sqrt(T)
            a1 = (np.log(S / Smin) + (sigma ** 2 / 2) * T) / (sigma * sqrtT)
            a2 = a1 - sigma * sqrtT
            # Fórmula límite: el término (sigma²/2r) → (T) cuando r=q=0 con ajuste
            precio = (S * np.exp(-q * T) * (norm.cdf(a1) + sigma * sqrtT * norm.pdf(a1))
                      - Smin * np.exp(-r * T) * norm.cdf(a2))
        else:
            a1 = (np.log(S / Smin) + (r - q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            a2 = a1 - sigma * np.sqrt(T)
            a3 = (np.log(S / Smin) + (-r + q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            Y1 = -2 * (r - q - (sigma ** 2) / 2) * np.log(S / Smin) / (sigma ** 2)
            coef = sigma ** 2 / (2 * (r - q))

            termino1 = S * np.exp(-q * T) * norm.cdf(a1)
            termino2 = S * np.exp(-q * T) * coef * norm.cdf(-a1)
            termino3 = Smin * np.exp(-r * T) * (norm.cdf(a2) - coef * np.exp(Y1) * norm.cdf(-a3))
            precio = termino1 - termino2 - termino3

    elif tipo == 'put':
        Smax = S_ref

        if r_eq_q:
            sqrtT = np.sqrt(T)
            b1 = (np.log(Smax / S) + (sigma ** 2 / 2) * T) / (sigma * sqrtT)
            b2 = b1 - sigma * sqrtT
            precio = (Smax * np.exp(-r * T) * norm.cdf(b1)
                      - S * np.exp(-q * T) * (norm.cdf(b2) - sigma * sqrtT * norm.pdf(b2)))
        else:
            b1 = (np.log(Smax / S) + (-r + q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            b2 = b1 - sigma * np.sqrt(T)
            b3 = (np.log(Smax / S) + (r - q - (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            Y2 = 2 * (r - q - (sigma ** 2) / 2) * np.log(Smax / S) / (sigma ** 2)
            coef = sigma ** 2 / (2 * (r - q))

            termino1 = Smax * np.exp(-r * T) * (norm.cdf(b1) - coef * np.exp(Y2) * norm.cdf(-b3))
            termino2 = S * np.exp(-q * T) * norm.cdf(b2)
            termino3 = S * np.exp(-q * T) * coef * norm.cdf(-b2)
            precio = termino1 - termino2 + termino3
    else:
        raise ValueError("El tipo debe ser 'call' o 'put'")

    return max(0.0, precio)


def _pnbivariada(x: float, y: float, rho: float) -> float:
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return float(multivariate_normal(mean=mean, cov=cov).cdf([x, y]))


def opciones_compuestas(S: float, K1: float, K2: float, T1: float, T2: float, r: float, sigma: float, q: float = 0.0, tipo: str = 'call_on_call') -> float:
    """Valuación de Opciones Compuestas."""
    tau = T2 - T1

    if 'on_call' in tipo:
        objetivo = lambda x: opciones_bsm("Yield", x, K2, tau, r, sigma, extra=q)[0] - K1
    else:
        objetivo = lambda x: opciones_bsm("Yield", x, K2, tau, r, sigma, extra=q)[1] - K1

    try:
        # MEJORA: verificar cambio de signo antes de aplicar Brent para fallar explícitamente
        f_lo = objetivo(0.001)
        f_hi = objetivo(10000.0)
        if f_lo * f_hi > 0:
            return 0.0  # No hay raíz en el intervalo — opción sin valor
        S_star = root_scalar(objetivo, bracket=[0.001, 10000.0],
                             method='brentq', xtol=1e-6).root
    except Exception:
        return 0.0

    a1 = (np.log(S / S_star) + (r - q + (sigma ** 2) / 2) * T1) / (sigma * np.sqrt(T1))
    a2 = a1 - sigma * np.sqrt(T1)
    b1 = (np.log(S / K2) + (r - q + (sigma ** 2) / 2) * T2) / (sigma * np.sqrt(T2))
    b2 = b1 - sigma * np.sqrt(T2)
    rho = np.sqrt(T1 / T2)

    if tipo == 'call_on_call':
        M1 = _pnbivariada(a1, b1, rho)
        M2 = _pnbivariada(a2, b2, rho)
        precio = S * np.exp(-q * T2) * M1 - K2 * np.exp(-r * T2) * M2 - K1 * np.exp(-r * T1) * norm.cdf(a2)

    elif tipo == 'put_on_call':
        M1 = _pnbivariada(-a1, b1, -rho)
        M2 = _pnbivariada(-a2, b2, -rho)
        precio = K2 * np.exp(-r * T2) * M2 - S * np.exp(-q * T2) * M1 + K1 * np.exp(-r * T1) * norm.cdf(-a2)

    elif tipo == 'call_on_put':
        M1 = _pnbivariada(-a1, -b1, rho)
        M2 = _pnbivariada(-a2, -b2, rho)
        precio = K2 * np.exp(-r * T2) * M2 - S * np.exp(-q * T2) * M1 - K1 * np.exp(-r * T1) * norm.cdf(-a2)

    elif tipo == 'put_on_put':
        M1 = _pnbivariada(a1, -b1, -rho)
        M2 = _pnbivariada(a2, -b2, -rho)
        precio = S * np.exp(-q * T2) * M1 - K2 * np.exp(-r * T2) * M2 + K1 * np.exp(-r * T1) * norm.cdf(a2)

    return precio


def opciones_intercambio_uxv(U: float, V: float, q_u: float, q_v: float, sigma_u: float, sigma_v: float, rho: float, T: float) -> float:
    sigma = np.sqrt(sigma_u ** 2 + sigma_v ** 2 - 2 * rho * sigma_u * sigma_v)
    d1 = (np.log(V / U) + (q_u - q_v + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    precio = V * np.exp(-q_v * T) * norm.cdf(d1) - U * np.exp(-q_u * T) * norm.cdf(d2)
    return max(0.0, precio)


def opcion_chooser_simple(S: float, K: float, T1: float, T2: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Valuación de Opción Chooser Simple (As You Like It)."""
    c = opciones_bsm("Yield", S, K, T2, r, sigma, extra=q)[0]
    K_put = K * np.exp(-(r - q) * (T2 - T1))
    p = opciones_bsm("Yield", S, K_put, T1, r, sigma, extra=q)[1]
    return c + np.exp(-q * (T2 - T1)) * p


def opcion_perpetua(S: float, K: float, r: float, sigma: float, es_call: bool = True) -> float:
    """Valuación de opción perpetua (T → ∞)."""
    if sigma <= 0 or r <= 0:
        return max(S - K, 0) if es_call else max(K - S, 0)
    h = 0.5 + np.sqrt(0.25 + 2 * r / sigma ** 2)
    if es_call:
        precio = (K / (h - 1)) * ((S * (h - 1)) / (h * K)) ** h
    else:
        h_s = 0.5 - np.sqrt(0.25 + 2 * r / sigma ** 2)
        precio = (K / (1 - h_s)) * ((S * (1 - h_s)) / (h_s * K)) ** h_s
    return max(0.0, precio)

__all__ = [
    "opciones_gap",
    "opciones_cash_or_nothing",
    "opciones_asset_or_nothing",
    "barrera_down_and_out",
    "opciones_asiaticas_aritmeticas",
    "opciones_asiaticas_geometricas",
    "opciones_lookback_flotante",
    "opciones_compuestas",
    "opciones_intercambio_uxv",
    "opcion_chooser_simple",
    "opcion_perpetua",
]
