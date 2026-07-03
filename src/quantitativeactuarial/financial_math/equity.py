"""Equity valuation primitives."""

from __future__ import annotations


def valuacion_gordon_shapiro(D1: float, k: float, g: float) -> float:
    if k <= g:
        return None
    return D1 / (k - g)


def rendimiento_requerido_accion(D1: float, P0: float, g: float) -> float:
    if P0 <= 0:
        return None
    return (D1 / P0) + g


def valuacion_multiplos(metrica_valor: float, multiplo_objetivo: float) -> float:
    return metrica_valor * multiplo_objetivo

__all__ = [
    "valuacion_gordon_shapiro",
    "rendimiento_requerido_accion",
    "valuacion_multiplos",
]
