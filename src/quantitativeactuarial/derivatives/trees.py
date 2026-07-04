"""Numerical lattice methods for option pricing."""

from __future__ import annotations

import numpy as np


def binomial_tree(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n: int,
    q: float = 0.0,
    tipo: str = "call",
    american: bool = False,
) -> tuple[float, tuple[np.ndarray, np.ndarray] | None]:
    tipo = tipo.lower().strip()
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if p < 0 or p > 1:
        return None, None

    S_tree = [np.zeros(i + 1) for i in range(n + 1)]
    V_tree = [np.zeros(i + 1) for i in range(n + 1)]

    for i in range(n + 1):
        for j in range(i + 1):
            S_tree[i][j] = S * (u ** (i - j)) * (d**j)

    for j in range(n + 1):
        if tipo == "call":
            V_tree[n][j] = max(0, S_tree[n][j] - K)
        else:
            V_tree[n][j] = max(0, K - S_tree[n][j])

    df = np.exp(-r * dt)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            val_hold = df * (p * V_tree[i + 1][j] + (1 - p) * V_tree[i + 1][j + 1])
            if american:
                if tipo == "call":
                    val_exercise = max(0, S_tree[i][j] - K)
                else:
                    val_exercise = max(0, K - S_tree[i][j])
                V_tree[i][j] = max(val_hold, val_exercise)
            else:
                V_tree[i][j] = val_hold

    return V_tree[0][0], (S_tree, V_tree)


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
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Wrapper de binomial_tree con firma usada en 10_Derivados_Vanilla.py."""
    tipo = "call" if es_call else "put"
    precio, arboles = binomial_tree(S, K, T, r, sigma, n, q, tipo, american)
    if arboles is None:
        return precio, None, None
    S_tree, V_tree = arboles
    return precio, S_tree, V_tree


__all__ = ["binomial_tree", "arbol_binomial_crr"]
