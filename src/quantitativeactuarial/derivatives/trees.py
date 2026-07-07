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
    option_type: str = "call",
    american: bool = False,
) -> tuple[float, tuple[np.ndarray, np.ndarray] | None]:
    option_type = option_type.lower().strip()
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
        if option_type == "call":
            V_tree[n][j] = max(0, S_tree[n][j] - K)
        else:
            V_tree[n][j] = max(0, K - S_tree[n][j])

    df = np.exp(-r * dt)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            val_hold = df * (p * V_tree[i + 1][j] + (1 - p) * V_tree[i + 1][j + 1])
            if american:
                if option_type == "call":
                    val_exercise = max(0, S_tree[i][j] - K)
                else:
                    val_exercise = max(0, K - S_tree[i][j])
                V_tree[i][j] = max(val_hold, val_exercise)
            else:
                V_tree[i][j] = val_hold

    return V_tree[0][0], (S_tree, V_tree)


def crr_binomial_tree(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n: int,
    is_call: bool = True,
    american: bool = False,
    q: float = 0.0,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Return a CRR binomial price and the stock/value trees."""
    option_type = "call" if is_call else "put"
    price, trees = binomial_tree(S, K, T, r, sigma, n, q, option_type, american)
    if trees is None:
        return price, None, None
    stock_tree, value_tree = trees
    return price, stock_tree, value_tree


__all__ = ["binomial_tree", "crr_binomial_tree"]
