"""Gaussian-copula dependence modeling for credit portfolios."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def thresholds_per_bond(rating_idx: int, trans_mat: np.ndarray) -> np.ndarray:
    """
    Calcula los umbrales N^-1(P_acumulada) para un bono dado.
    Usados en la cópula gaussiana y para mostrar al usuario.

    Devuelve np.ndarray (18,) — último elemento = +inf
    """
    cum_p = np.cumsum(trans_mat[rating_idx])
    with np.errstate(all='ignore'):
        thresh = norm.ppf(np.clip(cum_p, 1e-15, 1.0 - 1e-15))
    thresh[-1] = np.inf
    return thresh


def gaussian_copula_simulation(bonds_data: list, trans_mat: np.ndarray,
                                corr_mat: np.ndarray,
                                n_sims: int = 50_000,
                                seed: int = 42) -> np.ndarray:
    """
    Cópula Gaussiana Monte Carlo para el caso correlacionado.

    Parámetros
    ----------
    bonds_data : list de dicts con 'rating_idx' y 'values'
    trans_mat  : (18, 18) matriz de transición normalizada
    corr_mat   : (n, n) matriz de correlación entre activos (proxy accionaria)
    n_sims     : número de simulaciones
    seed       : semilla aleatoria determinística por defecto

    Devuelve
    --------
    np.ndarray (n_sims,) — valores simulados del portafolio
    """
    rng = np.random.default_rng(seed)
    n   = len(bonds_data)

    # Umbrales N^-1 para cada bono: centralizados para mantener una sola
    # convención numérica entre simulación y función pública.
    thresholds = [
        thresholds_per_bond(b['rating_idx'], trans_mat)
        for b in bonds_data
    ]

    # Descomposición de Cholesky (con jitter si no es definida positiva)
    C = corr_mat.copy().astype(float)
    np.fill_diagonal(C, 1.0)
    jitter = 0.0
    while True:
        try:
            L = np.linalg.cholesky(C + np.eye(n) * jitter)
            break
        except np.linalg.LinAlgError:
            jitter = jitter * 2 + 1e-8

    # Simular variables normales correlacionadas
    Z = rng.standard_normal((n_sims, n))
    X = Z @ L.T   # (n_sims, n)

    # Mapear a calificaciones y calcular valor del portafolio
    port_vals = np.zeros(n_sims)
    for b_idx, (b, thresh) in enumerate(zip(bonds_data, thresholds)):
        n_vals = len(b['values'])
        r_sim = np.searchsorted(thresh, X[:, b_idx]).clip(0, n_vals - 1)
        port_vals += np.array(b['values'], dtype=float)[r_sim]

    return port_vals

__all__ = ["thresholds_per_bond", "gaussian_copula_simulation"]
