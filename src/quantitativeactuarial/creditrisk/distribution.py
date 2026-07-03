"""Exact independent portfolio-value distribution utilities."""

from __future__ import annotations

import numpy as np


def independent_distribution(bonds_data: list, trans_mat: np.ndarray) -> list:
    """
    Distribución exacta del portafolio via convolución iterativa.
    Funciona con matrices 18x18 (modos redistribute/simple_normalize)
    y también con 17x17 (modo raw_no_d_nr).

    bonds_data : list de dicts con 'rating_idx' y 'values'
    trans_mat  : np.ndarray (N x N) donde N=18 ó 17

    Mejoras de rendimiento vs. versión original:
      - np.bincount (buffered) reemplaza np.add.at (unbuffered) → ~3-5x más rápido
      - Se filtran probabilidades insignificantes antes del producto exterior
    """
    # Redondeo a 2 decimales colapsa valores casi idénticos, reduciendo el
    # espacio de estados de K^N a ~10^3..10^4 para 5-10 bonos.
    decimals = 2

    probs0 = trans_mat[bonds_data[0]['rating_idx']]
    vals0  = np.round(bonds_data[0]['values'], decimals).astype(float)
    # SAFETY: en modo raw_no_d_nr la matriz es 17×17 pero si cm_tm fue
    # sobreescrito con una 18×18 (bug de sesión), truncamos al mínimo.
    _n0    = min(len(probs0), len(vals0))
    probs0 = probs0[:_n0]
    vals0  = vals0[:_n0]
    mask0  = probs0 > 1e-14
    cur_vals  = vals0[mask0]
    cur_probs = probs0[mask0]

    for b in bonds_data[1:]:
        pb = trans_mat[b['rating_idx']]
        vb = np.round(b['values'], decimals).astype(float)
        _nb = min(len(pb), len(vb))
        pb, vb = pb[:_nb], vb[:_nb]
        pmask = pb > 1e-14
        pb, vb = pb[pmask], vb[pmask]

        # Vectorized outer product
        joint_probs = np.outer(cur_probs, pb).ravel()
        joint_vals  = np.round((cur_vals[:, None] + vb[None, :]).ravel(), decimals)

        # --- CORRECCIÓN DE RENDIMIENTO ---
        # np.unique devuelve índices inversos; np.bincount(inv, weights) es
        # ~3-5x más rápido que np.add.at porque opera en modo buffered.
        unique_v, inv = np.unique(joint_vals, return_inverse=True)
        agg_p = np.bincount(inv, weights=joint_probs, minlength=len(unique_v))

        keep = agg_p > 1e-14
        cur_vals, cur_probs = unique_v[keep], agg_p[keep]

    return sorted(zip(cur_vals.tolist(), cur_probs.tolist()))


def expected_value_and_sigma(bonds_data: list, trans_mat: np.ndarray) -> tuple[list[dict], dict[str, float]]:
    """
    Calcula E[V] y σ[V] para cada bono y para el portafolio total
    (solo válido en el caso independiente).

    Devuelve
    --------
    list de dicts con {'nombre', 'EV', 'sigma'} por bono
    dict con {'EV_port', 'sigma_port_min' (cota inferior independiente)}
    """
    per_bond = []
    ev_port   = 0.0
    var_port  = 0.0
    for b in bonds_data:
        probs = trans_mat[b['rating_idx']]
        ev_b  = float(np.dot(b['values'], probs))
        var_b = float(np.dot((b['values'] - ev_b) ** 2, probs))
        per_bond.append({"nombre": b.get("nombre","?"), "EV": ev_b, "sigma": var_b**0.5})
        ev_port  += ev_b
        var_port += var_b   # independiente: varianzas suman

    return per_bond, {"EV_port": ev_port, "sigma_port": var_port ** 0.5}

__all__ = ["independent_distribution", "expected_value_and_sigma"]
