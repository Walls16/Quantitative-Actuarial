"""Credit VaR, CVaR, scaling, and parametric risk metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .data import TRADING_DAYS


def var_cvar_from_distribution(sorted_dist: list,
                                conf_levels: tuple[float, ...] = (0.90, 0.95, 0.99, 0.999),
                                normalize: bool = True) -> dict:
    """
    VaR y CVaR a partir de una distribución discreta ordenada (ascending).

    normalize : Si True (por defecto), normaliza las probabilidades a 1.

    Devuelve dict: {conf: {'EV', 'sigma', 'VaR', 'CVaR', 'q', 'sum_probs'}}

    CORRECCIÓN: el cálculo del cuantil usaba searchsorted(...) - 1, lo cual
    desplazaba el índice hacia un valor con CDF < alpha (sub-cuantil).
    Ahora se usa directamente el índice que devuelve searchsorted, que apunta
    al primer elemento con CDF >= alpha (definición estándar del cuantil inferior).
    """
    vals  = np.array([d[0] for d in sorted_dist], dtype=float)
    probs = np.array([d[1] for d in sorted_dist], dtype=float)
    sum_p = float(probs.sum())

    if normalize:
        probs_n = probs / sum_p if sum_p > 0 else probs
    else:
        probs_n = probs

    ev    = float(np.dot(vals, probs_n))
    var2  = float(np.dot((vals - ev) ** 2, probs_n))
    sigma = float(var2 ** 0.5)
    cum   = np.cumsum(probs_n)

    out = {}
    for conf in conf_levels:
        alpha = 1.0 - conf

        # CORRECCIÓN: searchsorted devuelve el primer i donde cum[i] >= alpha.
        # Ese es el cuantil inferior correcto; NO restar 1.
        idx   = int(np.searchsorted(cum, alpha, side='left'))
        idx   = min(idx, len(vals) - 1)   # clamping defensivo
        q_val = float(vals[idx])

        var_v = max(ev - q_val, 0.0)

        # CVaR = E[V | V <= q]  →  pérdida esperada en la cola
        tail_mask = vals <= q_val
        tail_sum  = probs_n[tail_mask].sum()
        if tail_sum > 1e-15:
            cvar_v = max(
                ev - float(np.dot(vals[tail_mask], probs_n[tail_mask]) / tail_sum),
                0.0,
            )
        else:
            cvar_v = var_v

        out[conf] = {
            "EV": ev, "sigma": sigma,
            "VaR": var_v, "CVaR": cvar_v,
            "q": q_val, "sum_probs": sum_p,
        }

    return out


def scale_var_cvar(results: dict,
                   conf_levels: tuple[float, ...] = (0.90, 0.95, 0.99, 0.999)) -> dict:
    """
    Scale annual CreditMetrics VaR/CVaR to regulatory horizons.

    CreditMetrics is a one-year model. The standard square-root-of-time
    conversion is:

    - ``VaR_1d = VaR_1y / sqrt(252)``
    - ``VaR_10d = VaR_1y * sqrt(10 / 252)``
    - ``CVaR`` scales like ``VaR``
    - ``Capital = 3 * VaR_10d``

    Parameters
    ----------
    results:
        Mapping returned by ``var_cvar_from_distribution`` or
        ``var_cvar_from_simulations``, keyed by confidence level.
    conf_levels:
        Confidence levels to include.

    Returns
    --------
    dict
        Mapping from confidence level to scaled risk metrics.
    """
    _k1d  = 1.0 / np.sqrt(TRADING_DAYS)          # ÷ √252
    _k10d = np.sqrt(10.0 / TRADING_DAYS)          # × √(10/252)
    CAPITAL_MULTIPLIER = 3.0                       # Basilea II/III

    out = {}
    for conf in conf_levels:
        if conf not in results:
            continue
        r = results[conf]
        var_1y  = r["VaR"]
        cvar_1y = r["CVaR"]

        var_1d   = var_1y  * _k1d
        cvar_1d  = cvar_1y * _k1d
        var_10d  = var_1y  * _k10d
        cvar_10d = cvar_1y * _k10d
        capital  = CAPITAL_MULTIPLIER * var_10d

        out[conf] = {
            "EV":      r["EV"],
            "sigma":   r["sigma"],
            "VaR_1y":  var_1y,
            "CVaR_1y": cvar_1y,
            "VaR_1d":  var_1d,
            "CVaR_1d": cvar_1d,
            "VaR_10d": var_10d,
            "CVaR_10d": cvar_10d,
            "Capital": capital,
            # campos opcionales
            "q":        r.get("q"),
            "sum_probs": r.get("sum_probs"),
        }
    return out


def var_cvar_from_simulations(sim_vals: np.ndarray,
                               conf_levels: tuple[float, ...] = (0.90, 0.95, 0.99, 0.999)) -> dict:
    """
    VaR y CVaR a partir de valores simulados.

    Devuelve dict: {conf: {'EV', 'sigma', 'VaR', 'CVaR', 'q'}}
    """
    ev    = float(np.mean(sim_vals))
    sigma = float(np.std(sim_vals))
    out   = {}
    for conf in conf_levels:
        q    = float(np.quantile(sim_vals, 1.0 - conf))
        var  = max(ev - q, 0.0)
        tail = sim_vals[sim_vals <= q]
        cvar = max(ev - float(tail.mean()), 0.0) if len(tail) > 0 else var
        out[conf] = {"EV": ev, "sigma": sigma, "VaR": var, "CVaR": cvar, "q": q}
    return out


def var_cvar_parametric(ev: float, sigma: float,
                        conf_levels: tuple[float, ...] = (0.90, 0.95, 0.99, 0.999)) -> dict:
    """
    VaR y CVaR paramétrico (aproximación normal — método clásico de CreditMetrics).

    Utiliza la media y varianza ya calculadas de la distribución exacta del
    portafolio, y asume que ésta es Normal(E[V], σ) para obtener el VaR.

        VaR(α)  = Φ⁻¹(α) · σ                    (pérdida respecto a E[V])
        CVaR(α) = φ(Φ⁻¹(α)) / (1 − α) · σ       (cola esperada bajo normal)

    Éste es el "VaR Paramétrico" que muestra la hoja de Excel (ver paso 5):
    Para σ = 16.13  →  VaR_95% ≈ 26.54,  VaR_99% ≈ 37.54,  VaR_99.9% ≈ 49.86

    Parámetros
    ----------
    ev          : E[V] del portafolio (calculado de la distribución exacta)
    sigma       : σ del portafolio   (calculado de la distribución exacta)
    conf_levels : niveles de confianza (fracción decimal)

    Devuelve
    --------
    dict: {conf: {'EV', 'sigma', 'VaR', 'CVaR'}}
    """
    out = {}
    for conf in conf_levels:
        z    = float(norm.ppf(conf))
        var  = max(z * sigma, 0.0)
        cvar = max(float(norm.pdf(z)) / (1.0 - conf) * sigma, var)
        out[conf] = {
            "EV":    ev,
            "sigma": sigma,
            "VaR":   var,
            "CVaR":  cvar,
        }
    return out

__all__ = [
    "var_cvar_from_distribution",
    "scale_var_cvar",
    "var_cvar_from_simulations",
    "var_cvar_parametric",
]
