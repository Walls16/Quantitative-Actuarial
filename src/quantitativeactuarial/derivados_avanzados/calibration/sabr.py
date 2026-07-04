"""SABR calibration workflows."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..models.sabr import SABREngine


def calibrate_sabr(
    forward: float,
    maturity: float,
    strikes: np.ndarray,
    market_ivs: np.ndarray,
    *,
    beta: float,
    initial_alpha: float = 0.2,
    initial_rho: float = -0.3,
    initial_nu: float = 0.4,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> dict[str, float]:
    """Calibrate SABR ``alpha``, ``rho``, and ``nu`` to a market IV smile."""

    def objective(params: np.ndarray) -> float:
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e6
        total = 0.0
        for strike, market_iv in zip(strikes, market_ivs):
            if np.isnan(market_iv):
                continue
            model_iv = SABREngine(forward, strike, maturity, alpha, beta, rho, nu).implied_vol()
            total += (model_iv - market_iv) ** 2
        return float(total)

    res = minimize(
        objective,
        [initial_alpha, initial_rho, initial_nu],
        method="L-BFGS-B",
        bounds=[(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)],
        options={"ftol": tol, "maxiter": max_iter},
    )
    alpha_cal, rho_cal, nu_cal = res.x
    rmse = np.sqrt(res.fun / max(len(strikes), 1))
    return {
        "alpha": float(alpha_cal),
        "beta": float(beta),
        "rho": float(rho_cal),
        "nu": float(nu_cal),
        "rmse": float(rmse),
        "success": bool(res.success),
    }
