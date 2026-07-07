"""Heston calibration workflows."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..models.heston import HestonEngine
from ..numerics.implied_volatility import implied_volatility_black_scholes


def calibrate_heston(
    S,
    T,
    r,
    q,
    strikes,
    market_ivs,
    kappa0=2.0,
    theta0=0.04,
    xi0=0.3,
    rho0=-0.5,
    v00=0.04,
    method="L-BFGS-B",
    tol=1e-8,
    max_iter=300,
):
    """
    Calibrate Heston to a market IV smile via minimizing RMSE.

    Parameters
    ----------
    S, T, r, q    : spot, maturity, rate, dividend
    strikes       : array of market strikes
    market_ivs    : array of market implied vols (as decimals, e.g. 0.20)
    Returns       : dict with calibrated params + RMSE + success flag
    """

    def objective(params):
        kappa, theta, xi, rho, v0 = params
        # Parameter constraints
        if (
            kappa <= 0
            or theta <= 0
            or xi <= 0
            or abs(rho) >= 1
            or v0 <= 0
            or 2 * kappa * theta <= xi**2 * 0.1
        ):  # soft Feller
            return 1e8
        total = 0.0
        for K, miv in zip(strikes, market_ivs):
            if np.isnan(miv) or miv <= 0:
                continue
            h = HestonEngine(S, K, T, r, q, v0, kappa, theta, xi, rho)
            price = h.call_price()
            model_iv = implied_volatility_black_scholes(price, S, K, T, r, q, "call")
            if np.isnan(model_iv):
                total += 1.0
            else:
                total += (model_iv - miv) ** 2
        return total

    x0 = [kappa0, theta0, xi0, rho0, v00]
    bounds = [(0.01, 15.0), (0.001, 1.0), (0.01, 2.0), (-0.999, 0.999), (0.0001, 1.0)]

    res = minimize(
        objective, x0, method=method, bounds=bounds, options={"maxiter": max_iter, "ftol": tol}
    )

    kappa_c, theta_c, xi_c, rho_c, v0_c = res.x
    n_valid = sum(1 for iv in market_ivs if not np.isnan(iv) and iv > 0)
    rmse = np.sqrt(res.fun / max(n_valid, 1))

    return {
        "kappa": kappa_c,
        "theta": theta_c,
        "xi": xi_c,
        "rho": rho_c,
        "v0": v0_c,
        "rmse": rmse,
        "success": res.success,
        "message": res.message,
    }
