"""Model-comparison workflows for advanced derivatives."""

from __future__ import annotations

from .models.black_scholes import BSMEngine
from .models.binomial import CRREngine
from .models.heston import HestonEngine
from .models.merton import MertonEngine


def compare_all_models(
    S, K, T, r, sigma, q=0.0, heston_params=None, merton_params=None, crr_N=200, crr_american=False
):
    """
    Run all four engines with compatible parameters.
    Returns a dict with call/put prices per model.
    heston_params: dict(v0, kappa, theta, xi, rho)
    merton_params: dict(lam, mu_j, sigma_j)
    """
    results = {}

    # BSM
    bsm = BSMEngine(S, K, T, r, sigma, q)
    results["BSM"] = {"call": bsm.call_price(), "put": bsm.put_price()}

    # CRR
    crr = CRREngine(S, K, T, r, sigma, q, crr_N, crr_american)
    results["CRR"] = {"call": crr.call_price(), "put": crr.put_price()}

    # Heston
    if heston_params is None:
        heston_params = {"v0": sigma**2, "kappa": 2.0, "theta": sigma**2, "xi": 0.3, "rho": -0.7}
    h = HestonEngine(
        S,
        K,
        T,
        r,
        q,
        heston_params["v0"],
        heston_params["kappa"],
        heston_params["theta"],
        heston_params["xi"],
        heston_params["rho"],
    )
    results["Heston"] = {"call": h.call_price(), "put": h.put_price()}

    # Merton
    if merton_params is None:
        merton_params = {"lam": 1.0, "mu_j": -0.1, "sigma_j": 0.15}
    m = MertonEngine(
        S, K, T, r, sigma, q, merton_params["lam"], merton_params["mu_j"], merton_params["sigma_j"]
    )
    results["Merton"] = {"call": m.call_price(), "put": m.put_price()}

    return results
