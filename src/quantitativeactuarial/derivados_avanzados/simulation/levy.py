"""Levy return simulation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..analytics.distributions import distribution_moments


def simulate_vg_nig_returns(
    T: float,
    vg_sigma: float,
    vg_theta: float,
    vg_nu: float,
    nig_alpha: float,
    nig_beta: float,
    nig_delta: float,
    sigma_bsm: float,
    n_sim: int = 50_000,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Simulate VG, NIG, and Gaussian benchmark log-return samples.

    Variance Gamma is sampled with a Gamma time change.  NIG is sampled through
    an inverse-Gaussian time change.  A Gaussian benchmark with volatility
    ``sigma_bsm`` is included for direct distributional comparison.
    """
    if n_sim <= 0:
        raise ValueError("n_sim must be positive.")
    if vg_nu <= 0:
        raise ValueError("vg_nu must be positive.")
    if nig_alpha <= abs(nig_beta):
        raise ValueError("nig_alpha must be greater than abs(nig_beta).")

    rng = np.random.default_rng(seed)
    gamma_times = rng.gamma(T / vg_nu, vg_nu, n_sim)
    vg_returns = vg_theta * gamma_times + vg_sigma * np.sqrt(gamma_times) * rng.standard_normal(
        n_sim
    )

    ig_mean = nig_delta * T / np.sqrt(nig_alpha**2 - nig_beta**2)
    ig_shape = (nig_delta * T) ** 2
    v_ig = rng.standard_normal(n_sim) ** 2
    x_ig = (
        ig_mean
        + ig_mean**2 * v_ig / (2 * ig_shape)
        - ig_mean / (2 * ig_shape) * np.sqrt(4 * ig_mean * ig_shape * v_ig + ig_mean**2 * v_ig**2)
    )
    u_ig = rng.random(n_sim)
    ig_times = np.where(u_ig <= ig_mean / (ig_mean + x_ig), x_ig, ig_mean**2 / x_ig)
    nig_returns = nig_beta * ig_times + np.sqrt(ig_times) * rng.standard_normal(n_sim)

    gbm_returns = rng.normal(0.0, sigma_bsm * np.sqrt(T), n_sim)
    return {
        "vg_returns": vg_returns,
        "nig_returns": nig_returns,
        "gbm_returns": gbm_returns,
        "moments": {
            "VG": distribution_moments(vg_returns),
            "NIG": distribution_moments(nig_returns),
            "GBM": distribution_moments(gbm_returns),
        },
    }
