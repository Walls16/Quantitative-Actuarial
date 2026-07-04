"""Merton jump-diffusion path simulation."""

from __future__ import annotations

import numpy as np

from ..models.merton import merton_jump_statistics


def simulate_merton_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    lam: float = 1.0,
    mu_j: float = 0.0,
    sigma_j: float = 0.2,
    n_paths: int = 30,
    n_steps: int = 252,
    seed: int | None = 42,
) -> dict[str, np.ndarray]:
    """Simulate Merton jump-diffusion price paths.

    Paths follow ``dS/S = (r-q-lambda*kappa)dt + sigma dW + dJ`` with Poisson
    jump arrivals and normally distributed log-jump sizes.  The function is
    deterministic for a fixed seed and returns the time grid, full path matrix,
    and terminal prices.
    """
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive.")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    kappa_j = merton_jump_statistics(lam, mu_j, sigma_j, sigma)["kappa_j"]
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S

    for path_idx in range(n_paths):
        for step in range(1, n_steps + 1):
            diffusion = (r - q - lam * kappa_j - 0.5 * sigma**2) * dt + sigma * np.sqrt(
                dt
            ) * rng.standard_normal()
            n_jumps = rng.poisson(lam * dt)
            jump = rng.normal(mu_j, sigma_j, n_jumps).sum() if n_jumps > 0 else 0.0
            paths[path_idx, step] = paths[path_idx, step - 1] * np.exp(diffusion + jump)

    return {
        "times": np.linspace(0.0, T, n_steps + 1),
        "paths": paths,
        "terminal_prices": paths[:, -1].copy(),
    }
