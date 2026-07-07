"""Geometric Brownian motion path simulation."""

from __future__ import annotations

import numpy as np


def simulate_gbm_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_paths: int = 30,
    n_steps: int = 252,
    seed: int | None = 99,
) -> dict[str, np.ndarray]:
    """Simulate geometric Brownian motion price paths under risk-neutral drift."""
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive.")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S
    paths[:, 1:] = S * np.exp(np.cumsum(increments, axis=1))
    return {
        "times": np.linspace(0.0, T, n_steps + 1),
        "paths": paths,
        "terminal_prices": paths[:, -1].copy(),
    }
