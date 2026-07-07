"""Distribution diagnostics and Monte Carlo error analytics."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def distribution_moments(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Return mean, standard deviation, skewness, and excess kurtosis."""
    x = np.asarray(values, dtype=float)
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        skew = 0.0
        kurt = 0.0
    else:
        centered = x - mean
        skew = np.mean(centered**3) / std**3
        kurt = np.mean(centered**4) / std**4 - 3.0
    return {"mean": float(mean), "std": float(std), "skew": float(skew), "kurt": float(kurt)}


def excess_kurtosis(values: Sequence[float] | np.ndarray) -> float:
    """Return excess kurtosis, where a normal distribution has value zero."""
    return distribution_moments(values)["kurt"]


def theoretical_mc_error(
    reference_price: float,
    sigma: float,
    T: float,
    n_paths: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Return the ``O(1/sqrt(N))`` Monte Carlo standard-error proxy."""
    n = np.asarray(n_paths, dtype=float)
    return reference_price * sigma * np.sqrt(T) / np.sqrt(n)
