"""Type aliases for credit-risk primitives."""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class BondData(TypedDict):
    """Minimal bond payload consumed by distribution and copula functions."""

    rating_idx: int
    values: np.ndarray


RiskResult = dict[str, float]
RiskResults = dict[float, RiskResult]
Distribution = list[tuple[float, float]]

__all__ = ["BondData", "RiskResult", "RiskResults", "Distribution"]
