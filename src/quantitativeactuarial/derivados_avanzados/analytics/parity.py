"""Parity and deterministic pricing analytics."""

from __future__ import annotations

import numpy as np


def parity_check(call, put, S, K, T, r, q=0.0):
    """Put-Call parity residual."""
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    return {"lhs": lhs, "rhs": rhs, "residual": abs(lhs - rhs)}


def intrinsic_time_value(price, S, K, option_type="call"):
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    return {"intrinsic": intrinsic, "time_value": max(price - intrinsic, 0)}


def forward_price_continuous(spot: float, r: float, q: float, T: float) -> float:
    """Return the continuously compounded forward price.

    Implements ``F_0 = S_0 exp((r - q)T)`` for a risky asset with continuous
    dividend yield ``q`` and risk-free rate ``r``.
    """
    return float(spot * np.exp((r - q) * T))
