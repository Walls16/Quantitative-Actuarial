"""Credit Default Swap valuation primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cds_probability_table(hazard_rate: float, maturity: int) -> pd.DataFrame:
    """
    Build annual survival and default probabilities for a CDS.

    The model assumes an exponential default time with constant hazard rate
    ``lambda``:

    ``S(t) = exp(-lambda t)``
    ``F(t) = 1 - S(t)``
    ``q(t) = S(t - 1) - S(t)``
    """
    rows = []
    for t in range(1, maturity + 1):
        survival_t = float(np.exp(-hazard_rate * t))
        survival_prev = float(np.exp(-hazard_rate * (t - 1)))
        rows.append(
            {
                "t (años)": t,
                "S(t) Sobrevivencia": survival_t,
                "F(t) Incump. Acumulada": 1.0 - survival_t,
                "q(t) Incump. Marginal": survival_prev - survival_t,
            }
        )
    return pd.DataFrame(rows)


def cds_premium_leg_table(hazard_rate: float, discount_rate: float, maturity: int) -> pd.DataFrame:
    """
    Present value table for the CDS premium leg before multiplying by spread.

    Each unit spread payment at date ``t`` has expected value
    ``S(t) exp(-r t)``.
    """
    rows = []
    for t in range(1, maturity + 1):
        survival_t = float(np.exp(-hazard_rate * t))
        discount = float(np.exp(-discount_rate * t))
        rows.append(
            {
                "Tiempo (años)": t,
                "Prob. Sobrevivencia S(t)": survival_t,
                "Pago Esperado (×s)": survival_t,
                "Factor VP  e^{-rt}": discount,
                "VP Pago Esperado (×s)": survival_t * discount,
            }
        )
    df = pd.DataFrame(rows)
    df.loc["Total"] = df[["VP Pago Esperado (×s)"]].sum()
    return df


def cds_contingent_leg_table(
    hazard_rate: float, discount_rate: float, maturity: int, recovery_rate: float
) -> pd.DataFrame:
    """
    Present value table for the CDS contingent default leg.

    Defaults are assumed to occur at midpoints ``t - 0.5`` and the protection
    payment is ``LGD = 1 - recovery_rate``.
    """
    lgd = 1.0 - recovery_rate
    rows = []
    for t in range(1, maturity + 1):
        t_mid = t - 0.5
        survival_prev = float(np.exp(-hazard_rate * (t - 1)))
        survival_t = float(np.exp(-hazard_rate * t))
        default_prob = survival_prev - survival_t
        expected_payment = lgd * default_prob
        discount = float(np.exp(-discount_rate * t_mid))
        rows.append(
            {
                "Tiempo (años)": t_mid,
                "Prob. Incumplimiento Marginal q(t)": default_prob,
                "Tasa Recuperación (RR)": recovery_rate,
                "LGD (1-RR)": lgd,
                "Pago Parcial Esperado": expected_payment,
                "Factor VP  e^{-rt_mid}": discount,
                "VP Pago Parcial Esperado": expected_payment * discount,
            }
        )
    df = pd.DataFrame(rows)
    df.loc["Total"] = df[["VP Pago Parcial Esperado"]].sum()
    return df


def cds_accrued_premium_table(
    hazard_rate: float, discount_rate: float, maturity: int
) -> pd.DataFrame:
    """
    Present value table for accrued premium paid on default.

    If default occurs in year ``t``, the buyer pays half a period of accrued
    premium, modeled as ``0.5 * q(t)`` and discounted from ``t - 0.5``.
    """
    rows = []
    for t in range(1, maturity + 1):
        t_mid = t - 0.5
        survival_prev = float(np.exp(-hazard_rate * (t - 1)))
        survival_t = float(np.exp(-hazard_rate * t))
        default_prob = survival_prev - survival_t
        expected_accrual = 0.5 * default_prob
        discount = float(np.exp(-discount_rate * t_mid))
        rows.append(
            {
                "Tiempo (años)": t_mid,
                "Prob. Incumplimiento Marginal q(t)": default_prob,
                "Pago Prorrateado (×s)": 0.5,
                "Pago Prorrateado Esperado (×s)": expected_accrual,
                "Factor VP  e^{-rt_mid}": discount,
                "VP Pago Prorrateado Esperado (×s)": expected_accrual * discount,
            }
        )
    df = pd.DataFrame(rows)
    df.loc["Total"] = df[["VP Pago Prorrateado Esperado (×s)"]].sum()
    return df


def cds_fair_spread(vpc_total: float, vppp_total: float, vpv_total: float) -> float:
    """
    Compute the fair CDS spread.

    The no-arbitrage condition is:

    ``spread * (VPC + VPPP) = VPV``
    """
    denominator = vpc_total + vppp_total
    if denominator == 0:
        return 0.0
    return float(vpv_total / denominator)


def value_cds(
    hazard_rate: float, discount_rate: float, maturity: int, recovery_rate: float
) -> dict[str, float | pd.DataFrame]:
    """Return the CDS legs, fair spread, and supporting tables."""
    probabilities = cds_probability_table(hazard_rate, maturity)
    fixed_leg = cds_premium_leg_table(hazard_rate, discount_rate, maturity)
    contingent_leg = cds_contingent_leg_table(hazard_rate, discount_rate, maturity, recovery_rate)
    accrued_premium = cds_accrued_premium_table(hazard_rate, discount_rate, maturity)

    vpc_total = float(fixed_leg.loc["Total", "VP Pago Esperado (×s)"])
    vpv_total = float(contingent_leg.loc["Total", "VP Pago Parcial Esperado"])
    vppp_total = float(accrued_premium.loc["Total", "VP Pago Prorrateado Esperado (×s)"])
    spread = cds_fair_spread(vpc_total, vppp_total, vpv_total)

    return {
        "probabilidades": probabilities,
        "vpc": fixed_leg,
        "vpv": contingent_leg,
        "vppp": accrued_premium,
        "vpc_total": vpc_total,
        "vpv_total": vpv_total,
        "vppp_total": vppp_total,
        "spread": spread,
    }


__all__ = [
    "cds_probability_table",
    "cds_premium_leg_table",
    "cds_contingent_leg_table",
    "cds_accrued_premium_table",
    "cds_fair_spread",
    "value_cds",
]
