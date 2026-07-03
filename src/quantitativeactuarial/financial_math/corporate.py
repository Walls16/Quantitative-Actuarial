"""Corporate valuation, CAPM, and DCF helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def capm_cost_of_equity(risk_free_rate: float, beta: float, market_return: float) -> float:
    """Compute cost of equity with CAPM: ``K_e = r_f + beta (E[R_m] - r_f)``."""
    return float(risk_free_rate + beta * (market_return - risk_free_rate))


def weighted_average_cost_of_capital(
    cost_of_equity: float,
    equity_weight: float,
    cost_of_debt: float,
    tax_rate: float,
) -> float:
    """Compute WACC from equity weight, after-tax cost of debt, and equity cost."""
    debt_weight = 1.0 - equity_weight
    return float(cost_of_equity * equity_weight + cost_of_debt * (1.0 - tax_rate) * debt_weight)


def dcf_valuation(
    fcf_base: float,
    projection_growth: float,
    terminal_growth: float,
    discount_rate: float,
    projection_years: int,
    net_debt: float,
    shares_outstanding: float,
) -> dict[str, float | list[float] | pd.DataFrame]:
    """
    Value a firm with a Gordon-growth terminal-value DCF.

    Free cash flows are projected as ``FCF_t = FCF_0 (1 + g)^t`` and terminal
    value is ``FCF_{n+1} / (WACC - g_terminal)``.
    """
    if discount_rate <= terminal_growth:
        raise ValueError("discount_rate must be greater than terminal_growth.")

    years = list(range(1, int(projection_years) + 1))
    fcfs = [float(fcf_base * (1 + projection_growth) ** t) for t in years]
    pv_fcfs = [float(f / (1 + discount_rate) ** t) for t, f in zip(years, fcfs)]
    terminal_fcf = float(fcfs[-1] * (1 + terminal_growth))
    terminal_value = float(terminal_fcf / (discount_rate - terminal_growth))
    pv_terminal_value = float(terminal_value / (1 + discount_rate) ** int(projection_years))
    pv_fcf_total = float(sum(pv_fcfs))
    enterprise_value = float(pv_fcf_total + pv_terminal_value)
    equity_value = float(enterprise_value - net_debt)
    price_per_share = float(equity_value / shares_outstanding) if shares_outstanding > 0 else 0.0

    table = pd.DataFrame(
        {
            "Año": years,
            "FCF Proyectado": fcfs,
            "Factor Descuento": [float(1 / (1 + discount_rate) ** t) for t in years],
            "Valor Presente": pv_fcfs,
        }
    )

    return {
        "fcfs": fcfs,
        "pv_fcfs": pv_fcfs,
        "terminal_fcf": terminal_fcf,
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal_value,
        "pv_fcf_total": pv_fcf_total,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "price_per_share": price_per_share,
        "projection_table": table,
    }


def dcf_sensitivity_matrix(
    fcfs: list[float] | np.ndarray,
    terminal_growth_values: list[float] | np.ndarray,
    discount_rate_values: list[float] | np.ndarray,
    net_debt: float,
    shares_outstanding: float,
) -> np.ndarray:
    """Build a price-per-share sensitivity grid over terminal growth and WACC."""
    cashflows = np.asarray(fcfs, dtype=float)
    g_vals = np.asarray(terminal_growth_values, dtype=float)
    wacc_vals = np.asarray(discount_rate_values, dtype=float)
    out = np.zeros((len(g_vals), len(wacc_vals)))
    projection_years = len(cashflows)

    for i, growth in enumerate(g_vals):
        for j, discount_rate in enumerate(wacc_vals):
            if discount_rate <= growth or discount_rate <= 0:
                out[i, j] = np.nan
                continue
            terminal_fcf = cashflows[-1] * (1 + growth)
            terminal_value = terminal_fcf / (discount_rate - growth)
            pv_terminal_value = terminal_value / (1 + discount_rate) ** projection_years
            pv_fcfs = sum(cashflows[t - 1] / (1 + discount_rate) ** t for t in range(1, projection_years + 1))
            equity_value = (pv_fcfs + pv_terminal_value) - net_debt
            out[i, j] = equity_value / shares_outstanding if shares_outstanding > 0 else 0.0
    return out


def beta_alpha_from_returns(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float,
    trading_days: int = 252,
) -> dict[str, float | pd.DataFrame]:
    """
    Estimate CAPM beta, alpha, annual returns, annual volatility, and cost of equity.

    Daily excess returns are regressed by ordinary least squares:
    ``R_i - r_f = alpha + beta (R_m - r_f)``.
    """
    df = pd.concat([asset_returns.rename("Accion"), market_returns.rename("Mercado")], axis=1, join="inner").dropna()
    rf_daily = (1 + risk_free_rate) ** (1 / trading_days) - 1
    df["Exc_A"] = df["Accion"] - rf_daily
    df["Exc_M"] = df["Mercado"] - rf_daily
    x = np.column_stack([np.ones(len(df)), df["Exc_M"].values])
    alpha_daily, beta = np.linalg.lstsq(x, df["Exc_A"].values, rcond=None)[0]
    alpha_annual = float(alpha_daily * trading_days)
    ret_asset = float((1 + df["Accion"].mean()) ** trading_days - 1)
    ret_market = float((1 + df["Mercado"].mean()) ** trading_days - 1)
    vol_asset = float(df["Accion"].std() * np.sqrt(trading_days))
    vol_market = float(df["Mercado"].std() * np.sqrt(trading_days))
    cost_of_equity = capm_cost_of_equity(risk_free_rate, float(beta), ret_market)

    return {
        "alpha": alpha_annual,
        "beta": float(beta),
        "ret_a": ret_asset,
        "ret_m": ret_market,
        "vol_a": vol_asset,
        "vol_m": vol_market,
        "ke": cost_of_equity,
        "rf": float(risk_free_rate),
        "returns": df,
    }


__all__ = [
    "capm_cost_of_equity",
    "weighted_average_cost_of_capital",
    "dcf_valuation",
    "dcf_sensitivity_matrix",
    "beta_alpha_from_returns",
]
