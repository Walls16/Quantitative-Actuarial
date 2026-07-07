"""
stress_testing.py
==================
Evalúa cómo se habría comportado un portafolio (con pesos fijos) durante
escenarios históricos de crisis reales: Crisis Financiera 2008, COVID-19,
Burbuja Dot-com y la Subida de Tasas de la Fed en 2022.

Si alguno de los activos no tiene datos históricos suficientes para un
escenario (por ejemplo, una criptomoneda que no existía en 2008), ese activo
se excluye del cálculo para ese escenario y los pesos restantes se
renormalizan, indicándolo claramente al usuario.
"""

from __future__ import annotations

import pandas as pd

from .risk_metrics import (
    annualized_volatility,
    max_drawdown,
    cvar_historical,
)

# Escenarios históricos: (fecha inicio, fecha fin, descripción)
SCENARIOS: dict[str, dict] = {
    "Crisis Financiera 2008": {
        "start": "2007-10-09",
        "end": "2009-03-09",
        "descripcion": "Colapso de Lehman Brothers y crisis de hipotecas subprime. "
        "El S&P 500 cayó cerca de -56% desde su máximo.",
    },
    "COVID-19 (2020)": {
        "start": "2020-02-19",
        "end": "2020-04-30",
        "descripcion": "Caída abrupta de los mercados por la pandemia global y "
        "posterior recuperación impulsada por estímulos monetarios.",
    },
    "Burbuja Dot-com (2000-2002)": {
        "start": "2000-03-10",
        "end": "2002-10-09",
        "descripcion": "Estallido de la burbuja tecnológica de finales de los 90. "
        "El Nasdaq perdió más de -78% en este periodo.",
    },
    "Subida de Tasas Fed 2022": {
        "start": "2022-01-01",
        "end": "2022-12-31",
        "descripcion": "Ciclo agresivo de subidas de tasas de interés de la Reserva "
        "Federal para combatir la inflación, afectando tanto a "
        "acciones como a bonos.",
    },
}


def run_stress_test(
    prices_full: pd.DataFrame,
    weights: pd.Series,
    scenario_name: str,
    min_coverage: float = 0.6,
) -> dict | None:
    """Aplica los pesos de un portafolio a un escenario histórico.

    Parameters
    ----------
    prices_full: DataFrame de precios con la mayor cobertura histórica posible.
    weights: Series con los pesos del portafolio (índice = tickers).
    scenario_name: clave de `SCENARIOS`.
    min_coverage: fracción mínima de días con datos requerida para incluir
        un activo en el análisis del escenario.

    Returns
    -------
    dict con resultados, o None si no hay datos suficientes para NINGÚN
    activo durante el escenario.
    """
    scenario = SCENARIOS[scenario_name]
    start, end = scenario["start"], scenario["end"]

    window = prices_full.loc[start:end]
    if window.empty:
        return None

    coverage = window.notna().mean()
    valid_cols = coverage[coverage >= min_coverage].index.tolist()
    excluded = [c for c in prices_full.columns if c not in valid_cols]

    if not valid_cols:
        return None

    window = window[valid_cols].dropna()
    if len(window) < 5:
        return None

    scenario_returns = window.pct_change().dropna()
    if scenario_returns.empty:
        return None

    valid_weights = weights.reindex(valid_cols).fillna(0.0)
    w_sum = valid_weights.sum()
    if w_sum == 0:
        valid_weights = pd.Series(1.0 / len(valid_cols), index=valid_cols)
    else:
        valid_weights = valid_weights / w_sum

    port_returns = scenario_returns @ valid_weights.values
    port_returns.name = "Portafolio"

    bench_weights = pd.Series(1.0 / len(valid_cols), index=valid_cols)
    bench_returns = scenario_returns @ bench_weights.values
    bench_returns.name = "Equal Weight"

    cum_port = (1 + port_returns).cumprod()
    cum_bench = (1 + bench_returns).cumprod()

    return {
        "scenario": scenario_name,
        "descripcion": scenario["descripcion"],
        "period_start": window.index[0],
        "period_end": window.index[-1],
        "returns": port_returns,
        "cumulative_returns": cum_port,
        "total_return": cum_port.iloc[-1] - 1,
        "max_drawdown": max_drawdown(port_returns),
        "volatility": annualized_volatility(port_returns),
        "cvar_95": cvar_historical(port_returns, 0.05),
        "benchmark_returns": bench_returns,
        "benchmark_cumulative": cum_bench,
        "benchmark_total_return": cum_bench.iloc[-1] - 1,
        "benchmark_max_drawdown": max_drawdown(bench_returns),
        "valid_assets": valid_cols,
        "excluded_assets": excluded,
        "weights_used": valid_weights,
    }


def run_all_scenarios(prices_full: pd.DataFrame, weights: pd.Series) -> dict:
    """Ejecuta todos los escenarios definidos en SCENARIOS."""
    results = {}
    for name in SCENARIOS:
        results[name] = run_stress_test(prices_full, weights, name)
    return results
