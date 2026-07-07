"""Portfolio optimization primitives for :mod:`quantitativeactuarial`.

The subpackage contains pure portfolio math: optimization strategies, risk
metrics, efficient-frontier generation, backtesting, and historical stress
testing.  It has no Streamlit dependency and does not use PyPortfolioOpt.
"""

from .backtesting import run_backtest
from .data import get_returns, validate_tickers
from .efficient_frontier import markowitz_frontier, monte_carlo_portfolios
from .optimization import PortfolioOptimizer, STRATEGY_NAMES
from .risk_metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compute_all_metrics,
    conditional_drawdown_at_risk,
    cumulative_returns,
    cvar_historical,
    downside_deviation,
    drawdown_series,
    max_drawdown,
    portfolio_returns,
    sharpe_ratio,
    sortino_ratio,
    var_historical,
)
from .stress_testing import SCENARIOS, run_all_scenarios, run_stress_test

__all__ = [
    "PortfolioOptimizer",
    "SCENARIOS",
    "STRATEGY_NAMES",
    "annualized_return",
    "annualized_volatility",
    "calmar_ratio",
    "compute_all_metrics",
    "conditional_drawdown_at_risk",
    "cumulative_returns",
    "cvar_historical",
    "downside_deviation",
    "drawdown_series",
    "get_returns",
    "markowitz_frontier",
    "max_drawdown",
    "monte_carlo_portfolios",
    "portfolio_returns",
    "run_all_scenarios",
    "run_backtest",
    "run_stress_test",
    "sharpe_ratio",
    "sortino_ratio",
    "validate_tickers",
    "var_historical",
]
