# Quantitative Actuarial

[![Tests](https://github.com/Walls16/Quantitative-Actuarial/actions/workflows/tests.yml/badge.svg)](https://github.com/Walls16/Quantitative-Actuarial/actions/workflows/tests.yml)
[![Docs](https://github.com/Walls16/Quantitative-Actuarial/actions/workflows/docs.yml/badge.svg)](https://github.com/Walls16/Quantitative-Actuarial/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-46A5F5?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/github/license/Walls16/Quantitative-Actuarial?logo=github)](LICENSE)

`quantitativeactuarial` is a pure Python library for actuarial mathematics,
quantitative finance, derivatives valuation, credit risk, and portfolio
optimization.

The package is imported as regular Python:

```python
import quantitativeactuarial as quact
```

It also includes a Streamlit app under `app/` that demonstrates the library, but
the app is only a presentation layer. The mathematical source of truth lives in
`src/quantitativeactuarial`.

## What Is Inside

- Interest-rate conversions, time value of money, annuities, amortization, bonds,
  equity valuation, cash-flow discounting, VaR and CVaR.
- Forward contracts, FRA valuation, Black-Scholes-Merton, Black-76, binomial
  trees, Greeks, implied volatility, payoff strategies, and exotic options.
- Advanced derivative models: BSM, CRR, Heston, Merton jump-diffusion, SABR,
  Variance Gamma, NIG, Bachelier, Monte Carlo, calibration, and model comparison.
- Credit risk models: transition matrices, CDS, CDO tranches, Gaussian copulas,
  default distributions, migration risk, VaR and CVaR.
- Portfolio optimization without PyPortfolioOpt as the library source:
  optimization strategies, efficient frontiers, risk metrics, backtesting, and
  stress testing.

## Architecture

This repository follows a strict separation between the domain package and the
visual interface:

```text
quantitative-actuarial/
├── src/
│   └── quantitativeactuarial/   # Pure library code
├── app/                         # Streamlit interface
├── docs/                        # Sphinx documentation
└── tests/                       # Pytest suite
```

The `src/quantitativeactuarial` package must not depend on Streamlit, Plotly
rendering, widgets, session state, or live UI behavior. It accepts ordinary
Python values, NumPy arrays, and pandas objects, then returns pure data
structures.

The `app/` layer handles user inputs, layout, charts, and rendering by calling
the library explicitly.

## Installation

For local development on this machine, use the Conda environment named `quact`.

```powershell
conda activate quact
python -m pip install -e ".[app,dev,docs]"
```

Or without activating the shell:

```powershell
conda run -n quact python -m pip install -e ".[app,dev,docs]"
```

Core package dependencies are intentionally small:

```text
numpy
pandas
scipy
```

Streamlit, Plotly, yfinance, pytest, Ruff, and Sphinx are optional development,
app, or documentation dependencies.

## Quick Examples

### Interest Rates

```python
import quantitativeactuarial as quact

effective = quact.tasa_nominal_a_efectiva(0.12, 12)
instantaneous = quact.tasa_nominal_a_instantanea(0.12, 12)

print(effective)
print(instantaneous)
```

### Vanilla Options

```python
import quantitativeactuarial as quact

call_price = quact.black_scholes(
    S=100.0,
    K=100.0,
    r=0.05,
    sigma=0.20,
    T=1.0,
    q=0.0,
    es_call=True,
)

print(call_price)
```

### Advanced Derivatives

```python
from quantitativeactuarial.derivados_avanzados import BSMEngine, SABREngine

bsm = BSMEngine(S=100, K=100, T=1, r=0.05, sigma=0.20)
sabr = SABREngine(F=100, K=105, T=1, alpha=0.20, beta=0.5, rho=-0.25, nu=0.40)

print(bsm.call_price())
print(sabr.implied_vol())
```

### Portfolio Optimization

```python
import pandas as pd
from quantitativeactuarial.portafolioopt import PortfolioOptimizer, compute_all_metrics

returns = pd.DataFrame(
    {
        "A": [0.01, -0.02, 0.015, 0.008],
        "B": [0.004, 0.006, -0.003, 0.012],
        "C": [-0.002, 0.010, 0.011, -0.004],
    }
)

optimizer = PortfolioOptimizer(returns)
weights = optimizer.optimize("Maximo Sharpe")
metrics = compute_all_metrics(returns @ weights)

print(weights)
print(metrics)
```

### Credit Risk

```python
import quantitativeactuarial as quact

transition_matrix = quact.DEFAULT_TM
print(transition_matrix)
```

## Streamlit App

Run the interactive app locally:

```powershell
conda run -n quact streamlit run app/main.py
```

The app includes pages for:

- rates and time value of money
- annuities, amortization, bonds, equity, corporate valuation
- portfolio analysis and optimization
- market risk
- forwards and derivative strategies
- vanilla, exotic, credit, and advanced derivative valuation
- VQD advanced derivative model pages
- live data demonstrations

If Streamlit shows an import error after code changes, fully stop and restart
the Streamlit process. During development it can keep old imported module state.

## Tests

The test suite uses pytest.

```powershell
conda run -n quact python -m pytest -q tests
```

Current expected baseline:

```text
164 passed
```

Regression expected values live in:

```text
tests/fixtures/results.json
```

Do not regenerate this file unless a deliberate mathematical change requires
new expected outputs.

## Formatting And Linting

Ruff is configured in `pyproject.toml`.

Format the repo:

```powershell
conda run -n quact python -m ruff format .
```

Check formatting:

```powershell
conda run -n quact python -m ruff format --check .
```

Run lint checks:

```powershell
conda run -n quact python -m ruff check .
```

The current lint baseline is intentionally conservative so the project can keep
moving while older Streamlit style debt is cleaned up gradually.

## Documentation

Documentation is built with Sphinx.

```powershell
conda run -n quact sphinx-build -W -b html docs docs\_build\html
```

The source files live in `docs/`, especially `docs/api/`. Generated HTML should
not be edited directly.

## GitHub Actions

The repository includes CI for tests and formatting:

```text
.github/workflows/tests.yml
```

It runs on push, pull request, and manual dispatch across Python:

```text
3.10, 3.11, 3.12, 3.13
```

The documentation workflow is:

```text
.github/workflows/docs.yml
```

It builds the Sphinx site for GitHub Pages.

## Development Principles

- Keep `src/quantitativeactuarial` pure, deterministic, typed, and testable.
- Keep Streamlit code in `app/` as interface code only.
- Move reusable financial math out of pages and into the library.
- Avoid global engines, hidden singleton state, and UI-coupled algorithms.
- Prefer small functional primitives and narrow model classes.
- Do not make PyPortfolioOpt the source of the portfolio library.
- Update docs and tests when public behavior changes.

The central rule is simple:

```text
library math in src, user interface in app
```

## Repository Guide For Agents

Additional repo-specific operating instructions are in:

```text
AGENTS.md
```

Agents working on this repository should read that file before making changes.
