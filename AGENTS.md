# AGENTS.md

Repository guide for Codex and other coding agents working on this project.

## Environment

This repo is worked on locally on Windows at:

```text
C:\git\Quantitative-Actuarial
```

Always use the Conda environment named `quact` for Python commands.

Preferred command pattern:

```powershell
conda run -n quact python -m pytest -q tests
conda run -n quact python -m ruff format .
conda run -n quact python -m ruff check .
conda run -n quact sphinx-build -W -b html docs docs\_build\html
```

If Conda has trouble printing Ruff output on Windows, use UTF-8/no-capture:

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; conda run --no-capture-output -n quact python -m ruff check .
```

Do not assume another Python environment is valid for this repo.

## Architecture

The core architectural rule is strict separation of concerns:

```text
src/quantitativeactuarial/  = pure Python library/domain package
app/                        = Streamlit presentation layer only
docs/                       = Sphinx documentation
tests/                      = pytest regression tests
```

The mathematical domain package must be presentation-agnostic:

- No `streamlit` imports in `src/quantitativeactuarial`.
- No `st.*` calls in `src/quantitativeactuarial`.
- No Plotly rendering or Streamlit layout code in `src/quantitativeactuarial`.
- No hidden UI state, global engines, singleton math objects, or Streamlit caches in `src`.
- Functions in `src` should accept native Python types, NumPy arrays, or pandas objects and return primitives, dictionaries, arrays, Series, or DataFrames.

The Streamlit app must consume the library explicitly:

```python
from quantitativeactuarial.some_module import some_function
```

The app may handle widgets, layout, charts, session state, yfinance/live data access, and user-facing formatting. It should not contain actuarial, quantitative, optimization, pricing, risk, or calibration algorithms. If math is found in `app/`, move it into `src` and call it from the page.

## Package Layout

Important package areas:

```text
src/quantitativeactuarial/
├── tasas.py
├── derivados.py
├── credito.py
├── financial_math/
├── derivatives/
├── creditrisk/
├── derivados_avanzados/
└── portafolioopt/
```

Current direction:

- `financial_math/`: rates, TVM, annuities, bonds, loans, cashflows, corporate finance, equity, risk, portfolio math.
- `derivatives/`: vanilla/exotic derivative routines, forwards, trees, strategies.
- `creditrisk/`: CDS/CDO/copula/default-risk routines.
- `derivados_avanzados/`: advanced derivative models split into models, numerics, calibration, simulation, analytics.
- `portafolioopt/`: pure portfolio optimization routines. Do not rely on PyPortfolioOpt as the library source.

Avoid recreating monolithic "engine" or "God object" classes. Prefer small, stateless, typed functions and narrowly scoped model classes where they represent a mathematical model.

## Streamlit App Rules

The app lives under `app/`.

`app/main.py` is the Streamlit entrypoint. Pages live in `app/pages/`.

Streamlit pages may:

- define widgets
- read user inputs
- call pure functions/classes from `quantitativeactuarial`
- render DataFrames, metrics, Plotly charts, warnings, and explanatory text
- fetch live market data through app/service helpers

Streamlit pages must not:

- implement reusable quantitative algorithms inline
- become the source of truth for pricing, rates, risk, calibration, simulation, or optimization
- use PyPortfolioOpt as the library implementation

When migrating another Streamlit app into this repo, put reusable Python math into `src/quantitativeactuarial/<domain>/` and convert the old `app.py` into a Streamlit page under `app/pages/`.

## Tests

Tests use pytest and live in `tests/`.

The regression fixture file is:

```text
tests/fixtures/results.json
```

The naming convention is `results`, not `golden_results`.

Do not regenerate `results.json` unless the user explicitly asks to update expected results or a deliberate mathematical bug fix changes outputs. When outputs change intentionally, explain which formulas changed and why.

Main test command:

```powershell
conda run -n quact python -m pytest -q tests
```

Expected current baseline:

```text
164 passed
```

## Formatting And Linting

Ruff is configured in `pyproject.toml`.

Formatting command:

```powershell
conda run -n quact python -m ruff format .
```

Format check:

```powershell
conda run -n quact python -m ruff format --check .
```

Lint check:

```powershell
conda run -n quact python -m ruff check .
```

The current Ruff lint baseline is intentionally conservative:

```toml
select = ["E9", "F63", "F7", "F82"]
```

That keeps critical correctness checks green while avoiding a giant cleanup of historical Streamlit style debt. Do not broaden lint rules unless the user asks or you are prepared to fix the resulting findings.

## Documentation

Docs are built with Sphinx from `docs/`.

Build command:

```powershell
conda run -n quact sphinx-build -W -b html docs docs\_build\html
```

Do not hand-edit generated HTML in `docs/_build` or `docs_build`. Edit `.rst` files and docstrings instead.

When adding public library modules or functions, update the relevant docs under:

```text
docs/api/
```

## Dependency Boundaries

Core package dependencies should stay minimal and library-safe:

- allowed core dependencies: NumPy, pandas, SciPy
- app-only dependencies: Streamlit, Plotly, yfinance
- dev-only dependencies: pytest, Ruff
- docs-only dependencies: Sphinx, Furo, MyST parser, sphinx-copybutton, sphinx-autobuild

Do not add frontend, plotting, live-data, or UI dependencies to the core package unless the user explicitly approves a broader package boundary.

## Verification Checklist

For most code changes, run:

```powershell
conda run -n quact python -m ruff format .
conda run -n quact python -m ruff check .
conda run -n quact python -m pytest -q tests
```

For docs or public API changes, also run:

```powershell
conda run -n quact sphinx-build -W -b html docs docs\_build\html
```

For Streamlit page changes, also compile the touched pages:

```powershell
conda run -n quact python -m compileall -q app src
```

If a Streamlit import error persists after code is correct, restart Streamlit completely. It can retain old module state while developing.

## Git And Worktree Notes

The worktree may already contain many modified or untracked files from ongoing migrations. Do not revert user changes. Do not run destructive Git commands such as `git reset --hard` or `git checkout --` unless the user explicitly asks.

Generated documentation build output may appear in Git status. Treat generated build folders as disposable unless the user specifically asks to publish or inspect the built site.

## Coding Style

- Prefer pure, typed functions for domain logic.
- Add docstrings for public mathematical functions.
- Keep edits scoped to the requested domain.
- Use existing package patterns before inventing new abstractions.
- Avoid broad `except Exception` in `src`; catch expected errors.
- Do not hide dependencies through global mutable state.
- Preserve compatibility imports when practical, but keep new implementations decentralized.

Above all: keep `src` as the distributable quantitative actuarial library and `app` as a consumer of that library.
