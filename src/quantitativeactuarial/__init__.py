"""Public API for the :mod:`quantitativeactuarial` package.

The package is intentionally presentation-agnostic: it contains no Streamlit
imports and exposes deterministic actuarial and quantitative-finance routines
that operate on Python, NumPy, and pandas objects.

Typical usage:

    import quantitativeactuarial as quact

    engine = quact.FinancialMathEngine()
    engine.tasa_nominal_a_efectiva(0.12, 12)
"""

from .tasas import FinancialMathEngine

__all__ = ["FinancialMathEngine"]
