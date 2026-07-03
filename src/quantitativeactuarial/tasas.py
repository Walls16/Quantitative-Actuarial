"""Backward-compatible financial-math facade.

The implementation is split into flat modules under
:mod:`quantitativeactuarial.financial_math`.  This module reexports the public
API so existing imports such as ``from quantitativeactuarial.tasas import
valor_presente`` continue to work.
"""

from .financial_math import *  # noqa: F403
from .financial_math import __all__  # noqa: F401
