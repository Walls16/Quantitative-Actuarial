"""Backward-compatible derivatives facade.

The implementation is split into flat modules under
:mod:`quantitativeactuarial.derivatives`.  This module reexports the public API so
existing imports such as ``from quantitativeactuarial.derivados import
black_scholes`` continue to work.
"""

from .derivatives import *  # noqa: F403
from .derivatives import __all__  # noqa: F401
