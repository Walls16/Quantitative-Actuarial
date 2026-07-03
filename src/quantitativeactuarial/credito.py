"""Backward-compatible credit-risk facade.

The implementation is split into flat modules under
:mod:`quantitativeactuarial.creditrisk`.  This module reexports the public API so
existing imports such as ``from quantitativeactuarial.credito import DEFAULT_TM``
continue to work.
"""

from .creditrisk import *  # noqa: F403
from .creditrisk import __all__  # noqa: F401
