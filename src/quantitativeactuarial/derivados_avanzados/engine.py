"""Compatibility facade for the former monolithic advanced-derivatives engine.

Implementation now lives in model, numerics, calibration, simulation, and
analytics modules.  Import from ``quantitativeactuarial.derivados_avanzados``
for the stable public API.
"""

from . import *  # noqa: F403
from . import __all__  # noqa: F401
