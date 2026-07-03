"""Functional public API for :mod:`quantitativeactuarial`.

The package exposes stateless actuarial, financial-math, derivatives-pricing,
and credit-risk primitives.  It intentionally contains no Streamlit dependency
and no global engine singleton.
"""

from .credito import *  # noqa: F403
from .credito import __all__ as _credito_all
from .derivados import *  # noqa: F403
from .derivados import __all__ as _derivados_all
from .tasas import *  # noqa: F403
from .tasas import __all__ as _tasas_all

__all__ = sorted(set(_credito_all) | set(_derivados_all) | set(_tasas_all))
