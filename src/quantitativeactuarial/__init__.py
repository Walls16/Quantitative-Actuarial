"""Functional public API for :mod:`quantitativeactuarial`.

The package exposes stateless actuarial, financial-math, derivatives-pricing,
advanced-derivatives, portfolio-optimization, and credit-risk primitives. It
contains no Streamlit dependency and no global engine singleton.
"""

from .advanced_derivatives import *  # noqa: F403
from .advanced_derivatives import __all__ as _advanced_derivatives_all
from .credit import *  # noqa: F403
from .credit import __all__ as _credit_all
from .derivatives import *  # noqa: F403
from .derivatives import __all__ as _derivatives_all
from .portfolioopt import *  # noqa: F403
from .portfolioopt import __all__ as _portfolioopt_all
from .rates import *  # noqa: F403
from .rates import __all__ as _rates_all

__all__ = sorted(
    set(_advanced_derivatives_all)
    | set(_credit_all)
    | set(_derivatives_all)
    | set(_portfolioopt_all)
    | set(_rates_all)
)
