"""Application computation namespace.

The Streamlit pages import this module as ``quact``.  Pure domain functions come
from ``quantitativeactuarial``; market-data helpers live in ``app`` because they
perform network I/O and are not part of the pure package.
"""

from quantitativeactuarial import *  # noqa: F403
from app.market_services import *  # noqa: F403
