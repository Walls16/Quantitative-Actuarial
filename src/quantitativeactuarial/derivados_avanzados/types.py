"""Shared type aliases for advanced derivatives routines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np

ArrayLike: TypeAlias = Sequence[float] | np.ndarray
CharacteristicFunction: TypeAlias = Callable[[complex | np.ndarray], complex | np.ndarray]
