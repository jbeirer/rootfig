"""Shared type aliases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import hist
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    Hist: TypeAlias = hist.Hist[Any]
    """A ``hist.Hist`` with any storage (``Weight`` in everything rootfig produces)."""
else:
    Hist = hist.Hist

FloatArray: TypeAlias = npt.NDArray[np.float64]
"""A one-dimensional float array."""

AnyArray: TypeAlias = npt.NDArray[Any]
"""A NumPy array of any dtype."""
