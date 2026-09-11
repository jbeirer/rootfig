"""Binning specifications and their resolution into ``hist`` axes."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import hist
import numpy as np

from rootfig.errors import BinningError

if TYPE_CHECKING:
    from rootfig.model.variables import Variable

__all__ = [
    "ROBUST_THRESHOLD",
    "Axis",
    "Bins",
    "RangeSpec",
    "auto_range",
    "log_bins",
    "resolve_axis",
    "validate_bins",
]

Axis: TypeAlias = hist.axis.Regular | hist.axis.Variable
"""The histogram axis types rootfig produces."""

Bins: TypeAlias = int | tuple[int, float, float] | Sequence[float] | np.ndarray | Axis
"""Accepted binning specifications.

* ``int`` - number of equal-width bins; the range is taken from ``range`` or
  inferred from the data.
* ``(n, low, high)`` - ``n`` equal-width bins between ``low`` and ``high``.
* a sequence of edges - variable-width bins.
* a :class:`hist.axis.Regular` or :class:`hist.axis.Variable` - used as is.
"""

RangeSpec: TypeAlias = tuple[float, float] | Literal["auto", "robust"] | None
"""How to choose the histogram range when ``bins`` is an integer.

* ``(low, high)`` - explicit.
* ``"auto"`` (default) - the finite minimum and maximum over all samples.
* ``"robust"`` - like ``"auto"`` but ignoring outliers far from the bulk of
  the data (see :func:`auto_range`), padded by 5 percent; useful when
  sentinel values such as ``-999`` would otherwise dominate the range.
"""


# --------------------------------------------------------------------------------------


def log_bins(n: int, low: float, high: float) -> np.ndarray:
    """Return ``n + 1`` logarithmically spaced bin edges between ``low`` and ``high``.

    Examples
    --------
    >>> log_bins(3, 1, 1000).tolist()
    [1.0, 10.0, 100.0, 1000.0]
    """
    if n < 1:
        msg = f"number of bins must be positive, got {n}"
        raise BinningError(msg)
    if not (0 < low < high):
        msg = f"log bins need 0 < low < high, got low={low}, high={high}"
        raise BinningError(msg)
    return np.geomspace(low, high, n + 1)


def validate_bins(bins: Bins, range_: RangeSpec) -> None:
    """Raise :class:`BinningError` if ``bins``/``range_`` are not a valid specification."""
    if isinstance(bins, hist.axis.Regular | hist.axis.Variable):
        return
    if isinstance(bins, bool):
        msg = "bins must be an int, (n, low, high), edges, or a hist axis, got a bool"
        raise BinningError(msg)
    if isinstance(bins, int):
        if bins < 1:
            msg = f"number of bins must be positive, got {bins}"
            raise BinningError(msg)
        if isinstance(range_, tuple):
            _validate_range(range_)
        elif range_ not in ("auto", "robust", None):
            msg = f"range must be (low, high), 'auto', or 'robust', got {range_!r}"
            raise BinningError(msg)
        return
    if isinstance(bins, tuple) and len(bins) == 3 and isinstance(bins[0], int):
        if bins[0] < 1:
            msg = f"number of bins must be positive, got {bins[0]}"
            raise BinningError(msg)
        _validate_range((bins[1], bins[2]))
        return
    edges = _edges_from(bins)
    if len(edges) < 2:
        msg = "bin edges must contain at least two values"
        raise BinningError(msg)
    if not np.all(np.diff(edges) > 0):
        msg = "bin edges must be strictly increasing"
        raise BinningError(msg)
    if not np.all(np.isfinite(edges)):
        msg = "bin edges must be finite"
        raise BinningError(msg)


def _validate_range(range_: tuple[float, float]) -> None:
    if len(range_) != 2:  # runtime guard for untyped callers
        msg = f"range must have two values (low, high), got {range_!r}"  # type: ignore[unreachable]
        raise BinningError(msg)
    low, high = range_
    if not (math.isfinite(low) and math.isfinite(high)):
        msg = f"range must be finite, got {range_!r}"
        raise BinningError(msg)
    if not low < high:
        msg = f"range must satisfy low < high, got {range_!r}"
        raise BinningError(msg)


def _edges_from(bins: Any) -> np.ndarray:
    try:
        edges = np.asarray(bins, dtype=float)
    except (TypeError, ValueError) as exc:
        msg = f"cannot interpret {bins!r} as bin edges"
        raise BinningError(msg) from exc
    if edges.ndim != 1:
        msg = f"bin edges must be one-dimensional, got shape {edges.shape}"
        raise BinningError(msg)
    return edges


ROBUST_THRESHOLD = 10.0
"""Modified z-score (in units of the median absolute deviation) beyond which values are
ignored by ``range="robust"``. Generous enough to keep physical tails, strict enough to
drop sentinel values such as ``-999``."""


def auto_range(
    arrays: Sequence[np.ndarray],
    *,
    mode: Literal["auto", "robust"] = "auto",
) -> tuple[float, float]:
    """Choose a histogram range covering the finite values of all ``arrays``.

    ``mode="auto"`` uses the overall minimum and maximum, with the upper edge
    nudged up so the maximum value lands inside the last bin. ``mode="robust"``
    first discards outliers whose modified z-score (``0.6745 * |x - median| /
    MAD``) exceeds :data:`ROBUST_THRESHOLD`, which removes sentinel values such
    as ``-999`` without trimming ordinary tails, and pads the result by 5
    percent of its span. A degenerate range (all values equal) is widened
    symmetrically; if there are no finite values at all, ``(0.0, 1.0)`` is
    returned.
    """
    values = [np.asarray(a, dtype=float).ravel() for a in arrays if len(a)]
    finite = [v[np.isfinite(v)] for v in values]
    finite = [v for v in finite if v.size]
    if not finite:
        return (0.0, 1.0)
    combined = np.concatenate(finite)
    if mode == "robust":
        kept = _reject_outliers(combined)
        low, high = float(kept.min()), float(kept.max())
        pad = 0.05 * (high - low)
        low, high = low - pad, high + pad
    else:
        low, high = float(combined.min()), float(combined.max())
        span = high - low
        high = high + (span * 1e-3 if span > 0 else 0.0)
    if not high > low:
        width = abs(low) * 0.1 if low != 0 else 0.5
        low, high = low - width, high + width
    return (low, high)


def _reject_outliers(values: np.ndarray, threshold: float = ROBUST_THRESHOLD) -> np.ndarray:
    median = np.median(values)
    deviation = np.abs(values - median)
    mad = np.median(deviation)
    if mad <= 0:
        # More than half the values are identical (e.g. all zero): fall back to the
        # mean absolute deviation, and if that is zero too, keep everything.
        mad = float(deviation.mean())
        if mad <= 0:
            return values
    score = 0.6745 * deviation / mad
    kept = values[score <= threshold]
    return kept if kept.size else values


def resolve_axis(
    variable: Variable,
    data: Sequence[np.ndarray] = (),
    *,
    name: str = "x",
) -> Axis:
    """Turn a :class:`Variable`'s binning into a concrete ``hist`` axis.

    Parameters
    ----------
    variable
        The variable whose ``bins`` and ``range`` are interpreted.
    data
        Flat arrays of values (one per sample) used to infer the range when
        ``bins`` is an integer and ``range`` is ``"auto"``/``"robust"``.
    name
        Axis name stored in the histogram. A ready-made ``hist`` axis keeps its
        own name.

    Raises
    ------
    BinningError
        If the range must be inferred but no data was given.
    """
    bins = variable.bins
    label = variable.axis_label
    if isinstance(bins, hist.axis.Regular | hist.axis.Variable):
        # Copy so the caller's axis (possibly shared between variables) is never modified;
        # the copy keeps edges, transform and flow traits. A shallow copy shares the
        # metadata dict on older boost-histogram releases, so copy deeply.
        axis = copy.deepcopy(bins)
        if not axis.label:
            axis.label = label
        return axis
    if isinstance(bins, int) and not isinstance(bins, bool):
        if isinstance(variable.range, tuple):
            low, high = variable.range
        else:
            if not data:
                msg = (
                    f"cannot infer a range for {variable.expression!r} without data; "
                    "pass range=(low, high) or bins=(n, low, high)"
                )
                raise BinningError(msg)
            mode: Literal["auto", "robust"] = "robust" if variable.range == "robust" else "auto"
            low, high = auto_range(data, mode=mode)
        return hist.axis.Regular(bins, low, high, name=name, label=label)
    if isinstance(bins, tuple) and len(bins) == 3 and isinstance(bins[0], int):
        n, low, high = bins
        return hist.axis.Regular(int(n), float(low), float(high), name=name, label=label)
    edges = _edges_from(bins)
    return hist.axis.Variable(edges, name=name, label=label)
