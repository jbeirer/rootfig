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
    "DEFAULT_RANGE",
    "ROBUST_COVERAGE_BUDGET",
    "ROBUST_DISCRETE_VALUES",
    "ROBUST_LADDER",
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
* ``"robust"`` (default) - like ``"auto"`` but ignoring outliers far from the
  bulk of the data and cutting the thin end of a tail (see :func:`auto_range`),
  so that neither a sentinel such as ``-999`` nor a long tail dominates the
  range. The result is padded by 5 percent but clamped to the ``"auto"`` range,
  so it never reaches past the data; a sample with a hard edge or with fewer
  than :data:`ROBUST_DISCRETE_VALUES` distinct values is left where ``"auto"``
  puts it. Values outside land in the under/overflow. Degenerate ranges are
  widened symmetrically.
* ``"auto"`` - the finite minimum and maximum over all samples.
"""

DEFAULT_RANGE: RangeSpec = "robust"
"""Range inference used when a :class:`~rootfig.model.Variable` does not ask for one."""


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


ROBUST_THRESHOLD = 30.0
"""Modified z-score (in units of the median absolute deviation) beyond which values are
ignored by ``range="robust"``. This is a distance threshold, not a sentinel detector:
physical tails (including log-normal and Student-t samples) can exceed it, and
sentinels are rejected only when sufficiently far from the bulk of the data.

It is the widest range ``"robust"`` will produce: a sentinel far from the bulk is
rejected here, and :data:`ROBUST_COVERAGE_BUDGET` may then tighten the result
further, never past it."""

ROBUST_COVERAGE_BUDGET = 0.005
"""Fraction of the entries ``range="robust"`` may move out of the view to cut a tail.

:data:`ROBUST_THRESHOLD` measures a distance from the bulk, so a tail that reaches
far but thins out smoothly stays inside it and leaves the interesting part of the
distribution in a small corner of the axis. Starting from that range, the threshold
is tightened along :data:`ROBUST_LADDER` for as long as the entries leaving the view
stay within this budget of the entries already outside it."""

ROBUST_LADDER = (30.0, 20.0, 15.0, 12.0, 10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0)
"""Thresholds tried by ``range="robust"``, widest first. Tightening stops at the first
one that would cost more than :data:`ROBUST_COVERAGE_BUDGET`."""

ROBUST_DISCRETE_VALUES = 20
"""Number of distinct values below which a sample counts as categorical.

Counts, flags and small multiplicities have no tail to cut: every value is a
category of its own, and dropping the rarest ones loses a bin rather than empty
space. Such samples are left at :data:`ROBUST_THRESHOLD`."""


def auto_range(
    arrays: Sequence[np.ndarray],
    *,
    mode: Literal["auto", "robust"] = "auto",
) -> tuple[float, float]:
    """Choose a histogram range covering the finite values of all ``arrays``.

    ``mode="auto"`` uses the overall minimum and maximum, with the upper edge
    nudged up so the maximum value lands inside the last bin. ``mode="robust"``
    first discards outliers whose modified z-score (``0.6745 * |x - median| /
    MAD``) exceeds :data:`ROBUST_THRESHOLD`, which rejects sentinels and other
    values far from the bulk. It then tightens the threshold along
    :data:`ROBUST_LADDER` while the entries leaving the view stay within
    :data:`ROBUST_COVERAGE_BUDGET`, which cuts a long thin tail that the
    distance threshold alone keeps. Each candidate is padded by 5 percent of its
    span and clamped to the ``"auto"`` range, so the result never reaches past
    the data or past the first step. Categorical samples (fewer than
    :data:`ROBUST_DISCRETE_VALUES` distinct values) are not tightened. Use
    ``mode="auto"`` to retain the full finite extent. If MAD is zero, the mean
    absolute deviation from the median is used instead. A degenerate range (all
    values equal) is widened symmetrically; if there are no finite values at
    all, ``(0.0, 1.0)`` is returned.
    """
    values = [np.asarray(a, dtype=float).ravel() for a in arrays if len(a)]
    finite = [v[np.isfinite(v)] for v in values]
    finite = [v for v in finite if v.size]
    if not finite:
        return (0.0, 1.0)
    combined = np.concatenate(finite)
    low, high = float(combined.min()), float(combined.max())
    span = high - low
    high = high + (span * 1e-3 if span > 0 else 0.0)
    if mode == "robust":
        low, high = _robust_range(combined, low, high)
    if not high > low:
        width = abs(low) * 0.1 if low != 0 else 0.5
        low, high = low - width, high + width
    return (low, high)


def _padded(kept: np.ndarray, low: float, high: float) -> tuple[float, float]:
    """Pad the extent of ``kept`` by 5 percent of its span, clamped to ``(low, high)``.

    Clamping to the data keeps the padding from adding empty bins or giving a
    positive variable a negative lower edge, and bounds every candidate range by
    the extent of the data.
    """
    kept_low, kept_high = float(kept.min()), float(kept.max())
    pad = 0.05 * (kept_high - kept_low)
    return max(kept_low - pad, low), min(kept_high + pad, high)


def _robust_range(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    """Reject outliers, then cut a long tail as far as the coverage budget allows.

    ``(low, high)`` is the ``"auto"`` range, which bounds the result. The first
    step of :data:`ROBUST_LADDER` is :data:`ROBUST_THRESHOLD` and sets that bound;
    later steps are accepted only while they push no more than
    :data:`ROBUST_COVERAGE_BUDGET` of the entries out of the view beyond what the
    first step already did, and the first step that costs more ends the walk.
    """
    best = _padded(_reject_outliers(values, ROBUST_LADDER[0]), low, high)
    if _distinct_values_below(values, ROBUST_DISCRETE_VALUES):
        return best
    outside = float(((values < best[0]) | (values > best[1])).mean())
    for threshold in ROBUST_LADDER[1:]:
        candidate = _padded(_reject_outliers(values, threshold), low, high)
        if candidate[0] < best[0] or candidate[1] > best[1]:
            break  # an emptied selection falls back to every value: stop widening
        cost = float(((values < candidate[0]) | (values > candidate[1])).mean())
        if cost - outside > ROBUST_COVERAGE_BUDGET:
            break
        best = candidate
    return best


def _distinct_values_below(values: np.ndarray, limit: int) -> bool:
    """Whether ``values`` takes fewer than ``limit`` distinct values.

    A strided probe answers the common continuous case without sorting the whole
    array; only a sample that looks categorical is counted exactly.
    """
    step = max(1, values.size // (4 * limit))
    if np.unique(values[::step]).size >= limit:
        return False
    return bool(np.unique(values).size < limit)


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
        ``bins`` is an integer and ``range`` is ``"auto"``/``"robust"`` or left
        unset (which uses :data:`DEFAULT_RANGE`).
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
            requested = DEFAULT_RANGE if variable.range is None else variable.range
            mode: Literal["auto", "robust"] = "robust" if requested == "robust" else "auto"
            low, high = auto_range(data, mode=mode)
        return hist.axis.Regular(bins, low, high, name=name, label=label)
    if isinstance(bins, tuple) and len(bins) == 3 and isinstance(bins[0], int):
        n, low, high = bins
        return hist.axis.Regular(int(n), float(low), float(high), name=name, label=label)
    edges = _edges_from(bins)
    return hist.axis.Variable(edges, name=name, label=label)
