"""Binning specifications and their resolution into ``hist`` axes."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import boost_histogram as bh
import hist
import numpy as np

from rootfig.errors import BinningError

if TYPE_CHECKING:
    from rootfig.model.variables import Variable

__all__ = [
    "DEFAULT_BINS",
    "DEFAULT_RANGE",
    "ROBUST_COVERAGE_BUDGET",
    "ROBUST_DISCRETE_VALUES",
    "ROBUST_LADDER",
    "ROBUST_THRESHOLD",
    "Axis",
    "Bins",
    "MergeTarget",
    "RangeSpec",
    "auto_range",
    "log_bins",
    "merge_target",
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
  so it never reaches past the data. Each sample is charged separately for the
  entries and the weight a cut would take off its axis, and a sample with fewer
  than :data:`ROBUST_DISCRETE_VALUES` distinct values keeps every value that
  survived the outlier rejection. Values outside land in the under/overflow.
  Degenerate ranges are widened symmetrically.
* ``"auto"`` - the finite minimum and maximum over all samples.
"""

DEFAULT_RANGE: RangeSpec = "robust"
"""Range inference used when a :class:`~rootfig.model.Variable` does not ask for one."""

DEFAULT_BINS: int = 50
"""Number of bins used to fill from a tree when a :class:`~rootfig.model.Variable` names none
(``bins=None``). A histogram stored in a file keeps its own binning instead."""

MergeTarget: TypeAlias = int | np.ndarray | tuple[float, float] | None
"""What the axis of a histogram that already exists is merged to: a bin count, the edges to
end up with, a crop range whose ends are existing edges, or ``None`` to keep it
(see :func:`merge_target`)."""


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


def validate_bins(bins: Bins | None, range_: RangeSpec) -> None:
    """Raise :class:`BinningError` if ``bins``/``range_`` are not a valid specification.

    ``None`` stands for :data:`DEFAULT_BINS` and is validated as that count.
    """
    if bins is None:
        bins = DEFAULT_BINS
    if isinstance(bins, hist.axis.Regular | hist.axis.Variable):
        return
    if isinstance(cast("object", bins), bh.axis.Axis):
        msg = (
            f"bins={bins!r} cannot be used: the axes rootfig bins with are "
            "hist.axis.Regular and hist.axis.Variable; give one of those, (n, low, high) or "
            "edges instead, e.g. bins=(6, 0, 6) for the integers 0 to 5"
        )
        raise BinningError(msg)
    if isinstance(bins, bool):
        msg = "bins must be an int, (n, low, high), edges, or a hist axis, got a bool"
        raise BinningError(msg)
    if isinstance(bins, int | np.integer):
        if bins < 1:
            msg = f"number of bins must be positive, got {bins}"
            raise BinningError(msg)
        if isinstance(range_, tuple):
            _validate_range(range_)
        elif range_ not in ("auto", "robust", None):
            msg = f"range must be (low, high), 'auto', or 'robust', got {range_!r}"
            raise BinningError(msg)
        return
    if isinstance(bins, tuple) and len(bins) == 3 and isinstance(bins[0], int | np.integer):
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


def merge_target(bins: Bins | None, range_: RangeSpec = None) -> MergeTarget:
    """Interpret a binning specification for a histogram that already exists.

    An integer without an explicit range merges the whole axis to that count.
    Explicit edges, including ``(n, low, high)`` and numeric axes, crop and merge
    to those edges. A range without bins crops to its ends and keeps the bins
    between them; ``None`` without an explicit range keeps the axis. Requested
    edges must coincide with existing edges, as checked by
    :meth:`~rootfig.histograms.Histogram.rebinned_to`. Cropped content belongs
    in the flow bins, as when filling a tree with the same Variable.

    Raises
    ------
    BinningError
        If the specification is invalid.
    """
    validate_bins(bins, range_)
    if bins is None:
        return (float(range_[0]), float(range_[1])) if isinstance(range_, tuple) else None
    if isinstance(bins, hist.axis.Regular | hist.axis.Variable):
        return np.asarray(bins.edges, dtype=float)
    if isinstance(bins, int | np.integer):
        if isinstance(range_, tuple):
            return np.linspace(range_[0], range_[1], int(bins) + 1)
        return int(bins)
    if isinstance(bins, tuple) and len(bins) == 3 and isinstance(bins[0], int | np.integer):
        n, low, high = bins
        return np.linspace(float(low), float(high), int(n) + 1)
    return _edges_from(bins)


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

ROBUST_COVERAGE_BUDGET = 0.01
"""Fraction of a sample ``range="robust"`` may move out of the view to cut a tail.

:data:`ROBUST_THRESHOLD` measures a distance from the bulk, so a tail that reaches
far but thins out smoothly stays inside it and leaves the interesting part of the
distribution in a small corner of the axis. Starting from that range, the threshold
is tightened along :data:`ROBUST_LADDER` for as long as the entries leaving the view
stay within this budget of the entries already outside it.

Every sample is charged for its own losses, never the pooled entries: one binning
is shared by all samples of a plot, and a small sample far from a large one stays a
small fraction of the pooled values however much of it is cut."""

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
    weights: Sequence[np.ndarray | None] | None = None,
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
    the data or past the first step. The budget is charged per sample, against
    entries and weight alike, and a categorical sample (fewer than
    :data:`ROBUST_DISCRETE_VALUES` distinct values) gets none, so the tightening
    takes no value off its axis - the outlier rejection still applies to it. Use
    ``mode="auto"`` to retain the full finite extent. If MAD is zero, the mean
    absolute deviation from the median is used instead. A degenerate range (all
    values equal) is widened symmetrically; if there are no finite values at
    all, ``(0.0, 1.0)`` is returned.

    Parameters
    ----------
    arrays
        Flat arrays of values, one per sample.
    mode
        ``"auto"`` for the full finite extent, ``"robust"`` to reject outliers
        and cut a thin tail.
    weights
        Fill weights aligned with ``arrays`` (``None`` per sample, or in place
        of the whole sequence, for unweighted data). ``"robust"`` charges its
        budget against the weight a cut would take off the axis as well as the
        entries, so rare high-weight entries are not treated as negligible.
    """
    if weights is None:
        weights = [None] * len(arrays)
    paired = [
        (np.asarray(a, dtype=float).ravel(), None if w is None else np.asarray(w).ravel())
        for a, w in zip(arrays, weights, strict=True)
        if len(a)
    ]
    kept_weights: list[np.ndarray | None] = []
    finite: list[np.ndarray] = []
    for v, w in paired:
        mask = np.isfinite(v)
        if not mask.any():
            continue
        finite.append(v[mask])
        kept_weights.append(None if w is None else w[mask])
    if not finite:
        return (0.0, 1.0)
    combined = np.concatenate(finite)
    low, high = float(combined.min()), float(combined.max())
    span = high - low
    high = high + (span * 1e-3 if span > 0 else 0.0)
    if mode == "robust":
        low, high = _robust_range(finite, kept_weights, combined, low, high)
    if not high > low:
        width = abs(low) * 0.1 if low != 0 else 0.5
        low, high = low - width, high + width
    return (low, high)


def _padded(kept_low: float, kept_high: float, low: float, high: float) -> tuple[float, float]:
    """Pad ``(kept_low, kept_high)`` by 5 percent of its span, clamped to ``(low, high)``.

    Clamping to the data keeps the padding from adding empty bins or giving a
    positive variable a negative lower edge, and bounds every candidate range by
    the extent of the data.
    """
    pad = 0.05 * (kept_high - kept_low)
    return max(kept_low - pad, low), min(kept_high + pad, high)


def _retained_extent(samples: Sequence[np.ndarray]) -> tuple[float, float]:
    """Span every sample keeps once each has had its own outliers rejected.

    The rejection is per sample and the extents are unioned, so a sample is
    measured against its own median and MAD. Judged against the pooled values a
    sample that simply sits somewhere else - a signal offset from a background,
    either of them normalised - scores as one big outlier and is dropped whole,
    the more easily the more the other sample outnumbers it. Its own spread is
    the honest scale to ask whether one of its entries is an outlier, and a
    sentinel is still far from the bulk of the sample it appears in.
    """
    extents = [_keep_within(v, _modified_z_scores(v), ROBUST_LADDER[0]) for v in samples]
    return min(float(k.min()) for k in extents), max(float(k.max()) for k in extents)


def _outside(
    samples: Sequence[np.ndarray],
    weights: Sequence[np.ndarray | None],
    low: float,
    high: float,
) -> np.ndarray:
    """Measure what ``(low, high)`` leaves out of the view, per sample.

    Row 0 is the fraction of each sample's entries, row 1 the fraction of its
    total ``|weight|``. A rare entry can carry a large share of a histogram's
    content, so both have to stay within budget for a cut to be cheap.

    The upper edge is exclusive, as it is on the axis this range becomes, so a
    value equal to ``high`` counts as out of the view rather than in it.
    """
    entries, content = [], []
    for values, weight in zip(samples, weights, strict=True):
        gone = (values < low) | (values >= high)
        entries.append(float(gone.mean()))
        if weight is None:
            content.append(entries[-1])
        else:
            magnitude = np.abs(weight)
            total = float(magnitude.sum())
            content.append(float(magnitude[gone].sum() / total) if total > 0 else entries[-1])
    return np.array([entries, content])


def _robust_range(
    samples: Sequence[np.ndarray],
    weights: Sequence[np.ndarray | None],
    values: np.ndarray,
    low: float,
    high: float,
) -> tuple[float, float]:
    """Reject outliers, then cut a long tail as far as the coverage budget allows.

    ``(low, high)`` is the ``"auto"`` range, which bounds the result. The first
    step of :data:`ROBUST_LADDER` is :data:`ROBUST_THRESHOLD` and sets that bound;
    later steps are accepted only while they push no more than
    :data:`ROBUST_COVERAGE_BUDGET` out of the view beyond what the first step
    already did, and the first step that costs more ends the walk.

    The budget is spent per sample, not over the pooled values. One binning is
    shared by every sample of a plot, and a small sample far from a large one is
    a small *fraction* of the pooled entries however much of it is cut: measured
    that way, a signal of a few hundred entries beside a background of a hundred
    thousand could be moved into the overflow in its entirety and still look
    cheap. Charging each sample for its own losses costs such a cut the whole
    sample, so the walk stops before it.

    A categorical sample gets a budget of zero, so no step of the walk may take a
    value off its axis; the outlier rejection that set ``best`` still applies to
    it, and can have rejected a sufficiently distant category already. This too
    is decided per sample: pooled with a continuous one it would look
    continuous, and its rarest category - a single entry among a thousand -
    would be cheap enough to cut.

    Both the entries and the ``|weight|`` they carry have to stay within the
    budget, since a handful of high-weight entries can be most of what a
    histogram draws while being a rounding error in the count.
    """
    # The ladder scores the pooled values; with one sample those are its own scores.
    scores = _modified_z_scores(values)
    if len(samples) == 1:
        only = _keep_within(values, scores, ROBUST_LADDER[0])
        extent = (float(only.min()), float(only.max()))
    else:
        extent = _retained_extent(samples)
    best = _padded(*extent, low, high)
    budgets = np.array(
        [
            0.0 if _distinct_values_below(v, ROBUST_DISCRETE_VALUES) else ROBUST_COVERAGE_BUDGET
            for v in samples
        ]
    )
    if not budgets.any():  # nothing may be cut: no walk to take
        return best
    outside = _outside(samples, weights, *best)
    for threshold in ROBUST_LADDER[1:]:
        kept = _keep_within(values, scores, threshold)
        candidate = _padded(float(kept.min()), float(kept.max()), low, high)
        if candidate[0] < best[0] or candidate[1] > best[1]:
            break  # an emptied selection falls back to every value: stop widening
        if np.any(_outside(samples, weights, *candidate) - outside > budgets):
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


def _modified_z_scores(values: np.ndarray) -> np.ndarray | None:
    """Score each value by its distance from the median in units of the MAD.

    ``None`` means the sample has no scale to measure against and nothing can be
    rejected. The scores do not depend on a threshold, so the ladder computes
    them once and compares the same array against each of its steps.
    """
    median = np.median(values)
    deviation = np.abs(values - median)
    mad = np.median(deviation)
    if mad <= 0:
        # More than half the values are identical (e.g. all zero): fall back to the
        # mean absolute deviation, and if that is zero too, keep everything.
        mad = float(deviation.mean())
        if mad <= 0:
            return None
    return np.asarray(0.6745 * deviation / mad, dtype=float)


def _reject_outliers(values: np.ndarray, threshold: float = ROBUST_THRESHOLD) -> np.ndarray:
    """Drop the values scoring above ``threshold``, keeping all of them if none is left."""
    return _keep_within(values, _modified_z_scores(values), threshold)


def _keep_within(values: np.ndarray, scores: np.ndarray | None, threshold: float) -> np.ndarray:
    if scores is None:
        return values
    kept = values[scores <= threshold]
    return kept if kept.size else values


def resolve_axis(
    variable: Variable,
    data: Sequence[np.ndarray] = (),
    *,
    name: str = "x",
    weights: Sequence[np.ndarray | None] | None = None,
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
    weights
        Fill weights aligned with ``data``, passed on to :func:`auto_range` so
        an inferred range accounts for the content a cut would remove.

    Raises
    ------
    BinningError
        If the range must be inferred but no data was given.
    """
    bins = DEFAULT_BINS if variable.bins is None else variable.bins
    label = variable.axis_label
    if isinstance(bins, hist.axis.Regular | hist.axis.Variable):
        # Copy so the caller's axis (possibly shared between variables) is never modified;
        # the copy keeps edges, transform and flow traits. A shallow copy shares the
        # metadata dict on older boost-histogram releases, so copy deeply.
        axis = copy.deepcopy(bins)
        if not axis.label:
            axis.label = label
        return axis
    if isinstance(bins, int | np.integer) and not isinstance(bins, bool):
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
            low, high = auto_range(data, mode=mode, weights=weights)
        return hist.axis.Regular(int(bins), low, high, name=name, label=label)
    if isinstance(bins, tuple) and len(bins) == 3 and isinstance(bins[0], int | np.integer):
        n, low, high = bins
        return hist.axis.Regular(int(n), float(low), float(high), name=name, label=label)
    edges = _edges_from(bins)
    return hist.axis.Variable(edges, name=name, label=label)
