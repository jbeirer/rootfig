"""Filling ``Hist`` objects and the :class:`Histogram` wrapper."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import hist
import numpy as np

from rootfig._typing import FloatArray, Hist
from rootfig.errors import RootfigWarning
from rootfig.histograms.stats import Summary

if TYPE_CHECKING:
    from rootfig.model.binning import Axis
    from rootfig.model.samples import HistType, Sample
    from rootfig.selection import Columns

__all__ = ["Histogram", "as_weight_storage", "fill"]


def fill(axes: Sequence[Axis], columns: Columns) -> Hist:
    """Create a ``Hist`` with ``Weight`` storage and fill it from ``columns``.

    One axis is expected per column in ``columns.arrays``. Unweighted data is
    filled without weights, so bin variances equal bin counts.
    """
    if len(axes) != len(columns.arrays):
        msg = f"got {len(axes)} axes for {len(columns.arrays)} columns"
        raise ValueError(msg)
    histogram = hist.Hist(*axes, storage=hist.storage.Weight())
    if columns.n_entries:
        if columns.weights is None:
            histogram.fill(*columns.arrays)
        else:
            histogram.fill(*columns.arrays, weight=columns.weights)
    return histogram


def as_weight_storage(histogram: Hist) -> Hist:
    """Return ``histogram`` with ``Weight`` storage (a copy if it had another storage).

    Plain count storages (``Double``, ``Int64``, ...) carry no sum of squared
    weights; their variances are taken as ``hist`` reports them, i.e. the
    counts (Poisson) for unweighted fills. If a storage reports no variances at
    all (``hist`` does so after any weighted fill or arithmetic on a count
    storage) the contents are used as variances, a Poisson guess, and a
    :class:`~rootfig.errors.RootfigWarning` says so: the true sum of squared
    weights is lost and cannot be reconstructed.
    """
    if histogram.storage_type is hist.storage.Weight:
        return histogram
    if histogram.ndim and histogram.storage_type not in (
        hist.storage.Double,
        hist.storage.Int64,
        hist.storage.AtomicInt64,
        hist.storage.Unlimited,
    ):
        msg = (
            f"histograms with {histogram.storage_type.__name__} storage are not supported; "
            "use Weight (or a plain count) storage"
        )
        raise TypeError(msg)
    values = np.asarray(histogram.values(flow=True), dtype=float)
    reported = histogram.variances(flow=True)
    if reported is None:
        warnings.warn(
            f"histogram with {histogram.storage_type.__name__} storage was filled with weights "
            "or rescaled, so hist reports no variances; using the bin contents as variances "
            "(Poisson guess). Fill with hist.storage.Weight() to keep the sum of squared weights",
            RootfigWarning,
            stacklevel=3,
        )
        variances = values
    else:
        variances = np.asarray(reported, dtype=float)
    result = hist.Hist(*histogram.axes, storage=hist.storage.Weight())
    view: Any = result.view(flow=True)
    view.value = values
    view.variance = variances
    return result


@dataclass(frozen=True)
class Histogram:
    """A filled histogram together with its provenance and drawing hints.

    The underlying :class:`Hist` (``Weight`` storage, so bin variances
    are the sums of squared weights) is available as :attr:`hist`; everything
    else is metadata used for legends, ratios and statistics.

    Attributes
    ----------
    hist
        The histogram itself (1D or 2D).
    label
        Legend label.
    sample
        The :class:`~rootfig.model.Sample` this was filled from, if any.
    stats
        Unbinned :class:`~rootfig.histograms.Summary` statistics of the filled
        values, if available.
    is_data
        Whether this represents observed data.
    color, histtype
        Drawing hints, ``None`` for style defaults.
    normalization
        Description of the normalisation applied (``None`` for raw counts).
    """

    hist: Hist
    label: str
    sample: Sample | None = None
    stats: Summary | None = None
    is_data: bool = False
    color: str | None = None
    histtype: HistType | None = None
    normalization: str | None = None

    def __post_init__(self) -> None:
        # Keep the documented invariant for histograms built by users from plain hist.Hist
        # objects; rootfig's own histograms already have Weight storage (no copy is made).
        object.__setattr__(self, "hist", as_weight_storage(self.hist))

    # -- convenience accessors -----------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of axes."""
        return int(self.hist.ndim)

    @property
    def axis(self) -> Any:
        """The first axis."""
        return self.hist.axes[0]

    @property
    def edges(self) -> FloatArray:
        """Bin edges of the first axis."""
        return np.asarray(self.hist.axes[0].edges, dtype=float)

    @property
    def centers(self) -> FloatArray:
        """Bin centres of the first axis."""
        return np.asarray(self.hist.axes[0].centers, dtype=float)

    @property
    def widths(self) -> FloatArray:
        """Bin widths of the first axis."""
        return np.asarray(self.hist.axes[0].widths, dtype=float)

    def values(self, *, flow: bool = False) -> FloatArray:
        """Bin contents (sum of weights)."""
        return np.asarray(self.hist.values(flow=flow), dtype=float)

    def variances(self, *, flow: bool = False) -> FloatArray:
        """Bin variances (sum of squared weights)."""
        return np.asarray(self.hist.variances(flow=flow), dtype=float)

    def errors(self, *, flow: bool = False) -> FloatArray:
        """Bin uncertainties, ``sqrt(variances)``."""
        return np.asarray(np.sqrt(self.variances(flow=flow)), dtype=float)

    @property
    def integral(self) -> float:
        """Sum of the visible bin contents (flow bins excluded)."""
        return float(self.values().sum())

    @property
    def sum_weights(self) -> float:
        """Sum of all bin contents including the flow bins."""
        return float(np.sum(self.hist.values(flow=True)))

    @property
    def underflow(self) -> float:
        """Content of the underflow bin (first axis, 1D only; ``0`` if the axis has none)."""
        return self._flow_cell("underflow", self.hist.values(flow=True))

    @property
    def overflow(self) -> float:
        """Content of the overflow bin (first axis, 1D only; ``0`` if the axis has none)."""
        return self._flow_cell("overflow", self.hist.values(flow=True))

    @property
    def underflow_variance(self) -> float:
        """Variance of the underflow bin (first axis, 1D only; ``0`` if the axis has none)."""
        return self._flow_cell("underflow", self.hist.variances(flow=True))

    @property
    def overflow_variance(self) -> float:
        """Variance of the overflow bin (first axis, 1D only; ``0`` if the axis has none)."""
        return self._flow_cell("overflow", self.hist.variances(flow=True))

    def _flow_cell(self, side: str, cells: Any) -> float:
        if self.ndim != 1:
            return float("nan")
        traits = self.axis.traits
        if not getattr(traits, side):
            return 0.0
        return float(cells[0 if side == "underflow" else -1])

    @property
    def entries(self) -> int | None:
        """Number of filled entries from the unbinned statistics, or ``None`` if unknown.

        A histogram that was not filled by rootfig carries no entry count: its
        bin contents are sums of weights, which only equal the number of fills
        for unweighted, unscaled histograms (see :attr:`sum_weights`).
        """
        return None if self.stats is None else self.stats.entries

    def with_(self, **changes: Any) -> Histogram:
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)

    def scaled(self, factor: float) -> Histogram:
        """Return a copy multiplied by ``factor``.

        Variances scale with ``factor**2``. The statistics' sum of weights (and
        sum of squared weights) scale along; the moments, entry count and
        effective entries are unchanged by a uniform rescaling.
        """
        stats = self.stats
        if stats is not None:
            stats = replace(
                stats,
                sum_weights=stats.sum_weights * factor,
                _sum_w2=stats._sum_w2 * factor**2,
            )
        return replace(self, hist=self.hist * factor, stats=stats)
