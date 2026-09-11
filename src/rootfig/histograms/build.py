"""Filling ``Hist`` objects and the :class:`Histogram` wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import hist
import numpy as np

from rootfig._typing import FloatArray, Hist
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
    all (a ``Double`` histogram filled with weights) the contents are used as
    variances, the best available guess, and the caller is not told otherwise
    because ``hist`` itself makes the same assumption when drawing.
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
    variances = values if reported is None else np.asarray(reported, dtype=float)
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
    def entries(self) -> int:
        """Number of filled entries, from the statistics if available."""
        if self.stats is not None:
            return self.stats.entries
        if self.normalization is not None:
            return 0
        return int(np.rint(float(np.sum(self.hist.values(flow=True)))))

    def with_(self, **changes: Any) -> Histogram:
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)

    def scaled(self, factor: float) -> Histogram:
        """Return a copy multiplied by ``factor`` (variances scale with ``factor**2``)."""
        return replace(self, hist=self.hist * factor)
