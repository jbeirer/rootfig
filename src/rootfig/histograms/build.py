"""Filling ``Hist`` objects and the :class:`Histogram` wrapper."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import hist
import numpy as np

from rootfig._mapping import FrozenMapping
from rootfig._typing import FloatArray, Hist
from rootfig.errors import RootfigWarning, SystematicError
from rootfig.histograms.stats import Summary

if TYPE_CHECKING:
    from rootfig.model.binning import Axis
    from rootfig.model.samples import HistType, Sample
    from rootfig.selection import Columns

__all__ = ["Histogram", "as_weight_storage", "compatible_binning", "fill", "mirror"]


def compatible_binning(a: Hist, b: Hist) -> bool:
    """Return True if both histograms are one-dimensional with identical edges.

    Edges may differ by round-off only: the tolerance is a millionth of the
    smallest bin width, so bins shifted by a whole width at large coordinates
    (where NumPy's default relative tolerance would accept them) are rejected.
    """
    if a.ndim != 1 or b.ndim != 1:
        return False
    ea, eb = np.asarray(a.axes[0].edges, dtype=float), np.asarray(b.axes[0].edges, dtype=float)
    if ea.shape != eb.shape:
        return False
    tolerance = 1e-6 * float(min(np.diff(ea).min(), np.diff(eb).min()))
    return bool(np.allclose(ea, eb, rtol=0.0, atol=tolerance))


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


def as_weight_storage(histogram: Hist, *, assume_poisson: bool = False) -> Hist:
    """Return ``histogram`` with ``Weight`` storage (a copy if it had another storage).

    Plain count storages (``Double``, ``Int64``, ...) carry no sum of squared
    weights; their variances are taken as ``hist`` reports them, i.e. the
    counts (Poisson) for unweighted fills. After a weighted fill or arithmetic
    on such a storage ``hist`` reports no variances at all: the sum of squared
    weights is lost and cannot be reconstructed. That is an error unless
    ``assume_poisson=True``, which uses the absolute bin contents as variances
    (the Poisson guess; a :class:`~rootfig.errors.RootfigWarning` says so).

    Raises
    ------
    TypeError
        If the storage is not a count or ``Weight`` storage (``Mean``, ...).
    ValueError
        If the histogram reports no variances and ``assume_poisson`` is False.
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
        what = (
            f"histogram with {histogram.storage_type.__name__} storage was filled with weights "
            "or rescaled, so hist reports no variances (the sum of squared weights is lost)"
        )
        if not assume_poisson:
            msg = (
                f"{what}. Fill it with hist.storage.Weight() to keep the uncertainties, or pass "
                "assume_poisson=True to use the absolute bin contents as variances"
            )
            raise ValueError(msg)
        warnings.warn(
            f"{what}; using the absolute bin contents as variances (Poisson guess)",
            RootfigWarning,
            stacklevel=3,
        )
        variances = np.abs(values)  # never a negative variance for signed contents
    else:
        variances = np.asarray(reported, dtype=float)
    result = hist.Hist(*histogram.axes, storage=hist.storage.Weight())
    view: Any = result.view(flow=True)
    view.value = values
    view.variance = variances
    return result


@dataclass(frozen=True, init=False)
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
    variations
        Systematic variations, ``{name: (up, down)}`` histograms with the binning
        of :attr:`hist`. A ``down`` given as ``None`` is filled in by mirroring the
        up shift around the nominal contents. Summarised by
        :func:`~rootfig.histograms.uncertainty`. The mapping is copied and made
        read-only; every stored pair contains two histograms. Use
        ``histogram.with_(variations=...)`` to replace it. The underlying
        ``hist.Hist`` objects remain mutable. Observed data (``is_data``) cannot
        carry variations, as for :class:`~rootfig.model.Sample`.
    """

    hist: Hist
    label: str
    sample: Sample | None = None
    stats: Summary | None = None
    is_data: bool = False
    color: str | None = None
    histtype: HistType | None = None
    normalization: str | None = None
    variations: Mapping[str, tuple[Hist, Hist]] = field(default_factory=dict)

    def __init__(  # noqa: PLR0917 - preserve the positional dataclass constructor API
        self,
        hist: Hist,
        label: str,
        sample: Sample | None = None,
        stats: Summary | None = None,
        is_data: bool = False,
        color: str | None = None,
        histtype: HistType | None = None,
        normalization: str | None = None,
        variations: Mapping[str, tuple[Hist, Hist | None]] | None = None,
    ) -> None:
        object.__setattr__(self, "hist", hist)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "sample", sample)
        object.__setattr__(self, "stats", stats)
        object.__setattr__(self, "is_data", is_data)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "histtype", histtype)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "variations", {} if variations is None else variations)
        self.__post_init__()

    def __post_init__(self) -> None:
        # Also validate subclasses that use a generated dataclass constructor.
        # Weight storage histograms are retained without copying their contents.
        object.__setattr__(self, "hist", as_weight_storage(self.hist))
        checked = self._checked_variations(self.variations)
        object.__setattr__(self, "variations", FrozenMapping(checked))

    def _checked_variations(
        self, variations: Mapping[str, tuple[Hist, Hist | None]]
    ) -> dict[str, tuple[Hist, Hist]]:
        if self.is_data and variations:
            msg = (
                f"histogram {self.label!r} is observed data and cannot carry systematic "
                f"variations ({sorted(variations)}); attach them to the simulated histograms"
            )
            raise SystematicError(msg)
        checked: dict[str, tuple[Hist, Hist]] = {}
        for name, pair in variations.items():
            if not isinstance(name, str) or not name.strip():
                msg = f"histogram {self.label!r}: variation names must be non-empty strings"
                raise SystematicError(msg)
            if not (isinstance(pair, tuple | list) and len(pair) == 2 and pair[0] is not None):
                msg = f"histogram {self.label!r}: variation {name!r} must be an (up, down) pair"  # type: ignore[unreachable]
                raise SystematicError(msg)
            given = [as_weight_storage(h, assume_poisson=True) for h in pair if h is not None]
            if not all(_same_edges(varied, self.hist) for varied in given):
                msg = (
                    f"histogram {self.label!r}: variation {name!r} does not have the binning "
                    "of the nominal histogram"
                )
                raise SystematicError(msg)
            up = given[0]
            checked[name] = (up, mirror(self.hist, up) if pair[1] is None else given[1])
        return checked

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

    def map_hists(self, transform: Callable[[Hist], Hist]) -> Histogram:
        """Return a copy with ``transform`` applied to the nominal histogram and every variation."""
        variations = {
            name: (transform(up), transform(down)) for name, (up, down) in self.variations.items()
        }
        return replace(self, hist=transform(self.hist), variations=variations)

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
        return replace(self.map_hists(lambda h: h * factor), stats=stats)


def _same_edges(a: Hist, b: Hist) -> bool:
    if a.ndim != b.ndim:
        return False
    for axis_a, axis_b in zip(a.axes, b.axes, strict=True):
        edges_a, edges_b = np.asarray(axis_a.edges), np.asarray(axis_b.edges)
        if edges_a.shape != edges_b.shape:
            return False
        tolerance = 1e-6 * float(min(np.diff(edges_a).min(), np.diff(edges_b).min()))
        if not np.allclose(edges_a, edges_b, rtol=0.0, atol=tolerance):
            return False
        if (axis_a.traits.underflow, axis_a.traits.overflow) != (
            axis_b.traits.underflow,
            axis_b.traits.overflow,
        ):
            return False
    return True


def mirror(nominal: Hist, up: Hist) -> Hist:
    """Return ``2 * nominal - up``, the up shift applied in the opposite direction.

    Variances are the up variation's: mirroring moves the contents, not their
    statistical precision.
    """
    down = up.copy()
    view: Any = down.view(flow=True)
    view.value = 2.0 * np.asarray(nominal.values(flow=True)) - np.asarray(up.values(flow=True))
    return down
