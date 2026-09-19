"""Filling ``Hist`` objects and the :class:`Histogram` wrapper."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import hist
import numpy as np

from rootfig._mapping import FrozenMapping
from rootfig._storage import as_weight_storage, is_category, same_axis, same_binning
from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError, SystematicError
from rootfig.histograms.stats import Summary

if TYPE_CHECKING:
    from rootfig.model.binning import Axis
    from rootfig.model.samples import HistType, Sample
    from rootfig.selection import Columns

__all__ = [
    "Histogram",
    "RebinTarget",
    "as_weight_storage",
    "compatible_binning",
    "fill",
    "from_sample",
    "mirror",
]

RebinTarget: TypeAlias = int | Sequence[float] | np.ndarray | None
"""What :meth:`Histogram.rebinned_to` makes of one axis: a bin count, the edges to merge to,
or ``None`` to keep it."""


def compatible_binning(a: Hist, b: Hist) -> bool:
    """Return True if both histograms are one-dimensional and bin the same way.

    Numeric axes must have the same edges up to round-off (a millionth of the
    smallest bin width, so bins shifted by a whole width at large coordinates
    are rejected); category axes the same categories in the same order. Flow
    bins and axis names or labels do not matter here: this decides whether two
    histograms can be stacked, summed or divided bin by bin.
    """
    if a.ndim != 1 or b.ndim != 1:
        return False
    return same_axis(a.axes[0], b.axes[0], flow=False)


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
        ``histogram.replace(variations=...)`` to replace it. The underlying
        ``hist.Hist`` objects remain mutable. Observed data (``is_data``) cannot
        carry variations, as for :class:`~rootfig.model.Sample`.
    per_object
        Whether an entry is an object rather than an event (a per-object
        variable), which words the ``Entries``/``Events`` y label.
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
    per_object: bool = False

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
        per_object: bool = False,
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
        object.__setattr__(self, "per_object", per_object)
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
            if not all(same_binning(varied, self.hist) for varied in given):
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

        A histogram that was not filled by rootfig, or that sums several (a
        group), carries no entry count: its bin contents are sums of weights,
        which only equal the number of fills for unweighted, unscaled
        histograms (see :attr:`sum_weights`).
        """
        return None if self.stats is None else self.stats.entries

    def replace(self, **changes: Any) -> Histogram:
        """Return a copy with the given fields changed, e.g. ``h.replace(label="B")``."""
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

    def rebinned(self, factor: int | Sequence[int]) -> Histogram:
        """Return a copy with every ``factor`` adjacent bins merged (per axis for a sequence).

        Contents and variances add up; the flow bins are kept. Variations are
        rebinned alike, so ratios and uncertainty bands stay consistent. Rebin
        before normalising: heights per unit width or per unit area do not add.

        Raises
        ------
        BinningError
            If the histogram is normalised, a factor is not a positive integer
            dividing the axis' bin count, an axis is categorical, or the
            sequence does not have one entry per axis.
        """
        if self.normalization is not None:
            msg = (
                f"histogram {self.label!r} is normalised ({self.normalization}); rebin before "
                "normalising"
            )
            raise BinningError(msg)
        # typed loosely: the checks below are what rejects floats, bools and strings at runtime
        given: list[Any] = list(factor) if isinstance(factor, Sequence) else [factor] * self.ndim
        if not all(_is_positive_integer(f) for f in given):
            msg = f"rebin factors must be positive integers, got {factor!r}"
            raise BinningError(msg)
        factors = [int(f) for f in given]
        if len(factors) != self.ndim:
            msg = f"got {len(factors)} rebin factors for a {self.ndim}D histogram"
            raise BinningError(msg)
        for axis, step in zip(self.hist.axes, factors, strict=True):
            if step == 1:
                continue
            if is_category(axis):
                msg = f"cannot merge the categories of axis {axis.label or axis.name!r}"
                raise BinningError(msg)
            if axis.size % step:
                divisors = [d for d in range(1, axis.size + 1) if axis.size % d == 0]
                msg = (
                    f"cannot merge bins of {axis.label or axis.name!r} in groups of {step}: "
                    f"{axis.size} bins can only be grouped by {divisors}"
                )
                raise BinningError(msg)
        if all(step == 1 for step in factors):
            return self
        selection = tuple(slice(None, None, hist.rebin(step)) for step in factors)

        def merge(h: Hist) -> Hist:
            # slicing with rebin keeps every axis, so the result is a Hist, never a float
            return cast("Hist", h[selection])

        return self.map_hists(merge)

    def rebinned_to(self, bins: RebinTarget | Sequence[RebinTarget]) -> Histogram:
        """Return a copy whose axes are merged to ``bins``: a bin count or edges per axis.

        A count merges adjacent bins as :meth:`rebinned` does, so it must divide
        the axis' bin count (the message names the counts that would). Edges
        must each coincide with an edge of the axis, the first and last with its
        ends, and the bins between two of them are merged into one; uniform
        groups keep the axis type (a ``Regular`` axis stays ``Regular``), others
        give a ``Variable`` axis. Edges that are exactly the axis' own, or its
        bin count, leave it unchanged, also when the histogram is normalised.
        ``None`` keeps an axis. A one-dimensional histogram takes the
        count or the edges directly; otherwise give one entry per axis.

        Raises
        ------
        BinningError
            If a count does not divide the axis' bin count or is not a positive
            integer, an edge is not one of the axis' edges or the first and last
            are not its ends (the range of an existing histogram is fixed), the
            sequence does not have one entry per axis, or the axis is
            categorical; and whatever :meth:`rebinned` refuses.
        """
        wanted = _targets_per_axis(bins, self.ndim)
        factors: list[int] = []
        boundaries: list[np.ndarray | None] = []
        for axis, target in zip(self.hist.axes, wanted, strict=True):
            groups: np.ndarray | None = None
            if target is None:
                factor = 1
            elif _is_edges(target):
                groups = _merge_boundaries(axis, np.asarray(target, dtype=float))
                steps = np.diff(groups)
                # uniform groups are a plain rebin, which keeps the axis type and transform
                factor, groups = (int(steps[0]), None) if np.all(steps == steps[0]) else (1, groups)
            elif not _is_positive_integer(target) or axis.size % int(target):
                possible = [axis.size // d for d in range(1, axis.size + 1) if axis.size % d == 0]
                msg = (
                    f"axis {axis.label or axis.name!r} has {axis.size} bins, which can be merged "
                    f"into {possible} bins, not {target!r}"
                )
                raise BinningError(msg)
            else:
                factor = axis.size // int(target)
            factors.append(factor)
            boundaries.append(groups)
        if all(factor == 1 for factor in factors) and all(groups is None for groups in boundaries):
            # the axes already have these bins: nothing to merge, so a normalised histogram
            # passes too (its Variable still describes it); any real merge is refused below
            return self
        result = self.rebinned(factors)
        if all(groups is None for groups in boundaries):
            return result
        return result.map_hists(lambda h: _merge_to_edges(h, boundaries))


def _is_positive_integer(value: object) -> bool:
    """Return True for ``1``, ``2``, ``numpy.int64(3)``, ...; not for bools, floats or strings."""
    return isinstance(value, int | np.integer) and not isinstance(value, bool) and int(value) >= 1


def _is_number(value: object) -> bool:
    return isinstance(value, int | float | np.number) and not isinstance(value, bool)


def _is_edges(value: object) -> bool:
    """Return True for a flat sequence of at least two numbers (bin edges, not a count)."""
    if isinstance(value, np.ndarray):
        return value.ndim == 1 and value.size >= 2 and np.issubdtype(value.dtype, np.number)
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and len(value) >= 2
        and all(_is_number(v) for v in value)
    )


def _targets_per_axis(bins: Any, ndim: int) -> list[Any]:
    """Return one :data:`RebinTarget` per axis for the ``bins`` given to :meth:`rebinned_to`."""
    if bins is None or _is_number(bins):
        return [bins] * ndim
    if ndim == 1 and _is_edges(bins):
        return [bins]
    wanted = list(bins)
    if len(wanted) != ndim:
        msg = (
            f"got {len(wanted)} bin counts for a {ndim}D histogram; give one count or edge "
            "sequence per axis"
        )
        raise BinningError(msg)
    return wanted


def _merge_boundaries(axis: Any, edges: np.ndarray) -> np.ndarray:
    """Return the positions of ``edges`` among the edges of ``axis`` (a merge of its bins).

    Every requested edge must coincide with one of the axis' edges (to a
    millionth of its smallest bin width, as :func:`~rootfig._storage.same_axis`
    compares), the first and last with its ends: the range of a histogram that
    already exists cannot change.
    """
    name = axis.label or axis.name
    if is_category(axis):
        msg = f"cannot merge the categories of axis {name!r} into bin edges"
        raise BinningError(msg)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        msg = f"bin edges must be strictly increasing, got {edges.tolist()!r}"
        raise BinningError(msg)
    own = np.asarray(axis.edges, dtype=float)
    extent = f"its {axis.size} bins run from {own[0]:g} to {own[-1]:g}"
    if not (np.isclose(edges[0], own[0]) and np.isclose(edges[-1], own[-1])):
        msg = (
            f"the range of a histogram that already exists is fixed: {extent}, and bins from "
            f"{edges[0]:g} to {edges[-1]:g} were asked for. Use xlim= to zoom, or edges that end "
            "where the axis does"
        )
        raise BinningError(msg)
    positions = np.clip(np.searchsorted(own, edges), 1, own.size - 1)
    positions = np.where(
        np.abs(own[positions - 1] - edges) < np.abs(own[positions] - edges),
        positions - 1,
        positions,
    )
    tolerance = 1e-6 * float(np.diff(own).min())
    missing = edges[np.abs(own[positions] - edges) > tolerance]
    if missing.size:
        msg = (
            f"axis {name!r} has no bin edge at {missing[0]:g}: {extent} and only its own edges "
            "can be kept when merging its bins. Fill from the tree to bin freely"
        )
        raise BinningError(msg)
    if np.any(np.diff(positions) <= 0):
        # requested edges increase, but two of them may lie within tolerance of the same
        # stored edge and merge to an empty bin
        repeated = own[positions[1:][np.diff(positions) <= 0][0]]
        msg = (
            f"two of the requested bin edges resolve to the same edge {repeated:g} of axis "
            f"{name!r}: {extent}, and each of its edges can be kept only once"
        )
        raise BinningError(msg)
    return positions


def _merge_to_edges(h: Hist, boundaries: Sequence[np.ndarray | None]) -> Hist:
    """Merge the bins of ``h`` between the ``boundaries`` of each axis (``None`` keeps an axis).

    Contents and variances add up and the flow bins are kept; a merged axis
    becomes a ``Variable`` axis with the same name, label and flow bins.
    """
    axes = []
    values = np.asarray(h.values(flow=True), dtype=float)
    variances = np.asarray(h.variances(flow=True), dtype=float)
    for index, (axis, groups) in enumerate(zip(h.axes, boundaries, strict=True)):
        if groups is None:
            axes.append(axis)
            continue
        traits = axis.traits
        axes.append(
            hist.axis.Variable(
                np.asarray(axis.edges, dtype=float)[groups],
                name=axis.name,
                label=axis.label,
                underflow=traits.underflow,
                overflow=traits.overflow,
            )
        )
        # groups are visible edge positions; in flow coordinates the underflow cell comes
        # first and the overflow cell last, each as a group of its own
        offset = 1 if traits.underflow else 0
        starts = [*([0] if traits.underflow else []), *(offset + groups[:-1])]
        if traits.overflow:
            starts.append(offset + axis.size)
        values = np.add.reduceat(values, starts, axis=index)
        variances = np.add.reduceat(variances, starts, axis=index)
    merged = hist.Hist(*axes, storage=hist.storage.Weight())
    view: Any = merged.view(flow=True)
    view.value = values
    view.variance = variances
    return merged


def from_sample(
    sample: Sample,
    hist_: Hist,
    *,
    stats: Summary | None = None,
    variations: Mapping[str, tuple[Hist, Hist | None]] | None = None,
    per_object: bool = False,
) -> Histogram:
    """Wrap ``hist_`` as the :class:`Histogram` of ``sample`` (label, data flag, drawing hints)."""
    return Histogram(
        hist=hist_,
        label=sample.label,
        sample=sample,
        stats=stats,
        is_data=sample.is_data,
        color=sample.color,
        histtype=sample.histtype,
        variations=variations,
        per_object=per_object,
    )


def mirror(nominal: Hist, up: Hist) -> Hist:
    """Return ``2 * nominal - up``, the up shift applied in the opposite direction.

    Variances are the up variation's: mirroring moves the contents, not their
    statistical precision.
    """
    down = up.copy()
    view: Any = down.view(flow=True)
    view.value = 2.0 * np.asarray(nominal.values(flow=True)) - np.asarray(up.values(flow=True))
    return down
