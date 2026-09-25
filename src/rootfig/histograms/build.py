"""Filling ``Hist`` objects and the :class:`Histogram` wrapper."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import boost_histogram as bh
import hist
import numpy as np

from rootfig._mapping import FrozenMapping
from rootfig._storage import as_weight_storage, is_category, same_axis, same_binning
from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError, SystematicError
from rootfig.histograms.intervals import count_problem, count_scale, poisson_errors
from rootfig.histograms.stats import Summary
from rootfig.model.binning import Bins, RangeSpec, merge_target

if TYPE_CHECKING:
    from rootfig.model.binning import Axis
    from rootfig.model.samples import HistType, Sample
    from rootfig.selection import Columns

__all__ = [
    "Histogram",
    "as_weight_storage",
    "compatible_binning",
    "fill",
    "from_sample",
    "mirror",
    "negative_bins",
]


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


def negative_bins(axis: Axis, columns: Columns) -> np.ndarray:
    """Flag the visible bins of ``axis`` that an entry of ``columns`` with a negative weight fills.

    What the sums of a filled histogram cannot always tell: a bin whose
    negative weights are outweighed still sums like one without them.
    """
    if columns.weights is None:
        return np.zeros(axis.size, dtype=bool)
    negative = columns.weights < 0
    counts: Hist = hist.Hist(axis).fill(columns.values[negative])
    return np.asarray(counts.values() > 0, dtype=bool)


def _one_count(histogram: Hist) -> Hist:
    """Return one unit count in every cell of ``histogram``, unit-weight counts.

    Transformed along with the contents (scaled, normalised, merged, moved),
    its ``variance / value`` stays the factor of a count in every cell, also
    where the contents are empty (see :func:`~rootfig.histograms.intervals.count_scale`).
    """
    unit = histogram.copy()
    view: Any = unit.view(flow=True)
    view.value = 1.0
    view.variance = 1.0
    return unit


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
        variable), which words the ``Entries``/``Events`` y label. Follows
        ``stats`` unless given.
    poisson
        Whether the statistical uncertainty is the Poisson (Garwood) interval
        of the counts rather than ``sqrt(variances)`` (see :meth:`errors`):
        ROOT's ``TH1::kPoisson``. Needs unit-weight counts (whole numbers equal
        to their variances); anything else raises ``ValueError``, since sums of
        weights cannot say whether they are counts. Scaling, normalising or
        rebinning such a histogram keeps the interval, scaled like the contents
        (ROOT falls back to ``sqrt(variances)`` once a histogram is scaled), so
        counts scaled by ``c`` are ``Histogram(counts, poisson=True).scaled(c)``.
        ``plot(data_errors=...)`` sets it for observed data.
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
    poisson: bool = False
    # one count per cell, transformed with the contents: the Poisson factor of empty cells
    _unit: Hist | None = field(default=None, compare=False, repr=False)

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
        per_object: bool | None = None,
        poisson: bool = False,
        _unit: Hist | None = None,
    ) -> None:
        if per_object is None:  # not given: follow the statistics
            per_object = stats is not None and stats.per_object
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
        object.__setattr__(self, "poisson", poisson)
        object.__setattr__(self, "_unit", _unit)
        self.__post_init__()

    def __post_init__(self) -> None:
        # Also validate subclasses that use a generated dataclass constructor.
        # Weight storage histograms are retained without copying their contents.
        object.__setattr__(self, "hist", as_weight_storage(self.hist))
        checked = self._checked_variations(self.variations)
        object.__setattr__(self, "variations", FrozenMapping(checked))
        if self.poisson:
            if self._unit is None:  # new contents: only unit-weight counts are known counts
                problem = count_problem(self.values(flow=True), self.variances(flow=True))
                if problem is not None:
                    msg = (
                        f"histogram {self.label!r} {problem}, so it is not known to hold counts "
                        "and gets no Poisson interval; keep poisson=False for sqrt(sum of "
                        "squared weights), or pass the counts with poisson=True and scale them "
                        "with .scaled()"
                    )
                    raise ValueError(msg)
                object.__setattr__(self, "_unit", _one_count(self.hist))
            elif not same_binning(self._unit, self.hist):
                msg = f"histogram {self.label!r}: its count scale does not have its binning"
                raise BinningError(msg)
            elif np.any(self.values(flow=True) < 0):
                msg = (
                    f"histogram {self.label!r} has negative contents, so its uncertainty is "
                    "not the Poisson interval of counts"
                )
                raise ValueError(msg)
        else:
            object.__setattr__(self, "_unit", None)

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

    def errors(self, *, flow: bool = False) -> tuple[FloatArray, FloatArray]:
        """Statistical uncertainty below and above the contents, ``(down, up)``.

        Both are ``sqrt(variances)``, the uncertainty of a sum of weights, or,
        with :attr:`poisson`, the distances to the Garwood 68 % interval of the
        counts, scaled like the contents (see
        :func:`~rootfig.histograms.intervals.poisson_errors`). The factor of
        each bin comes from a record of one count per bin that goes through
        every scaling, normalisation and rebinning with the contents, never from
        the contents themselves, so an empty histogram scaled by 3 gets
        ``0 +5.52`` and an empty bin divided by its width its own share. The
        pair is matplotlib's ``yerr`` order; drawing, comparisons and
        :func:`~rootfig.histograms.uncertainty` take it from here.
        """
        if self.poisson and self._unit is not None:
            unit = self._unit
            scale = count_scale(unit.values(flow=flow), np.asarray(unit.variances(flow=flow)))
            return poisson_errors(self.values(flow=flow), scale)
        sigma = np.asarray(np.sqrt(self.variances(flow=flow)), dtype=float)
        return sigma, sigma.copy()

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
        """Return a copy with the given fields changed, e.g. ``h.replace(label="B")``.

        A new ``hist`` brings its own counts: the record of one count per bin
        that the Poisson interval uses is rebuilt from it (use
        :meth:`map_hists` to transform the contents and keep that record).
        """
        if "hist" in changes and "_unit" not in changes:
            changes["_unit"] = None
        return replace(self, **changes)

    def map_hists(self, transform: Callable[[Hist], Hist]) -> Histogram:
        """Return a copy with ``transform`` applied to the nominal histogram and every variation."""
        variations = {
            name: (transform(up), transform(down)) for name, (up, down) in self.variations.items()
        }
        unit = None if self._unit is None else transform(self._unit)
        return replace(self, hist=transform(self.hist), variations=variations, _unit=unit)

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
                f"histogram {self.label!r} is normalised ({self.normalization}); "
                "crop and rebin before normalising"
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

    def rebinned_to(
        self,
        bins: Bins | Sequence[Bins | None] | None,
        range: RangeSpec | Sequence[RangeSpec] = None,
    ) -> Histogram:
        """Crop and merge axes using the same ``bins`` and ``range`` as a Variable.

        For 1D, give one binning and range specification. For multiple axes,
        give one of each per axis; a single ``None`` or integer bin count, and a
        single range (``None``, a string or ``(low, high)``), apply to every
        axis. A range without bins keeps the existing bins between its ends.
        Requested edges, range ends included, must coincide with existing
        edges. Cropped values and variances
        join the flow bins, and variations follow the same transformation.
        Uniform merges preserve Regular axes and their transforms; uneven
        merges produce Variable axes. Unbinned statistics describe the entries
        and are kept. Asking for the existing bins returns ``self``.

        Raises
        ------
        BinningError
            If edges do not coincide, a count does not divide the axis size,
            specifications do not match the dimensionality, a categorical axis
            is merged, a cropped side lacks its flow bin, or a histogram is
            normalised. Crop and rebin before normalising.
        """
        wanted = _targets_per_axis(bins, self.ndim)
        ranges = _ranges_per_axis(range, self.ndim)
        crops = []
        merges = []
        changed = False
        for axis, spec, range_ in zip(self.hist.axes, wanted, ranges, strict=True):
            target = merge_target(spec, cast("RangeSpec", range_))
            start, stop, factor = 0, axis.size, 1
            steps = None
            if isinstance(target, int):
                if axis.size % target:
                    possible = [
                        axis.size // n
                        for n in builtins.range(1, axis.size + 1)
                        if axis.size % n == 0
                    ]
                    msg = (
                        f"axis {axis.label or axis.name!r} has {axis.size} bins, which can be "
                        f"merged into {possible} bins, not {target!r}"
                    )
                    raise BinningError(msg)
                factor = axis.size // target
            elif target is not None:
                positions = _merge_boundaries(axis, np.asarray(target, dtype=float))
                start, stop = int(positions[0]), int(positions[-1])
                if not isinstance(target, tuple):
                    steps = np.diff(positions)
                    if np.all(steps == steps[0]):
                        factor, steps = int(steps[0]), None
            axis_changed = start != 0 or stop != axis.size or factor != 1 or steps is not None
            if axis_changed:
                name = axis.label or axis.name
                if is_category(axis):
                    msg = f"cannot merge the categories of axis {name!r}"
                    raise BinningError(msg)
                for cropped, side in ((start > 0, "underflow"), (stop < axis.size, "overflow")):
                    if cropped and not getattr(axis.traits, side):
                        msg = f"cannot crop axis {name!r}: its {side} bin is missing"
                        raise BinningError(msg)
                if self.normalization is not None:
                    msg = (
                        f"histogram {self.label!r}, axis {name!r}, is normalised "
                        f"({self.normalization}); crop and rebin before normalising"
                    )
                    raise BinningError(msg)
            changed |= axis_changed
            crops.append(slice(start, stop) if start != 0 or stop != axis.size else slice(None))
            merge = hist.rebin(groups=steps.tolist()) if steps is not None else hist.rebin(factor)
            merges.append(
                slice(None, None, merge) if steps is not None or factor != 1 else slice(None)
            )
        if not changed:
            return self

        def transform(h: Hist) -> Hist:
            # Separate indexing preserves cropped values and variances in the flow bins.
            cropped = cast("Hist", h[tuple(crops)])
            return cast("Hist", cropped[tuple(merges)])

        return self.map_hists(transform)


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
    """Return one binning specification per axis for the ``bins`` given to :meth:`rebinned_to`."""
    if bins is None or isinstance(bins, int | float | np.number | str):
        return [bins] * ndim
    if ndim == 1 and (isinstance(bins, bh.axis.Axis) or _is_edges(bins)):
        return [bins]
    wanted = list(bins)
    if len(wanted) != ndim:
        msg = (
            f"got {len(wanted)} bin specifications for a {ndim}D histogram; give one "
            "binning specification per axis"
        )
        raise BinningError(msg)
    return wanted


def _ranges_per_axis(range_: Any, ndim: int) -> list[Any]:
    """Return one range specification per axis for the ``range`` given to :meth:`rebinned_to`.

    ``None``, a string or a pair of numbers is one range for every axis; anything
    else is a sequence with one range per axis.
    """
    single = (
        range_ is None
        or isinstance(range_, str)
        or (isinstance(range_, Sequence) and len(range_) == 2 and all(map(_is_number, range_)))
    )
    if single:
        return [range_] * ndim
    ranges = list(range_)
    if len(ranges) != ndim:
        msg = f"got {len(ranges)} ranges for a {ndim}D histogram; give one per axis"
        raise BinningError(msg)
    return ranges


def _merge_boundaries(axis: Any, edges: np.ndarray) -> np.ndarray:
    """Return the positions of ``edges`` among the edges of ``axis`` (a merge of its bins).

    Every requested edge must coincide with one of the axis' edges (to a
    millionth of its smallest bin width, as :func:`~rootfig._storage.same_axis`
    compares). The first and last edges bound the cropped axis.
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
            f"axis {name!r} has no bin edge at {missing[0]:g}: {extent}, and cropping or "
            "merging keeps only its own edges. Fill from the tree to bin freely"
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


def from_sample(
    sample: Sample,
    hist_: Hist,
    *,
    stats: Summary | None = None,
    variations: Mapping[str, tuple[Hist, Hist | None]] | None = None,
    per_object: bool | None = None,
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
