"""Filling ``Hist`` objects and the :class:`Histogram` wrapper."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import hist
import numpy as np

from rootfig._mapping import FrozenMapping
from rootfig._storage import as_weight_storage, same_binning
from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError, SystematicError
from rootfig.histograms.stats import Summary

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
]


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

        A histogram that was not filled by rootfig carries no entry count: its
        bin contents are sums of weights, which only equal the number of fills
        for unweighted, unscaled histograms (see :attr:`sum_weights`).
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
            if isinstance(axis, hist.axis.StrCategory | hist.axis.IntCategory):
                msg = f"cannot merge the categories of axis {axis.label or axis.name!r}"
                raise BinningError(msg)
            if axis.size % step:
                divisors = [d for d in range(1, axis.size + 1) if axis.size % d == 0]
                msg = (
                    f"cannot merge bins of {axis.label or axis.name!r} in groups of {step}: "
                    f"{axis.size} bins can only be grouped by {divisors}"
                )
                raise BinningError(msg)
        selection = tuple(slice(None, None, hist.rebin(step)) for step in factors)

        def merge(h: Hist) -> Hist:
            # slicing with rebin keeps every axis, so the result is a Hist, never a float
            return cast("Hist", h[selection])

        return self.map_hists(merge)


def _is_positive_integer(value: object) -> bool:
    """Return True for ``1``, ``2``, ``numpy.int64(3)``, ...; not for bools, floats or strings."""
    return isinstance(value, int | np.integer) and not isinstance(value, bool) and int(value) >= 1


def from_sample(
    sample: Sample,
    hist_: Hist,
    *,
    stats: Summary | None = None,
    variations: Mapping[str, tuple[Hist, Hist | None]] | None = None,
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
