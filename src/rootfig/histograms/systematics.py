"""Combining statistical and systematic uncertainties of histograms.

Every source of systematic uncertainty is a pair of varied histograms
(:attr:`~rootfig.histograms.Histogram.variations`). Per bin, each source shifts
the contents by ``up - nominal`` and ``down - nominal``; the larger positive
shift counts towards the upper uncertainty, the larger negative shift towards
the lower one (so two variations moving the same way enlarge one side only).
Different sources are independent and added in quadrature; sources with the
same name in several histograms are fully correlated, so a sum of histograms
adds their variations linearly (:func:`sum_histograms`). The total uncertainty
is the statistical one and the systematic one added in quadrature, per side.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from rootfig._storage import add_hists
from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError
from rootfig.histograms.build import Histogram, compatible_binning
from rootfig.histograms.intervals import count_scale

__all__ = ["Uncertainty", "sum_histograms", "uncertainty"]


@dataclass(frozen=True)
class Uncertainty:
    """Per-bin statistical and systematic uncertainties of a one-dimensional histogram.

    All arrays cover the visible bins. Uncertainties are non-negative
    magnitudes below (``down``) and above (``up``) the nominal contents.

    Attributes
    ----------
    edges
        Bin edges.
    nominal
        Nominal bin contents.
    stat_down, stat_up
        Statistical uncertainty below and above the contents
        (:meth:`Histogram.errors() <rootfig.histograms.Histogram.errors>`).
    components
        Signed shifts of every source, ``{name: (up - nominal, down - nominal)}``.
    """

    edges: FloatArray
    nominal: FloatArray
    stat_down: FloatArray
    stat_up: FloatArray
    components: Mapping[str, tuple[FloatArray, FloatArray]] = field(default_factory=dict)

    @property
    def syst_up(self) -> FloatArray:
        """Systematic uncertainty above the nominal contents."""
        shifts = [np.maximum(np.maximum(up, down), 0.0) for up, down in self.components.values()]
        return _quadrature(shifts, self.nominal)

    @property
    def syst_down(self) -> FloatArray:
        """Systematic uncertainty below the nominal contents."""
        shifts = [np.minimum(np.minimum(up, down), 0.0) for up, down in self.components.values()]
        return _quadrature(shifts, self.nominal)

    @property
    def total_up(self) -> FloatArray:
        """Statistical and systematic uncertainty above the nominal, in quadrature.

        The usual convention for an uncertainty band: with a Poisson
        interval, which is no Gaussian standard deviation, the sum has no
        exact coverage.
        """
        return np.asarray(np.hypot(self.stat_up, self.syst_up), dtype=float)

    @property
    def total_down(self) -> FloatArray:
        """Statistical and systematic uncertainty below the nominal (see :attr:`total_up`)."""
        return np.asarray(np.hypot(self.stat_down, self.syst_down), dtype=float)

    @property
    def has_systematics(self) -> bool:
        """True if at least one systematic source contributes."""
        return bool(self.components)


def _quadrature(shifts: Sequence[FloatArray], like: FloatArray) -> FloatArray:
    """Add ``shifts`` in quadrature bin by bin; zero (shaped like ``like``) without any."""
    if not shifts:
        return np.zeros_like(like)
    return np.asarray(np.sqrt(np.sum(np.square(shifts), axis=0)), dtype=float)


def uncertainty(histogram: Histogram) -> Uncertainty:
    """Summarise the statistical and systematic uncertainties of a 1D ``histogram``.

    Raises
    ------
    BinningError
        If the histogram is not one-dimensional.
    """
    if histogram.ndim != 1:
        msg = "uncertainty() needs a one-dimensional histogram"
        raise BinningError(msg)
    nominal = histogram.values()
    components = {
        name: (
            np.asarray(up.values(), dtype=float) - nominal,
            np.asarray(down.values(), dtype=float) - nominal,
        )
        for name, (up, down) in histogram.variations.items()
    }
    stat_down, stat_up = histogram.errors()
    return Uncertainty(
        edges=histogram.edges,
        nominal=nominal,
        stat_down=stat_down,
        stat_up=stat_up,
        components=components,
    )


def sum_histograms(histograms: Sequence[Histogram], *, label: str = "Total") -> Histogram:
    """Add one-dimensional histograms bin by bin, keeping their systematic variations.

    Variations are matched by name and added linearly (fully correlated); a
    histogram without a source contributes its nominal contents to it. The sum
    keeps the inputs' ``normalization`` when they all share it and has none
    otherwise, so it never claims a scaling one of its parts lacks; it counts
    objects (``per_object``) if any input does, and is observed data
    (``is_data``) if every input is. It keeps the Poisson interval
    (:attr:`~rootfig.histograms.Histogram.poisson`) when every input has it
    at one confidence level with the same factor per count in every bin,
    however the inputs reached their binning, since counts of one factor add up
    to counts of it,
    as ``TH1::Add`` keeps ``kPoisson`` for unweighted histograms. Otherwise, if
    an input has errors of its own (``stat_errors``), each input's ``(down,
    up)`` errors add in quadrature side by side, an approximation for
    asymmetric errors; else the sum has ``sqrt(sum w^2)``.

    Raises
    ------
    BinningError
        If there are no histograms, an input is not one-dimensional, or the
        bin edges or flow-bin traits differ. Axis names and labels may differ.
    """
    if not histograms:
        msg = "nothing to sum: no histograms given"
        raise BinningError(msg)
    first = histograms[0]
    if first.ndim != 1:
        msg = "sum_histograms() needs one-dimensional histograms"
        raise BinningError(msg)
    for other in histograms[1:]:
        if not compatible_binning(first.hist, other.hist):
            msg = (
                f"cannot add histograms with different bin edges ({first.label!r} and "
                f"{other.label!r})"
            )
            raise BinningError(msg)
        if (first.axis.traits.underflow, first.axis.traits.overflow) != (
            other.axis.traits.underflow,
            other.axis.traits.overflow,
        ):
            msg = "cannot add histograms with different flow-bin traits"
            raise BinningError(msg)
    names = list(dict.fromkeys(name for h in histograms for name in h.variations))
    variations: dict[str, tuple[Hist, Hist]] = {}
    for name in names:
        up = add_hists(
            [h.variations[name][0] if name in h.variations else h.hist for h in histograms]
        )
        down = add_hists(
            [h.variations[name][1] if name in h.variations else h.hist for h in histograms]
        )
        variations[name] = (up, down)
    unit = first._unit  # one count per bin, set exactly for Poisson histograms
    shared = unit is not None and all(
        h._unit is not None
        and h._cl == first._cl
        and np.allclose(_count_scale(h._unit), _count_scale(unit), rtol=1e-12, atol=0)
        for h in histograms[1:]
    )
    total = add_hists([h.hist for h in histograms])
    errors: tuple[FloatArray, FloatArray] | None = None
    if not shared and any(h._errors is not None for h in histograms):
        # given errors add in quadrature side by side, every input with its own (down, up)
        sides = [h.errors(flow=True) for h in histograms]
        errors = (
            np.sqrt(np.sum([side[0] ** 2 for side in sides], axis=0)),
            np.sqrt(np.sum([side[1] ** 2 for side in sides], axis=0)),
        )
    return Histogram(
        total,
        label=label,
        normalization=first.normalization
        if all(h.normalization == first.normalization for h in histograms)
        else None,
        variations=variations,
        per_object=any(h.per_object for h in histograms),
        is_data=all(h.is_data for h in histograms),
        poisson=first.poisson if shared else False,
        _unit=unit.copy() if shared and unit is not None else None,
        _weighted=any(h._weighted for h in histograms),
        stat_errors=errors,
    )


def _count_scale(unit: Hist) -> FloatArray:
    """Return the factor of a count in every cell of a Poisson histogram's record of counts."""
    return count_scale(unit.values(flow=True), np.asarray(unit.variances(flow=True)))
