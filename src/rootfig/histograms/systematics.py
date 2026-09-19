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
    stat
        Statistical uncertainty, ``sqrt(variances)``.
    components
        Signed shifts of every source, ``{name: (up - nominal, down - nominal)}``.
    """

    edges: FloatArray
    nominal: FloatArray
    stat: FloatArray
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
        """Statistical and systematic uncertainty above the nominal, in quadrature."""
        return np.asarray(np.hypot(self.stat, self.syst_up), dtype=float)

    @property
    def total_down(self) -> FloatArray:
        """Statistical and systematic uncertainty below the nominal, in quadrature."""
        return np.asarray(np.hypot(self.stat, self.syst_down), dtype=float)

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
    return Uncertainty(
        edges=histogram.edges,
        nominal=nominal,
        stat=histogram.errors(),
        components=components,
    )


def sum_histograms(histograms: Sequence[Histogram], *, label: str = "Total") -> Histogram:
    """Add one-dimensional histograms bin by bin, keeping their systematic variations.

    Variations are matched by name and added linearly (fully correlated); a
    histogram without a source contributes its nominal contents to it. The sum
    keeps the inputs' ``normalization`` when they all share it and has none
    otherwise, so it never claims a scaling one of its parts lacks; it counts
    objects (``per_object``) if any input does.

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
    return Histogram(
        add_hists([h.hist for h in histograms]),
        label=label,
        normalization=first.normalization
        if all(h.normalization == first.normalization for h in histograms)
        else None,
        variations=variations,
        per_object=any(h.per_object for h in histograms),
    )
