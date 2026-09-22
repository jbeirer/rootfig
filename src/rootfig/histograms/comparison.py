"""Bin-by-bin comparisons of two histograms: ratios, differences, pulls and significances."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np

from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError, RootfigWarning
from rootfig.histograms.build import Histogram, compatible_binning
from rootfig.histograms.systematics import Uncertainty

__all__ = [
    "COMPARISON_KINDS",
    "Comparison",
    "ComparisonKind",
    "UncertaintyMode",
    "compare",
]

ComparisonKind: TypeAlias = Literal[
    "ratio", "difference", "relative_difference", "pull", "s/sqrt(b)", "s/sqrt(s+b)"
]
"""What a comparison computes per bin: see :func:`compare`."""

COMPARISON_KINDS: tuple[ComparisonKind, ...] = (
    "ratio",
    "difference",
    "relative_difference",
    "pull",
    "s/sqrt(b)",
    "s/sqrt(s+b)",
)
"""Every :data:`ComparisonKind`, for checking strings at runtime."""

BAND_KINDS: tuple[ComparisonKind, ...] = ("ratio", "relative_difference", "difference")
"""The kinds with a reference band, between which the :data:`UncertaintyMode` chooses."""

SIGNIFICANCE_KINDS: tuple[ComparisonKind, ...] = ("s/sqrt(b)", "s/sqrt(s+b)")
"""The kinds comparing a signal with its background, statistical only."""

UncertaintyMode: TypeAlias = Literal["propagate", "numerator"]
"""How the uncertainties of a ratio, relative difference or difference enter its error bars.

* ``"propagate"`` - the statistical uncertainties of both sides, uncorrelated,
  and the systematic ones source by source through the varied comparison, so a
  source shared by both sides cancels where the comparison allows.
* ``"numerator"`` - only the numerator's; the reference's uncertainty is the
  :attr:`Comparison.band` (the usual data/MC convention, mplhep's ``split_ratio``).
"""

_Variations: TypeAlias = Mapping[str, tuple[Hist, Hist]]


@dataclass(frozen=True)
class Comparison:
    """A bin-by-bin comparison of a numerator with a reference.

    Attributes
    ----------
    kind
        The :data:`ComparisonKind` computed.
    label, reference
        The labels of the numerator and the reference; ``""`` for a plain
        ``hist.Hist``.
    values
        The comparison per bin; ``nan`` where it is undefined (an empty
        reference for a ratio, a zero uncertainty for a pull).
    errors
        Statistical uncertainty on ``values`` (see :data:`UncertaintyMode`); 1
        for a pull, whose uncertainty is its unit.
    edges
        Bin edges shared by both histograms.
    band
        The reference's statistical uncertainty, drawn around the baseline: relative
        for a ratio and a relative difference, absolute for a difference;
        ``None`` for a pull and a significance.
    syst_errors
        Systematic uncertainty on ``values`` as ``(down, up)`` (matplotlib's
        ``yerr`` order), following :data:`UncertaintyMode`; ``None`` without
        systematic variations, and for a pull, which divides by them.
    syst_band
        The reference's systematic uncertainty, in the units of ``band``, as
        ``(down, up)``; ``None`` if the reference has no variations.
    """

    kind: ComparisonKind
    label: str
    reference: str
    values: FloatArray
    errors: FloatArray
    edges: FloatArray
    band: FloatArray | None = None
    syst_errors: tuple[FloatArray, FloatArray] | None = None
    syst_band: tuple[FloatArray, FloatArray] | None = None

    @property
    def centers(self) -> FloatArray:
        """Bin centres."""
        return np.asarray(0.5 * (self.edges[1:] + self.edges[:-1]), dtype=float)

    @property
    def half_widths(self) -> FloatArray:
        """Half bin widths (for horizontal error bars)."""
        return np.asarray(0.5 * np.diff(self.edges), dtype=float)

    def total_errors(self) -> tuple[FloatArray, FloatArray]:
        """Statistical and systematic uncertainty on ``values`` in quadrature, ``(down, up)``."""
        if self.syst_errors is None:
            return self.errors, self.errors
        down, up = self.syst_errors
        return np.hypot(self.errors, down), np.hypot(self.errors, up)

    def total_band(self) -> tuple[FloatArray, FloatArray] | None:
        """Return the reference's total uncertainty, ``(down, up)``; ``None`` without a band."""
        if self.band is None:
            return None
        if self.syst_band is None:
            return self.band, self.band
        down, up = self.syst_band
        return np.hypot(self.band, down), np.hypot(self.band, up)


def compare(
    numerator: Hist | Histogram,
    reference: Hist | Histogram,
    *,
    kind: ComparisonKind = "ratio",
    uncertainty: UncertaintyMode = "propagate",
) -> Comparison:
    """Compare ``numerator`` with ``reference`` bin by bin, with uncertainties.

    With ``n`` and ``d`` the contents and ``vn`` and ``vd`` the variances:

    * ``"ratio"`` is ``n / d``, ``"relative_difference"`` ``n / d - 1`` (same
      uncertainties), ``"difference"`` ``n - d``;
    * ``"pull"`` is ``(n - d) / sqrt(vn + vd + syst**2)``, where ``syst`` is the
      systematic uncertainty of ``n - d`` on the side facing the reference (the
      lower one where ``n > d``, the upper one elsewhere);
    * ``"s/sqrt(b)"`` and ``"s/sqrt(s+b)"`` are per-bin significances of the
      numerator as signal over the reference as background, statistical only.

    Statistical uncertainties of the two sides are uncorrelated. ``Histogram``
    inputs with systematic variations enter source by source through the varied
    comparison itself: a source present on both sides varies both together (a
    shared luminosity uncertainty cancels in a ratio), a source on one side only
    varies that side against the other's nominal contents. The shifts of
    different sources then combine like those of one histogram (see
    :mod:`rootfig.histograms.systematics`). With ``uncertainty="numerator"``
    only the numerator's sources enter the error bars; the reference's are the
    band. A variation that empties a reference bin of a ratio leaves that bin's
    systematic uncertainty ``nan``, with a :class:`~rootfig.errors.RootfigWarning`.

    Raises
    ------
    BinningError
        If the histograms do not share the same one-dimensional binning.
    ValueError
        For an unknown ``kind`` or ``uncertainty``, or ``uncertainty="numerator"``
        with a pull or a significance, which have no band.
    """
    if kind not in COMPARISON_KINDS:
        msg = f"kind must be one of {COMPARISON_KINDS}, got {kind!r}"
        raise ValueError(msg)
    if uncertainty not in ("propagate", "numerator"):
        msg = f"uncertainty must be 'propagate' or 'numerator', got {uncertainty!r}"
        raise ValueError(msg)
    if uncertainty == "numerator" and kind not in BAND_KINDS:
        msg = f"uncertainty='numerator' needs a reference band, which kind={kind!r} does not have"
        raise ValueError(msg)
    num, ref = _Side.of(numerator), _Side.of(reference)
    if not compatible_binning(num.hist, ref.hist):
        msg = "compare requires two one-dimensional histograms with identical bin edges"
        raise BinningError(msg)
    propagate = uncertainty == "propagate"
    with np.errstate(divide="ignore", invalid="ignore"):
        match kind:
            case "ratio" | "relative_difference":
                fields = _ratio(num, ref, propagate=propagate)
                if kind == "relative_difference":
                    fields["values"] = fields["values"] - 1.0
            case "difference":
                fields = _difference(num, ref, propagate=propagate)
            case "pull":
                fields = _pull(num, ref)
            case _:
                fields = _significance(num, ref, kind)
    return Comparison(
        kind=kind,
        label=num.label,
        reference=ref.label,
        edges=np.asarray(num.hist.axes[0].edges, dtype=float),
        **fields,
    )


@dataclass(frozen=True)
class _Side:
    """One side of a comparison: its contents, variances and systematic variations."""

    hist: Hist
    label: str
    values: FloatArray
    variances: FloatArray
    variations: _Variations

    @classmethod
    def of(cls, histogram: Hist | Histogram) -> _Side:
        variations: _Variations = {}
        label = ""
        if isinstance(histogram, Histogram):
            variations, label, histogram = histogram.variations, histogram.label, histogram.hist
        return cls(
            hist=histogram,
            label=label,
            values=np.asarray(histogram.values(), dtype=float),
            variances=np.asarray(histogram.variances(), dtype=float),
            variations=variations,
        )

    def nominal(self) -> _Side:
        """Return this side without variations, for a mode that leaves its sources out."""
        return _Side(self.hist, self.label, self.values, self.variances, {})

    def varied(self, name: str, index: int) -> FloatArray:
        """Contents of variation ``name`` (``index`` 0 up, 1 down), or the nominal without it."""
        if name not in self.variations:
            return self.values
        return np.asarray(self.variations[name][index].values(), dtype=float)


def _ratio(num: _Side, ref: _Side, *, propagate: bool) -> dict[str, Any]:
    """``n / d``, its error bars and the reference's relative uncertainty as the band."""
    n, d, vn, vd = num.values, ref.values, num.variances, ref.variances

    def divide(top: FloatArray, bottom: FloatArray) -> FloatArray:
        return np.asarray(np.where((d != 0) & (bottom != 0), top / bottom, np.nan))

    values = divide(n, d)
    if propagate:
        errors = divide(np.sqrt(vn + n**2 * vd / d**2), np.abs(d))
        varied_ref = ref
    else:
        errors = divide(np.sqrt(vn), np.abs(d))
        varied_ref = ref.nominal()  # the reference's sources are the band
    for name in dict.fromkeys([*num.variations, *varied_ref.variations]):
        for index, direction in ((0, "up"), (1, "down")):
            _warn_emptied(name, direction, (d != 0) & (varied_ref.varied(name, index) == 0))
    band_shifts = {
        name: (divide(up.values() - d, d), divide(down.values() - d, d))
        for name, (up, down) in ref.variations.items()
    }
    return {
        "values": values,
        "errors": errors,
        "band": divide(np.sqrt(vd), np.abs(d)),
        "syst_errors": _combined(values, _source_shifts(divide, values, num, varied_ref)),
        "syst_band": _combined(values, band_shifts),
    }


def _difference(num: _Side, ref: _Side, *, propagate: bool) -> dict[str, Any]:
    """``n - d``, its error bars and the reference's absolute uncertainty as the band."""
    n, d, vn, vd = num.values, ref.values, num.variances, ref.variances
    values = n - d
    varied_ref = ref if propagate else ref.nominal()
    band_shifts = {
        name: (up.values() - d, down.values() - d) for name, (up, down) in ref.variations.items()
    }
    return {
        "values": values,
        "errors": np.sqrt(vn + vd) if propagate else np.sqrt(vn),
        "band": np.sqrt(vd),
        "syst_errors": _combined(values, _source_shifts(np.subtract, values, num, varied_ref)),
        "syst_band": _combined(values, band_shifts),
    }


def _pull(num: _Side, ref: _Side) -> dict[str, Any]:
    """``(n - d) / sigma`` with statistical and systematic ``sigma``; ``nan`` where it is zero.

    The systematic part is taken on the side of ``n - d`` facing the other
    histogram, as mplhep does for Poisson pulls: the lower uncertainty where
    ``n > d``, the upper one elsewhere.
    """
    difference = num.values - ref.values
    syst = _combined(difference, _source_shifts(np.subtract, difference, num, ref))
    facing = np.zeros_like(difference) if syst is None else np.where(difference > 0, *syst)
    sigma = np.sqrt(num.variances + ref.variances + facing**2)
    values = np.asarray(np.where(sigma > 0, difference / sigma, np.nan), dtype=float)
    return {"values": values, "errors": np.where(np.isfinite(values), 1.0, np.nan)}


def _significance(signal: _Side, background: _Side, kind: ComparisonKind) -> dict[str, Any]:
    """Per-bin significance of ``signal`` over ``background``, statistical only.

    ``nan`` where the background (or, for ``s/sqrt(s+b)``, the total) is not positive.
    """
    s, b, vs, vb = signal.values, background.values, signal.variances, background.variances
    if kind == "s/sqrt(b)":
        ok = b > 0
        values = np.where(ok, s / np.sqrt(b), np.nan)
        # d/ds = 1/sqrt(b), d/db = -s / (2 b^1.5)
        errors = np.where(ok, np.sqrt(vs / b + s**2 * vb / (4 * b**3)), np.nan)
    else:
        total = s + b
        ok = total > 0
        values = np.where(ok, s / np.sqrt(total), np.nan)
        # d/ds = (s + 2b) / (2 (s+b)^1.5), d/db = -s / (2 (s+b)^1.5)
        ds = (s + 2 * b) / (2 * total**1.5)
        db = -s / (2 * total**1.5)
        errors = np.where(ok, np.sqrt(ds**2 * vs + db**2 * vb), np.nan)
    return {
        "values": np.asarray(values, dtype=float),
        "errors": np.asarray(errors, dtype=float),
    }


def _source_shifts(
    operation: Callable[[FloatArray, FloatArray], FloatArray],
    nominal: FloatArray,
    num: _Side,
    ref: _Side,
) -> dict[str, tuple[FloatArray, FloatArray]]:
    """Signed shifts of ``operation(n, d)`` per source, both sides varied together.

    A side without the source keeps its nominal contents, so a source shared by
    both sides moves them together and cancels where the operation allows.
    """
    shifts = {}
    for name in dict.fromkeys([*num.variations, *ref.variations]):
        up, down = (
            np.asarray(operation(num.varied(name, i), ref.varied(name, i)) - nominal, dtype=float)
            for i in (0, 1)
        )
        shifts[name] = (up, down)
    return shifts


def _warn_emptied(name: str, direction: str, emptied: np.ndarray) -> None:
    """Warn that a variation empties denominator bins, leaving the ratio's systematic undefined."""
    if emptied.any():
        warnings.warn(
            f"systematic {name!r} {direction} empties the denominator in "
            f"{int(emptied.sum())} bin(s); the ratio's systematic uncertainty there is nan",
            RootfigWarning,
            stacklevel=4,  # the caller of compare()
        )


def _combined(
    like: FloatArray, shifts: Mapping[str, tuple[FloatArray, FloatArray]]
) -> tuple[FloatArray, FloatArray] | None:
    """Combine signed per-source shifts into ``(down, up)`` magnitudes, or ``None`` without any."""
    if not shifts:
        return None
    edges = np.arange(len(like) + 1, dtype=float)  # combined bin by bin: the edges do not enter
    summary = Uncertainty(edges=edges, nominal=like, stat=np.zeros_like(like), components=shifts)
    return summary.syst_down, summary.syst_up
