"""Bin-by-bin comparisons: ratios, differences, asymmetries, pulls and significances."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np

from rootfig._storage import same_edges
from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError, RootfigWarning
from rootfig.histograms.build import Histogram, compatible_binning
from rootfig.histograms.efficiency import Efficiency, Profile
from rootfig.histograms.systematics import Uncertainty

__all__ = [
    "COMPARISON_KINDS",
    "Comparison",
    "ComparisonKind",
    "UncertaintyMode",
    "compare",
]

ComparisonKind: TypeAlias = Literal[
    "ratio", "difference", "relative_difference", "pull", "asymmetry", "s/sqrt(b)", "s/sqrt(s+b)"
]
"""What a comparison computes per bin: see :func:`compare`."""

COMPARISON_KINDS: tuple[ComparisonKind, ...] = (
    "ratio",
    "difference",
    "relative_difference",
    "pull",
    "asymmetry",
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
_Errors: TypeAlias = tuple[FloatArray, FloatArray]


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
        for a pull, whose uncertainty is its unit. A ``(down, up)`` pair for
        efficiencies, whose intervals are asymmetric.
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
    uncertainty
        The :data:`UncertaintyMode` of the error bars: with ``"numerator"`` the
        reference's uncertainty is only in the band, which is then drawn.
    reference_hist
        The histogram compared with, which a panel drawing several comparisons
        checks they share; ``None`` for a :class:`Comparison` built by hand.
    """

    kind: ComparisonKind
    label: str
    reference: str
    values: FloatArray
    errors: FloatArray | _Errors
    edges: FloatArray
    band: FloatArray | None = None
    syst_errors: tuple[FloatArray, FloatArray] | None = None
    syst_band: tuple[FloatArray, FloatArray] | None = None
    uncertainty: UncertaintyMode = "propagate"
    reference_hist: Hist | None = None

    @property
    def centers(self) -> FloatArray:
        """Bin centres."""
        return np.asarray(0.5 * (self.edges[1:] + self.edges[:-1]), dtype=float)

    @property
    def half_widths(self) -> FloatArray:
        """Half bin widths (for horizontal error bars)."""
        return np.asarray(0.5 * np.diff(self.edges), dtype=float)

    def total_errors(self) -> _Errors:
        """Statistical and systematic uncertainty on ``values`` in quadrature, ``(down, up)``."""
        down, up = self.errors if isinstance(self.errors, tuple) else (self.errors, self.errors)
        if self.syst_errors is None:
            return down, up
        syst_down, syst_up = self.syst_errors
        return np.hypot(down, syst_down), np.hypot(up, syst_up)

    def total_band(self) -> tuple[FloatArray, FloatArray] | None:
        """Return the reference's total uncertainty, ``(down, up)``; ``None`` without a band."""
        if self.band is None:
            return None
        if self.syst_band is None:
            return self.band, self.band
        down, up = self.syst_band
        return np.hypot(self.band, down), np.hypot(self.band, up)


def compare(
    numerator: Hist | Histogram | Efficiency | Profile,
    reference: Hist | Histogram | Efficiency | Profile,
    *,
    kind: ComparisonKind = "ratio",
    uncertainty: UncertaintyMode = "propagate",
) -> Comparison:
    """Compare ``numerator`` with ``reference`` bin by bin, with uncertainties.

    With ``n`` and ``d`` the contents and ``vn`` and ``vd`` the variances:

    * ``"ratio"`` is ``n / d``, ``"relative_difference"`` ``n / d - 1`` (same
      uncertainties), ``"difference"`` ``n - d``;
    * ``"asymmetry"`` is ``(n - d) / (n + d)``, both sides propagated;
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
    band. In a ratio or a relative difference, a variation that empties a
    reference bin leaves that bin's systematic uncertainty ``nan``, with a
    :class:`~rootfig.errors.RootfigWarning`.

    Two :class:`~rootfig.histograms.Efficiency` or two
    :class:`~rootfig.histograms.Profile` objects compare their values the same
    way, with their intervals propagated to first order as independent: the
    lower error of the result takes each side's lower or upper error, whichever
    lowers it, so the asymmetric intervals of an efficiency stay asymmetric
    (``errors`` is then ``(down, up)``). A pull divides by the errors facing the
    other side. Significances, which count events, and ``"numerator"``, which
    needs a band, are refused for them.

    Raises
    ------
    BinningError
        If the inputs do not share the same one-dimensional binning.
    TypeError
        For a histogram compared with an efficiency or a profile, or an
        efficiency with a profile.
    ValueError
        For an unknown ``kind`` or ``uncertainty``, ``uncertainty="numerator"``
        with a pull, an asymmetry or a significance, which have no band, and
        for efficiencies and profiles as described above, or profiles of
        different statistics.
    """
    if kind not in COMPARISON_KINDS:
        msg = f"kind must be one of {COMPARISON_KINDS}, got {kind!r}"
        raise ValueError(msg)
    if uncertainty not in ("propagate", "numerator"):
        msg = f"uncertainty must be 'propagate' or 'numerator', got {uncertainty!r}"
        raise ValueError(msg)
    if isinstance(numerator, Efficiency | Profile) or isinstance(reference, Efficiency | Profile):
        return _compare_points(numerator, reference, kind=kind, uncertainty=uncertainty)
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
            case "asymmetry":
                fields = _asymmetry(num, ref)
            case _:
                fields = _significance(num, ref, kind)
    return Comparison(
        kind=kind,
        label=num.label,
        reference=ref.label,
        edges=np.asarray(num.hist.axes[0].edges, dtype=float),
        uncertainty=uncertainty,
        reference_hist=ref.hist,
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


def _asymmetry(num: _Side, ref: _Side) -> dict[str, Any]:
    """``(n - d) / (n + d)`` with both sides propagated; ``nan`` where the sum is zero."""
    n, d, vn, vd = num.values, ref.values, num.variances, ref.variances

    def asymmetry(top: FloatArray, bottom: FloatArray) -> FloatArray:
        total = top + bottom
        return np.asarray(np.where(total != 0, (top - bottom) / total, np.nan))

    values = asymmetry(n, d)
    total = n + d
    # d/dn = 2d / (n+d)^2, d/dd = -2n / (n+d)^2
    errors = np.where(total != 0, 2 * np.sqrt(d**2 * vn + n**2 * vd) / total**2, np.nan)
    return {
        "values": values,
        "errors": np.asarray(errors, dtype=float),
        "syst_errors": _combined(values, _source_shifts(asymmetry, values, num, ref)),
    }


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
    """Warn that a variation empties reference bins, so the comparison's systematic is undefined."""
    if emptied.any():
        warnings.warn(
            f"systematic {name!r} {direction} empties the reference in "
            f"{int(emptied.sum())} bin(s); the comparison's systematic uncertainty there is nan",
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


def _compare_points(
    numerator: object, reference: object, *, kind: ComparisonKind, uncertainty: UncertaintyMode
) -> Comparison:
    """Compare two efficiencies or two profiles, their intervals independent."""
    pair = (type(numerator).__name__, type(reference).__name__)
    num: Efficiency | Profile
    ref: Efficiency | Profile
    if isinstance(numerator, Efficiency) and isinstance(reference, Efficiency):
        num, ref = numerator, reference
    elif isinstance(numerator, Profile) and isinstance(reference, Profile):
        if numerator.statistic != reference.statistic:
            msg = (
                f"compare needs profiles of one statistic, got {numerator.statistic!r} and "
                f"{reference.statistic!r}"
            )
            raise ValueError(msg)
        num, ref = numerator, reference
    else:
        msg = f"compare takes two histograms, two efficiencies or two profiles, got {pair}"
        raise TypeError(msg)
    if kind in SIGNIFICANCE_KINDS:
        msg = f"kind={kind!r} counts signal and background events; compare {pair} as a ratio"
        raise ValueError(msg)
    if uncertainty != "propagate":
        msg = (
            f"uncertainty={uncertainty!r} needs a reference band, which {pair} do not have: "
            "their intervals are independent and always propagated"
        )
        raise ValueError(msg)
    if not same_edges(num.edges, ref.edges):
        msg = "compare requires two inputs with identical bin edges"
        raise BinningError(msg)
    a, b = num.values, ref.values
    a_err, b_err = _intervals(num), _intervals(ref)
    errors: FloatArray | _Errors
    with np.errstate(divide="ignore", invalid="ignore"):
        match kind:
            case "ratio" | "relative_difference":
                values = np.where(b != 0, a / b, np.nan)
                errors = _propagated(1 / b, -a / b**2, a_err, b_err)
                if kind == "relative_difference":
                    values = values - 1.0
            case "difference":
                values = a - b
                errors = _propagated(1.0, -1.0, a_err, b_err)
            case "asymmetry":
                total = a + b
                values = np.where(total != 0, (a - b) / total, np.nan)
                errors = _propagated(2 * b / total**2, -2 * a / total**2, a_err, b_err)
            case _:  # pull
                down, up = _propagated(1.0, -1.0, a_err, b_err)
                sigma = np.where(a > b, down, up)  # the side of a - b facing zero
                values = np.where(sigma > 0, (a - b) / sigma, np.nan)
                errors = np.where(np.isfinite(values), 1.0, np.nan)
    values = np.asarray(values, dtype=float)
    if isinstance(errors, tuple):
        undefined = ~np.isfinite(values)
        errors = (np.where(undefined, np.nan, errors[0]), np.where(undefined, np.nan, errors[1]))
    return Comparison(
        kind=kind,
        label=num.label,
        reference=ref.label,
        values=values,
        errors=errors,
        edges=np.asarray(num.edges, dtype=float),
    )


def _intervals(points: Efficiency | Profile) -> _Errors:
    """Return the ``(down, up)`` errors of an efficiency or a profile."""
    if isinstance(points, Efficiency):
        return points.errors
    return points.errors, points.errors


def _propagated(pa: Any, pb: Any, a_err: _Errors, b_err: _Errors) -> _Errors:
    """First-order ``(down, up)`` errors of ``f(a, b)`` with partial derivatives ``pa`` and ``pb``.

    ``f`` goes down with ``a`` where ``pa >= 0`` and up with it elsewhere, so its
    lower error takes ``a``'s lower error there and ``a``'s upper error elsewhere;
    the same for ``b``. The two sides are independent and add in quadrature.
    """

    def moved(partial: Any, err: _Errors) -> _Errors:
        down, up = err
        rising = np.asarray(partial) >= 0
        return (
            np.asarray(np.where(rising, partial * down, -partial * up), dtype=float),
            np.asarray(np.where(rising, partial * up, -partial * down), dtype=float),
        )

    (a_down, a_up), (b_down, b_up) = moved(pa, a_err), moved(pb, b_err)
    return np.hypot(a_down, b_down), np.hypot(a_up, b_up)
