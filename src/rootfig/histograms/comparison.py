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
from rootfig.histograms.binomial import clopper_pearson
from rootfig.histograms.build import Histogram, compatible_binning
from rootfig.histograms.efficiency import Efficiency, Profile
from rootfig.histograms.intervals import ONE_SIGMA
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

UncertaintyMode: TypeAlias = Literal["propagate", "numerator", "poisson-ratio"]
"""How the uncertainties of a ratio, relative difference or difference enter its error bars.

* ``"propagate"`` - the statistical uncertainties of both sides, uncorrelated,
  and the systematic ones source by source through the varied comparison, so a
  source shared by both sides cancels where the comparison allows.
* ``"numerator"`` - only the numerator's; the reference's uncertainty is the
  :attr:`Comparison.band` (the usual data/MC convention, mplhep's ``split_ratio``).
* ``"poisson-ratio"`` - the exact interval of the ratio of two independent
  Poisson means, for two histograms of counts (a ratio or a relative difference
  only): the Clopper-Pearson interval of ``n / (n + d)`` turned into one of
  ``n / d``, as ROOT's ``TGraphAsymmErrors::Divide(..., "pois")``. Scaled counts
  keep it, scaled like the contents; systematics enter as with ``"propagate"``.
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
        Statistical uncertainty on ``values`` as ``(down, up)`` (matplotlib's
        ``yerr`` order; see :data:`UncertaintyMode`), ``nan`` where ``values``
        is; 1 on both sides for a pull, whose uncertainty is its unit.
    edges
        Bin edges shared by both histograms.
    band
        The reference's statistical uncertainty as ``(down, up)``, drawn around
        the baseline: relative for a ratio and a relative difference, absolute
        for a difference; ``None`` for a pull, an asymmetry and a significance.
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
        checks they share; ``None`` for efficiencies and profiles and for a
        :class:`Comparison` built by hand.
    reference_points
        The efficiency or profile compared with, checked like ``reference_hist``;
        ``None`` for histograms and for a :class:`Comparison` built by hand.
    """

    kind: ComparisonKind
    label: str
    reference: str
    values: FloatArray
    errors: _Errors
    edges: FloatArray
    band: _Errors | None = None
    syst_errors: tuple[FloatArray, FloatArray] | None = None
    syst_band: tuple[FloatArray, FloatArray] | None = None
    uncertainty: UncertaintyMode = "propagate"
    reference_hist: Hist | None = None
    reference_points: Efficiency | Profile | None = None

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
        return _in_quadrature(self.errors, self.syst_errors)

    def total_band(self) -> _Errors | None:
        """Return the reference's total uncertainty, ``(down, up)``; ``None`` without a band."""
        if self.band is None:
            return None
        return _in_quadrature(self.band, self.syst_band)


def _in_quadrature(stat: _Errors, syst: _Errors | None) -> _Errors:
    """Add statistical and systematic ``(down, up)`` uncertainties in quadrature, side by side."""
    if syst is None:
        return stat
    return np.hypot(stat[0], syst[0]), np.hypot(stat[1], syst[1])


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

    Statistical uncertainties of the two sides are uncorrelated and propagated
    to first order from each side's ``(down, up)`` errors
    (:meth:`Histogram.errors() <rootfig.histograms.Histogram.errors>`; ``sqrt``
    of the variances for a plain ``hist.Hist``): the lower error of the result
    takes each side's lower or upper error, whichever lowers it, so
    asymmetric errors stay asymmetric, and a pull divides by the errors
    facing the other side. With symmetric errors these are the formulas
    above. ``Histogram``
    inputs with systematic variations enter source by source through the varied
    comparison itself: a source present on both sides varies both together (a
    shared luminosity uncertainty cancels in a ratio), a source on one side only
    varies that side against the other's nominal contents. The shifts of
    different sources then combine like those of one histogram (see
    :mod:`rootfig.histograms.systematics`). With ``uncertainty="numerator"``
    only the numerator's sources enter the error bars; the reference's are the
    band. ``uncertainty="poisson-ratio"`` takes the exact interval of the ratio
    of two Poisson means instead of the propagated one, for two histograms of
    counts (see :data:`UncertaintyMode`). In a ratio or a relative difference,
    a variation that empties a reference bin leaves that bin's systematic
    uncertainty ``nan``, with a :class:`~rootfig.errors.RootfigWarning`.

    Two :class:`~rootfig.histograms.Efficiency` or two
    :class:`~rootfig.histograms.Profile` objects compare their values the same
    way, their intervals propagated as independent, so the asymmetric intervals
    of an efficiency stay asymmetric. Significances, which count events, and
    ``"numerator"``, which needs a band, and ``"poisson-ratio"``, which needs
    counts, are refused for them.

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
    if uncertainty not in ("propagate", "numerator", "poisson-ratio"):
        msg = (
            f"uncertainty must be 'propagate', 'numerator' or 'poisson-ratio', got {uncertainty!r}"
        )
        raise ValueError(msg)
    if isinstance(numerator, Efficiency | Profile) or isinstance(reference, Efficiency | Profile):
        return _compare_points(numerator, reference, kind=kind, uncertainty=uncertainty)
    if uncertainty == "numerator" and kind not in BAND_KINDS:
        msg = f"uncertainty='numerator' needs a reference band, which kind={kind!r} does not have"
        raise ValueError(msg)
    if uncertainty == "poisson-ratio" and kind not in ("ratio", "relative_difference"):
        msg = f"uncertainty='poisson-ratio' is the interval of a ratio, not of kind={kind!r}"
        raise ValueError(msg)
    num, ref = _Side.of(numerator), _Side.of(reference)
    if not compatible_binning(num.hist, ref.hist):
        msg = "compare requires two one-dimensional histograms with identical bin edges"
        raise BinningError(msg)
    propagate = uncertainty != "numerator"
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
        if uncertainty == "poisson-ratio":
            ratio = fields["values"] + (1.0 if kind == "relative_difference" else 0.0)
            fields["errors"] = _poisson_ratio_errors(numerator, reference, ratio)
    return Comparison(
        kind=kind,
        label=num.label,
        reference=ref.label,
        edges=np.asarray(num.hist.axes[0].edges, dtype=float),
        uncertainty=uncertainty,
        reference_hist=ref.hist,
        **fields,
    )


def _poisson_ratio_errors(
    numerator: Hist | Histogram, reference: Hist | Histogram, ratio: FloatArray
) -> _Errors:
    """Return the interval of the ratio of two Poisson means around ``ratio``, ``(down, up)``.

    Given ``n + d`` counts, ``n`` is binomial with ``f = mu_n / (mu_n + mu_d)``;
    the Clopper-Pearson interval of ``f`` maps to ``f / (1 - f)``, times the
    ratio of the counts' factors. Undefined (``nan``) where ``d`` is 0.

    Raises
    ------
    ValueError
        If a side does not hold known counts (see :meth:`Histogram.counts`).
    """
    sides = [
        h if isinstance(h, Histogram) else Histogram(h, label="") for h in (numerator, reference)
    ]
    (n, n_factor), (d, d_factor) = (h.counts() for h in sides)
    poisson = [h for h in sides if h.poisson]
    cl = poisson[0]._cl if poisson else ONE_SIGMA
    lower, upper = clopper_pearson(n, n + d, cl)
    defined = d > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = n_factor / d_factor
        low = lower / (1.0 - lower) * scale
        high = np.where(upper < 1.0, upper / (1.0 - upper), np.inf) * scale
        down = np.where(defined, np.maximum(ratio - low, 0.0), np.nan)
        up = np.where(defined, np.maximum(high - ratio, 0.0), np.nan)
    return np.asarray(down, dtype=float), np.asarray(up, dtype=float)


@dataclass(frozen=True)
class _Side:
    """One side of a comparison: its contents, ``(down, up)`` errors and systematic variations."""

    hist: Hist
    label: str
    values: FloatArray
    errors: _Errors
    variations: _Variations

    @classmethod
    def of(cls, histogram: Hist | Histogram) -> _Side:
        if isinstance(histogram, Histogram):
            return cls(
                hist=histogram.hist,
                label=histogram.label,
                values=histogram.values(),
                errors=histogram.errors(),
                variations=histogram.variations,
            )
        sigma = np.sqrt(np.asarray(histogram.variances(), dtype=float))
        return cls(
            hist=histogram,
            label="",
            values=np.asarray(histogram.values(), dtype=float),
            errors=(sigma, sigma),
            variations={},
        )

    def nominal(self) -> _Side:
        """Return this side without variations, for a mode that leaves its sources out."""
        return _Side(self.hist, self.label, self.values, self.errors, {})

    def varied(self, name: str, index: int) -> FloatArray:
        """Contents of variation ``name`` (``index`` 0 up, 1 down), or the nominal without it."""
        if name not in self.variations:
            return self.values
        return np.asarray(self.variations[name][index].values(), dtype=float)


def _ratio(num: _Side, ref: _Side, *, propagate: bool) -> dict[str, Any]:
    """``n / d``, its error bars and the reference's relative uncertainty as the band."""
    n, d = num.values, ref.values

    def divide(top: FloatArray, bottom: FloatArray) -> FloatArray:
        return np.asarray(np.where((d != 0) & (bottom != 0), top / bottom, np.nan))

    values = divide(n, d)
    defined = d != 0
    if propagate:
        errors = _propagated(1 / d, -n / d**2, num.errors, ref.errors)
        varied_ref = ref
    else:
        errors = _propagated(1 / d, 0.0, num.errors, _NONE)
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
        "errors": _where_defined(defined, errors),
        "band": _where_defined(defined, _propagated(1 / d, 0.0, ref.errors, _NONE)),
        "syst_errors": _combined(values, _source_shifts(divide, values, num, varied_ref)),
        "syst_band": _combined(values, band_shifts),
    }


def _difference(num: _Side, ref: _Side, *, propagate: bool) -> dict[str, Any]:
    """``n - d``, its error bars and the reference's absolute uncertainty as the band."""
    n, d = num.values, ref.values
    values = n - d
    varied_ref = ref if propagate else ref.nominal()
    band_shifts = {
        name: (up.values() - d, down.values() - d) for name, (up, down) in ref.variations.items()
    }
    return {
        "values": values,
        "errors": _propagated(1.0, -1.0, num.errors, ref.errors) if propagate else num.errors,
        "band": ref.errors,
        "syst_errors": _combined(values, _source_shifts(np.subtract, values, num, varied_ref)),
        "syst_band": _combined(values, band_shifts),
    }


def _pull(num: _Side, ref: _Side) -> dict[str, Any]:
    """``(n - d) / sigma`` with statistical and systematic ``sigma``; ``nan`` where it is zero.

    Both parts are taken on the side of ``n - d`` facing the other histogram,
    as mplhep does for Poisson pulls: the lower uncertainty where ``n > d``,
    the upper one elsewhere.
    """
    difference = num.values - ref.values
    above = difference > 0
    stat = np.where(above, *_propagated(1.0, -1.0, num.errors, ref.errors))
    syst = _combined(difference, _source_shifts(np.subtract, difference, num, ref))
    facing = np.zeros_like(difference) if syst is None else np.where(above, *syst)
    sigma = np.hypot(stat, facing)
    values = np.asarray(np.where(sigma > 0, difference / sigma, np.nan), dtype=float)
    unit = np.where(np.isfinite(values), 1.0, np.nan)
    return {"values": values, "errors": (unit, unit.copy())}


def _asymmetry(num: _Side, ref: _Side) -> dict[str, Any]:
    """``(n - d) / (n + d)`` with both sides propagated; ``nan`` where the sum is zero."""
    n, d = num.values, ref.values

    def asymmetry(top: FloatArray, bottom: FloatArray) -> FloatArray:
        total = top + bottom
        return np.asarray(np.where(total != 0, (top - bottom) / total, np.nan))

    values = asymmetry(n, d)
    total = n + d
    # d/dn = 2d / (n+d)^2, d/dd = -2n / (n+d)^2
    errors = _propagated(2 * d / total**2, -2 * n / total**2, num.errors, ref.errors)
    return {
        "values": values,
        "errors": _where_defined(total != 0, errors),
        "syst_errors": _combined(values, _source_shifts(asymmetry, values, num, ref)),
    }


def _significance(signal: _Side, background: _Side, kind: ComparisonKind) -> dict[str, Any]:
    """Per-bin significance of ``signal`` over ``background``, statistical only.

    ``nan`` where the background (or, for ``s/sqrt(s+b)``, the total) is not positive.
    """
    s, b = signal.values, background.values
    if kind == "s/sqrt(b)":
        ok = b > 0
        values = np.where(ok, s / np.sqrt(b), np.nan)
        # d/ds = 1/sqrt(b), d/db = -s / (2 b^1.5)
        errors = _propagated(1 / np.sqrt(b), -s / (2 * b**1.5), signal.errors, background.errors)
    else:
        total = s + b
        ok = total > 0
        values = np.where(ok, s / np.sqrt(total), np.nan)
        # d/ds = (s + 2b) / (2 (s+b)^1.5), d/db = -s / (2 (s+b)^1.5)
        ds = (s + 2 * b) / (2 * total**1.5)
        db = -s / (2 * total**1.5)
        errors = _propagated(ds, db, signal.errors, background.errors)
    return {
        "values": np.asarray(values, dtype=float),
        "errors": _where_defined(ok, errors),
    }


_NONE: _Errors = (np.zeros(1), np.zeros(1))
"""No uncertainty, for the side of :func:`_propagated` that does not enter."""


def _where_defined(defined: np.ndarray, errors: _Errors) -> _Errors:
    """Return ``errors`` with both sides ``nan`` where the comparison is not ``defined``."""
    return (
        np.asarray(np.where(defined, errors[0], np.nan), dtype=float),
        np.asarray(np.where(defined, errors[1], np.nan), dtype=float),
    )


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
    zeros = np.zeros_like(like)
    summary = Uncertainty(
        edges=edges, nominal=like, stat_down=zeros, stat_up=zeros, components=shifts
    )
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
        needs = "counts" if uncertainty == "poisson-ratio" else "a reference band"
        msg = (
            f"uncertainty={uncertainty!r} needs {needs}, which {pair} do not have: "
            "their intervals are independent and always propagated"
        )
        raise ValueError(msg)
    if not same_edges(num.edges, ref.edges):
        msg = "compare requires two inputs with identical bin edges"
        raise BinningError(msg)
    a, b = num.values, ref.values
    a_err, b_err = _intervals(num), _intervals(ref)
    errors: _Errors
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
                errors = (np.ones_like(values), np.ones_like(values))
    values = np.asarray(values, dtype=float)
    errors = _where_defined(np.isfinite(values), errors)
    return Comparison(
        kind=kind,
        label=num.label,
        reference=ref.label,
        values=values,
        errors=errors,
        edges=np.asarray(num.edges, dtype=float),
        reference_points=ref,
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
    the same for ``b``. The two sides are independent and add in quadrature;
    with symmetric errors this is the usual ``hypot(pa * a_err, pb * b_err)``.
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
