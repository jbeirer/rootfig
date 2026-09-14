"""Ratios of histograms with uncertainty propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np

from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError
from rootfig.histograms.build import Histogram, compatible_binning
from rootfig.histograms.systematics import Uncertainty

__all__ = [
    "SIGNIFICANCE_KINDS",
    "Ratio",
    "RatioUncertainty",
    "SignificanceKind",
    "compatible_binning",
    "ratio",
    "significance",
]

RatioUncertainty: TypeAlias = Literal["propagate", "numerator"]
SignificanceKind: TypeAlias = Literal["s/sqrt(b)", "s/sqrt(s+b)"]
"""Per-bin significance estimators: ``S / sqrt(B)`` or ``S / sqrt(S + B)``."""
SIGNIFICANCE_KINDS: tuple[str, ...] = ("significance", "s/sqrt(b)", "s/sqrt(s+b)")
"""Strings accepted by ``ratio=`` for a significance panel (``"significance"`` means S/sqrt(B))."""
"""How ratio uncertainties are computed.

* ``"propagate"`` - numerator and denominator uncertainties both enter the
  error bars (statistical ones uncorrelated, systematic ones source by source).
* ``"numerator"`` - error bars carry only the numerator uncertainty; the
  denominator's relative uncertainty is returned separately as a band around
  one (the usual data/MC convention, mplhep's ``split_ratio``).

Systematic uncertainties of :class:`~rootfig.histograms.Histogram` inputs follow
the same rule, per side.
"""


@dataclass(frozen=True)
class Ratio:
    """Bin-by-bin ratio of two histograms.

    Attributes
    ----------
    values
        ``numerator / denominator``; ``nan`` where the denominator is zero.
    errors
        Uncertainty on ``values`` (see :data:`RatioUncertainty`).
    band
        Relative statistical uncertainty of the denominator, ``sqrt(var_den) / den``;
        use as a band around one. ``nan`` where the denominator is zero.
    edges
        Bin edges shared by both histograms.
    syst_errors
        Systematic uncertainty on ``values`` as ``(down, up)`` (matplotlib's
        ``yerr`` order), following :data:`RatioUncertainty`; ``None`` without
        systematic variations.
    syst_band
        Relative systematic uncertainty of the denominator below and above one,
        ``(down, up)``; ``None`` if the denominator has no variations.
    """

    values: FloatArray
    errors: FloatArray
    band: FloatArray
    edges: FloatArray
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

    def total_band(self) -> tuple[FloatArray, FloatArray]:
        """Relative statistical and systematic denominator uncertainty, ``(down, up)``."""
        if self.syst_band is None:
            return self.band, self.band
        down, up = self.syst_band
        return np.hypot(self.band, down), np.hypot(self.band, up)


def ratio(
    numerator: Hist | Histogram,
    denominator: Hist | Histogram,
    *,
    uncertainty: RatioUncertainty = "propagate",
) -> Ratio:
    """Compute ``numerator / denominator`` bin by bin with uncertainties.

    Statistical uncertainties of numerator and denominator are uncorrelated.
    ``Histogram`` inputs with systematic variations also give
    :attr:`Ratio.syst_errors` and :attr:`Ratio.syst_band`, propagated source by
    source through the varied ratio itself: a source present on both sides varies
    numerator and denominator together (a shared luminosity uncertainty cancels),
    a source on one side only varies that side against the other's nominal
    contents. The shifts of different sources then combine like those of one
    histogram (see :mod:`rootfig.histograms.systematics`). With ``"numerator"``
    only the numerator's sources enter the error bars; the denominator's are the
    band.

    Raises
    ------
    BinningError
        If the histograms do not share the same one-dimensional binning.
    """
    num_variations = numerator.variations if isinstance(numerator, Histogram) else {}
    den_variations = denominator.variations if isinstance(denominator, Histogram) else {}
    if isinstance(numerator, Histogram):
        numerator = numerator.hist
    if isinstance(denominator, Histogram):
        denominator = denominator.hist
    if not compatible_binning(numerator, denominator):
        msg = "ratio requires two one-dimensional histograms with identical bin edges"
        raise BinningError(msg)
    if uncertainty not in ("propagate", "numerator"):  # runtime guard for untyped callers
        msg = f"uncertainty must be 'propagate' or 'numerator', got {uncertainty!r}"
        raise BinningError(msg)
    n = np.asarray(numerator.values(), dtype=float)
    d = np.asarray(denominator.values(), dtype=float)
    vn = np.asarray(numerator.variances(), dtype=float)
    vd = np.asarray(denominator.variances(), dtype=float)
    edges = np.asarray(numerator.axes[0].edges, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):

        def divide(top: FloatArray, bottom: FloatArray) -> FloatArray:
            return np.asarray(np.where((d != 0) & (bottom != 0), top / bottom, np.nan))

        values = divide(n, d)
        band = divide(np.sqrt(vd), np.abs(d))
        if uncertainty == "propagate":
            errors = divide(np.sqrt(vn + n**2 * vd / d**2), np.abs(d))
            varied_denominators = den_variations
        else:
            errors = divide(np.sqrt(vn), np.abs(d))
            varied_denominators = {}  # the denominator's sources are the band
        sources = list(dict.fromkeys([*num_variations, *varied_denominators]))

        def varied(variations: Any, name: str, index: int, nominal: FloatArray) -> FloatArray:
            if name not in variations:
                return nominal
            return np.asarray(variations[name][index].values(), dtype=float)

        ratio_shifts = {
            name: tuple(
                divide(varied(num_variations, name, i, n), varied(varied_denominators, name, i, d))
                - values
                for i in (0, 1)
            )
            for name in sources
        }
        band_shifts = {
            name: tuple(divide(up_or_down.values() - d, d) for up_or_down in pair)
            for name, pair in den_variations.items()
        }
    return Ratio(
        values=values,
        errors=errors,
        band=band,
        edges=edges,
        syst_errors=_combined(edges, values, ratio_shifts),
        syst_band=_combined(edges, values, band_shifts),
    )


def _combined(
    edges: FloatArray, like: FloatArray, shifts: dict[str, Any]
) -> tuple[FloatArray, FloatArray] | None:
    """Combine signed per-source shifts into ``(down, up)`` magnitudes, or ``None`` without any."""
    if not shifts:
        return None
    summary = Uncertainty(edges=edges, nominal=like, stat=np.zeros_like(like), components=shifts)
    return summary.syst_down, summary.syst_up


def significance(signal: Hist, background: Hist, *, kind: SignificanceKind = "s/sqrt(b)") -> Ratio:
    """Per-bin significance of ``signal`` over ``background`` with propagated uncertainties.

    Returned as a :class:`Ratio` (``values``, ``errors``, ``edges``; ``band`` is
    ``nan``) so it can be drawn like a ratio panel. Bins with zero background
    (or zero total for ``"s/sqrt(s+b)"``) are ``nan``.
    """
    if not compatible_binning(signal, background):
        msg = "significance requires two one-dimensional histograms with identical bin edges"
        raise BinningError(msg)
    s = np.asarray(signal.values(), dtype=float)
    b = np.asarray(background.values(), dtype=float)
    vs = np.asarray(signal.variances(), dtype=float)
    vb = np.asarray(background.variances(), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "s/sqrt(b)":
            ok = b > 0
            values = np.where(ok, s / np.sqrt(b), np.nan)
            # d/ds = 1/sqrt(b), d/db = -s / (2 b^1.5)
            errors = np.where(ok, np.sqrt(vs / b + s**2 * vb / (4 * b**3)), np.nan)
        elif kind == "s/sqrt(s+b)":
            total = s + b
            ok = total > 0
            values = np.where(ok, s / np.sqrt(total), np.nan)
            # d/ds = (s + 2b) / (2 (s+b)^1.5), d/db = -s / (2 (s+b)^1.5)
            ds = (s + 2 * b) / (2 * total**1.5)
            db = -s / (2 * total**1.5)
            errors = np.where(ok, np.sqrt(ds**2 * vs + db**2 * vb), np.nan)
        else:  # runtime guard for untyped callers
            msg = f"kind must be 's/sqrt(b)' or 's/sqrt(s+b)', got {kind!r}"  # type: ignore[unreachable]
            raise BinningError(msg)
    return Ratio(
        values=np.asarray(values, dtype=float),
        errors=np.asarray(errors, dtype=float),
        band=np.full(len(s), np.nan),
        edges=np.asarray(signal.axes[0].edges, dtype=float),
    )
