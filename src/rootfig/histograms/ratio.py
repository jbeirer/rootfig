"""Ratios of histograms with uncertainty propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError
from rootfig.histograms.build import Histogram, compatible_binning
from rootfig.histograms.systematics import uncertainty as uncertainty_of

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

* ``"propagate"`` - numerator and denominator uncertainties are combined in
  quadrature (uncorrelated) into the error bars.
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

    ``Histogram`` inputs with systematic variations also give
    :attr:`Ratio.syst_errors` and :attr:`Ratio.syst_band`. Numerator and
    denominator are treated as uncorrelated: for positive contents with
    ``"propagate"`` the ratio's upper uncertainty combines the numerator's upward
    and the denominator's downward shift, and vice versa. For signed contents the
    sides follow the signs of the derivatives of ``numerator / denominator``.

    Raises
    ------
    BinningError
        If the histograms do not share the same one-dimensional binning.
    """
    num_syst, den_syst = _systematics(numerator), _systematics(denominator)
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
    zero = np.zeros_like(d)
    num_down, num_up = num_syst or (zero, zero)
    den_down, den_up = den_syst or (zero, zero)
    with np.errstate(divide="ignore", invalid="ignore"):

        def relative(shift: FloatArray) -> FloatArray:
            return np.asarray(np.where(d != 0, shift / np.abs(d), np.nan), dtype=float)

        values = np.where(d != 0, n / d, np.nan)
        band = relative(np.sqrt(vd))
        syst_band = (
            (
                relative(np.where(d >= 0, den_down, den_up)),
                relative(np.where(d >= 0, den_up, den_down)),
            )
            if den_syst
            else None
        )
        # d(n/d)/dn = 1/d; d(n/d)/dd = -n/d**2. Signed bins
        # can reverse which side of an asymmetric uncertainty contributes.
        num_low = np.where(d >= 0, num_down, num_up)
        num_high = np.where(d >= 0, num_up, num_down)
        den_low = np.where(n >= 0, den_up, den_down)
        den_high = np.where(n >= 0, den_down, den_up)
        if uncertainty == "propagate":
            errors = relative(np.sqrt(vn + n**2 * vd / d**2))
            syst_errors = (
                (
                    relative(np.hypot(num_low, n * den_low / d)),
                    relative(np.hypot(num_high, n * den_high / d)),
                )
                if num_syst or den_syst
                else None
            )
        else:
            errors = relative(np.sqrt(vn))
            syst_errors = (relative(num_low), relative(num_high)) if num_syst else None
    return Ratio(
        values=np.asarray(values, dtype=float),
        errors=errors,
        band=band,
        edges=np.asarray(numerator.axes[0].edges, dtype=float),
        syst_errors=syst_errors,
        syst_band=syst_band,
    )


def _systematics(histogram: Hist | Histogram) -> tuple[FloatArray, FloatArray] | None:
    """Systematic ``(down, up)`` uncertainty of a histogram with variations, else ``None``."""
    if not (isinstance(histogram, Histogram) and histogram.variations):
        return None
    summary = uncertainty_of(histogram)
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
