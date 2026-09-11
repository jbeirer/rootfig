"""Ratios of histograms with uncertainty propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError

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
  one (the usual data/MC convention).
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
        Relative uncertainty of the denominator, ``sqrt(var_den) / den``; use as
        a band around one. ``nan`` where the denominator is zero.
    edges
        Bin edges shared by both histograms.
    """

    values: FloatArray
    errors: FloatArray
    band: FloatArray
    edges: FloatArray

    @property
    def centers(self) -> FloatArray:
        """Bin centres."""
        return np.asarray(0.5 * (self.edges[1:] + self.edges[:-1]), dtype=float)

    @property
    def half_widths(self) -> FloatArray:
        """Half bin widths (for horizontal error bars)."""
        return np.asarray(0.5 * np.diff(self.edges), dtype=float)


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


def ratio(
    numerator: Hist,
    denominator: Hist,
    *,
    uncertainty: RatioUncertainty = "propagate",
) -> Ratio:
    """Compute ``numerator / denominator`` bin by bin with uncertainties.

    Raises
    ------
    BinningError
        If the histograms do not share the same one-dimensional binning.
    """
    if not compatible_binning(numerator, denominator):
        msg = "ratio requires two one-dimensional histograms with identical bin edges"
        raise BinningError(msg)
    n = np.asarray(numerator.values(), dtype=float)
    d = np.asarray(denominator.values(), dtype=float)
    vn = np.asarray(numerator.variances(), dtype=float)
    vd = np.asarray(denominator.variances(), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(d != 0, n / d, np.nan)
        band = np.where(d != 0, np.sqrt(vd) / np.abs(d), np.nan)
        if uncertainty == "propagate":
            errors = np.where(d != 0, np.sqrt(vn / d**2 + n**2 * vd / d**4), np.nan)
        elif uncertainty == "numerator":
            errors = np.where(d != 0, np.sqrt(vn) / np.abs(d), np.nan)
        else:  # runtime guard for untyped callers
            msg = f"uncertainty must be 'propagate' or 'numerator', got {uncertainty!r}"  # type: ignore[unreachable]
            raise BinningError(msg)
    return Ratio(
        values=np.asarray(values, dtype=float),
        errors=np.asarray(errors, dtype=float),
        band=np.asarray(band, dtype=float),
        edges=np.asarray(numerator.axes[0].edges, dtype=float),
    )


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
