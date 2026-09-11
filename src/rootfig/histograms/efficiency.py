"""Efficiencies (pass / total with binomial intervals) and profiles (a statistic of y per x bin)."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError, RootfigWarning
from rootfig.histograms.ratio import compatible_binning

__all__ = ["Efficiency", "Profile", "ProfileStatistic", "efficiency", "profile"]

ProfileStatistic: TypeAlias = Literal["mean", "std"]
"""What a profile shows per bin: the weighted mean of ``y`` or its standard deviation."""


@dataclass(frozen=True)
class Efficiency:
    """Bin-by-bin efficiency ``passed / total`` with a binomial confidence interval.

    Attributes
    ----------
    values
        The efficiency; ``nan`` where ``total`` is empty.
    lower, upper
        Bounds of the Wilson score interval (``z`` standard deviations; ``z = 1``
        is the usual 68 % band), computed with the effective number of entries
        so weighted samples get sensible intervals. The interval always contains
        the value; it is ``nan`` where the value is outside ``[0, 1]`` (negative
        weights), since a binomial interval is undefined there.
    edges
        Bin edges.
    label
        Legend label.
    """

    values: FloatArray
    lower: FloatArray
    upper: FloatArray
    edges: FloatArray
    label: str = ""

    @property
    def centers(self) -> FloatArray:
        """Bin centres."""
        return np.asarray(0.5 * (self.edges[1:] + self.edges[:-1]), dtype=float)

    @property
    def half_widths(self) -> FloatArray:
        """Half bin widths."""
        return np.asarray(0.5 * np.diff(self.edges), dtype=float)

    @property
    def errors(self) -> tuple[FloatArray, FloatArray]:
        """``(values - lower, upper - values)``, ready for ``yerr``."""
        return (
            np.asarray(self.values - self.lower, dtype=float),
            np.asarray(self.upper - self.values, dtype=float),
        )


def efficiency(passed: Hist, total: Hist, *, z: float = 1.0, label: str = "") -> Efficiency:
    """Compute ``passed / total`` per bin with a Wilson score interval.

    Both histograms must share their binning; ``passed`` should be a subset of
    ``total``. With weights the effective counts ``(sum w)^2 / sum w^2`` of the
    total replace the raw counts in the interval. The interval is clipped to
    ``[0, 1]`` and always contains the efficiency (at 0 % and 100 % the
    respective bound coincides with the value). Bins whose efficiency falls
    outside ``[0, 1]``, which can only happen with negative weights, get ``nan``
    bounds and a :class:`~rootfig.errors.RootfigWarning`: no binomial interval
    describes them.

    Raises
    ------
    BinningError
        If the binnings differ or ``z`` is not a positive finite number.
    """
    if not compatible_binning(passed, total):
        msg = "efficiency requires two one-dimensional histograms with identical bin edges"
        raise BinningError(msg)
    if not (math.isfinite(z) and z > 0):
        msg = f"z must be a positive finite number of standard deviations, got {z!r}"
        raise BinningError(msg)
    k = np.asarray(passed.values(), dtype=float)
    n = np.asarray(total.values(), dtype=float)
    vn = np.asarray(total.variances(), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = n > 0
        p = np.where(ok, k / n, np.nan)
        n_eff = np.where(vn > 0, n**2 / vn, n)
        z2 = z * z
        denominator = 1.0 + z2 / n_eff
        centre = (p + z2 / (2.0 * n_eff)) / denominator
        half = (z / denominator) * np.sqrt(p * (1.0 - p) / n_eff + z2 / (4.0 * n_eff**2))
        valid = ok & (p >= 0.0) & (p <= 1.0)
        # the Wilson interval contains p by construction; guard against round-off at 0 and 1
        lower = np.where(valid, np.minimum(np.clip(centre - half, 0.0, 1.0), p), np.nan)
        upper = np.where(valid, np.maximum(np.clip(centre + half, 0.0, 1.0), p), np.nan)
    undefined = int(np.count_nonzero(ok & ~valid))
    if undefined:
        warnings.warn(
            f"{label + ': ' if label else ''}{undefined} bin(s) have an efficiency outside "
            "[0, 1] (negative weights?); no confidence interval is drawn for them",
            RootfigWarning,
            stacklevel=2,
        )
    return Efficiency(
        values=np.asarray(p, dtype=float),
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        edges=np.asarray(total.axes[0].edges, dtype=float),
        label=label,
    )


@dataclass(frozen=True)
class Profile:
    """A statistic of ``y`` in bins of ``x``.

    Attributes
    ----------
    values
        The weighted mean (``statistic="mean"``) or standard deviation
        (``"std"``) of ``y`` per bin; ``nan`` for empty bins, and ``nan`` for
        the standard deviation where negative weights make the weighted
        variance negative.
    errors
        Standard error: ``std / sqrt(n_eff)`` for the mean, ``std / sqrt(2 n_eff)``
        for the standard deviation, with the effective entries ``n_eff``.
    counts
        Sum of weights per bin.
    edges
        Bin edges.
    statistic, label
        What is shown and the legend label.
    """

    values: FloatArray
    errors: FloatArray
    counts: FloatArray
    edges: FloatArray
    statistic: ProfileStatistic = "mean"
    label: str = ""

    @property
    def centers(self) -> FloatArray:
        """Bin centres."""
        return np.asarray(0.5 * (self.edges[1:] + self.edges[:-1]), dtype=float)

    @property
    def half_widths(self) -> FloatArray:
        """Half bin widths."""
        return np.asarray(0.5 * np.diff(self.edges), dtype=float)


def profile(
    x: FloatArray,
    y: FloatArray,
    edges: FloatArray,
    *,
    weights: FloatArray | None = None,
    statistic: ProfileStatistic = "mean",
    label: str = "",
) -> Profile:
    """Bin ``x`` with ``edges`` and compute the weighted mean or standard deviation of ``y``.

    The variance is computed from the deviations from the bin mean (two passes),
    so a narrow spread at a large offset, e.g. ``std([1e9, 1e9 + 1]) = 0.5``, is
    not lost to cancellation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.asarray(edges, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    if not (x.shape == y.shape == w.shape):
        msg = "x, y and weights must have the same length"
        raise BinningError(msg)
    if statistic not in ("mean", "std"):
        msg = f"statistic must be 'mean' or 'std', got {statistic!r}"
        raise BinningError(msg)
    n_bins = len(edges) - 1
    index = np.digitize(x, edges) - 1
    inside = (index >= 0) & (index < n_bins)
    index, y, w = index[inside], y[inside], w[inside]
    sw = np.bincount(index, weights=w, minlength=n_bins)
    sw2 = np.bincount(index, weights=w * w, minlength=n_bins)
    swy = np.bincount(index, weights=w * y, minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = sw > 0
        mean = np.where(ok, swy / sw, np.nan)
        residual = y - np.where(ok, mean, 0.0)[index]
        swr2 = np.bincount(index, weights=w * residual * residual, minlength=n_bins)
        variance = np.where(ok, swr2 / sw, np.nan)
        variance = np.where(variance < 0, np.nan, variance)  # negative weights: undefined
        std = np.sqrt(variance)
        n_eff = np.where(sw2 > 0, sw**2 / sw2, 0.0)
        if statistic == "mean":
            values, errors = mean, np.where(n_eff > 0, std / np.sqrt(n_eff), np.nan)
        else:
            values, errors = std, np.where(n_eff > 0, std / np.sqrt(2.0 * n_eff), np.nan)
    return Profile(
        values=np.asarray(values, dtype=float),
        errors=np.asarray(errors, dtype=float),
        counts=np.asarray(sw, dtype=float),
        edges=edges,
        statistic=statistic,
        label=label,
    )
