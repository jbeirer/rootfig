"""Efficiencies (pass / total with binomial intervals) and profiles (a statistic of y per x bin)."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError, RootfigWarning
from rootfig.histograms.build import compatible_binning
from rootfig.histograms.intervals import wilson_interval

__all__ = ["Efficiency", "Profile", "ProfileStatistic", "efficiency", "profile"]

ProfileStatistic: TypeAlias = Literal["mean", "std"]
"""What a profile shows per bin: the weighted mean of ``y`` or its standard deviation."""


@dataclass(frozen=True)
class Efficiency:
    """Bin-by-bin efficiency ``passed / total`` with a binomial confidence interval.

    Attributes
    ----------
    values
        The efficiency ``passed / total``; ``nan`` where the total weight is zero
        (an empty bin, or weights that cancel).
    lower, upper
        Bounds of the Wilson score interval (``z`` standard deviations; ``z = 1``
        is the usual 68 % band), computed with the effective number of entries
        for weighted samples (see :func:`efficiency`). The interval always
        contains the value; it is ``nan`` where
        negative weights enter the bin (see :func:`efficiency`), since a
        binomial interval is undefined there while the ratio itself is still
        reported.
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


def efficiency(
    passed: Hist,
    total: Hist,
    *,
    z: float = 1.0,
    label: str = "",
    negative_weights: npt.ArrayLike | None = None,
) -> Efficiency:
    """Compute ``passed / total`` per bin with a Wilson score interval.

    Both histograms must share their binning; ``passed`` should be a subset of
    ``total``. With weights the effective counts ``(sum w)^2 / sum w^2`` of the
    total replace the raw counts in the interval (see
    :func:`~rootfig.histograms.intervals.wilson_interval`, which also says how
    this differs from ROOT). The interval is clipped to ``[0, 1]`` and always
    contains the efficiency (at 0 % and 100 % the respective bound coincides
    with the value).

    The interval is binomial, so it needs non-negative weights. A bin gets
    ``nan`` bounds and a :class:`~rootfig.errors.RootfigWarning` where negative
    weights enter it, as far as can be told: where the total is negative, the
    efficiency outside ``[0, 1]``, or the passing, failing or all entries have a
    sum of squared weights above the square of their sum (non-negative weights
    never do). The sums cannot reveal every negative weight, so
    ``negative_weights``, one flag per bin, marks the bins in which an entry of
    ``total`` has one; :func:`rootfig.efficiency` knows them from filling.

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
    vk = np.asarray(passed.variances(), dtype=float)
    vn = np.asarray(total.variances(), dtype=float)
    # The ratio is defined whenever the total is non-zero; the Wilson interval needs a
    # positive total (a negative sum of weights is not a sample size) and p in [0, 1].
    ok = n != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(ok, k / n, np.nan)
    lower, upper = wilson_interval(k, n, vn, z)
    signed = _signed(k, vk, n, vn) | _signed(n - k, vn - vk, n, vn) | _signed(n, vn, n, vn)
    if negative_weights is not None:
        signed |= np.asarray(negative_weights, dtype=bool)
    lower, upper = np.where(signed, np.nan, lower), np.where(signed, np.nan, upper)
    undefined = int(np.count_nonzero(ok & np.isnan(lower)))
    if undefined:
        warnings.warn(
            f"{label + ': ' if label else ''}{undefined} bin(s) hold negative weights (a "
            "negative total, an efficiency outside [0, 1] or a signed entry); no binomial "
            "confidence interval is drawn for them",
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


def _signed(total: FloatArray, squares: FloatArray, scale: FloatArray, spread: FloatArray) -> Any:
    """Return where weights summing to ``total`` with ``squares`` must include a negative one.

    Non-negative weights never sum below zero, and the square of their sum is
    at least the sum of their squares. ``scale`` and ``spread`` (the whole
    bin's sums) set the round-off below which neither counts.
    """
    tolerance = 1e-9
    return (total < -tolerance * np.abs(scale)) | (squares > total**2 + tolerance * spread)


@dataclass(frozen=True)
class Profile:
    """A statistic of ``y`` in bins of ``x``.

    Attributes
    ----------
    values
        The weighted mean (``statistic="mean"``) or standard deviation
        (``"std"``) of ``y`` per bin; ``nan`` for empty bins (zero total
        weight). The standard deviation (and hence the error) is also ``nan``
        where negative weights make the total weight negative or the weighted
        variance negative; the mean is still reported there.
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
        # The weighted mean is defined whenever the weights do not cancel; a weighted
        # variance needs a positive total weight and a non-negative second moment.
        ok = sw != 0
        mean = np.where(ok, swy / sw, np.nan)
        residual = y - np.where(ok, mean, 0.0)[index]
        swr2 = np.bincount(index, weights=w * residual * residual, minlength=n_bins)
        variance = np.where(sw > 0, swr2 / sw, np.nan)
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
