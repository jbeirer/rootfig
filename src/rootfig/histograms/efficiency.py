"""Efficiencies (pass / total with confidence intervals) and profiles (a statistic of y by x)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError, RootfigWarning
from rootfig.histograms.bayesian import Bayesian, bayesian_interval
from rootfig.histograms.binomial import (
    EfficiencyInterval,
    efficiency_interval,
    is_unweighted,
    resolve_interval,
)
from rootfig.histograms.build import compatible_binning
from rootfig.histograms.intervals import ONE_SIGMA, check_cl

__all__ = ["Efficiency", "Profile", "ProfileStatistic", "efficiency", "profile"]

ProfileStatistic: TypeAlias = Literal["mean", "std"]
"""What a profile shows per bin: the weighted mean of ``y`` or its standard deviation."""


@dataclass(frozen=True)
class Efficiency:
    """Bin-by-bin efficiency ``passed / total`` with a confidence interval.

    Attributes
    ----------
    values
        The efficiency ``passed / total``, or the posterior's mean or mode for a
        Bayesian interval (as ROOT's ``TEfficiency``); ``nan`` where the total
        weight is zero (an empty bin, or weights that cancel), except for empty
        bins that are shown (see :func:`efficiency`).
    lower, upper
        Bounds of the confidence interval (see :func:`efficiency` for the
        method and confidence level). It is ``nan`` where none is defined (see
        :func:`efficiency`) while the ratio itself is still reported.
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
        """``(values - lower, upper - values)``, ready for ``yerr``.

        Never negative: a posterior's mode can lie outside its central interval,
        which then reaches only one way.
        """
        with np.errstate(invalid="ignore"):
            return (
                np.asarray(np.maximum(self.values - self.lower, 0.0), dtype=float),
                np.asarray(np.maximum(self.upper - self.values, 0.0), dtype=float),
            )


def efficiency(
    passed: Hist,
    total: Hist,
    *,
    cl: float = ONE_SIGMA,
    label: str = "",
    negative_weights: npt.ArrayLike | None = None,
    interval: EfficiencyInterval = "auto",
    show_empty: bool = False,
) -> Efficiency:
    """Compute ``passed / total`` per bin with a confidence interval of confidence level ``cl``.

    Both histograms must share their binning; ``passed`` should be a subset of
    ``total``. ``interval`` (see
    :data:`~rootfig.histograms.binomial.EfficiencyInterval`) defaults to what
    ROOT's ``TEfficiency`` gives: Clopper-Pearson when both histograms are
    unweighted (their sum of weights equals their sum of squared weights, as
    ROOT decides), else the normal approximation of a weighted pass fraction.
    The interval is clipped to ``[0, 1]``. A :class:`~rootfig.histograms.bayesian.Bayesian`
    interval (``"jeffreys"``, ``"uniform"`` or any prior) reports the posterior's
    mean or mode as the efficiency, as ``TEfficiency`` does.

    An empty bin has no efficiency; ``show_empty=True`` shows it as ROOT's
    ``TGraphAsymmErrors::Divide`` does with ``"e0"``: 0 in ``[0, 1]``, or the
    prior's mean and interval for a Bayesian interval. Only a bin without
    entries is empty (no sum of weights and no sum of squared weights): one
    whose signed weights cancel holds entries and stays undefined.

    The intervals other than ``"normal"`` are binomial, so they need
    non-negative weights. A bin gets ``nan`` bounds from them and a
    :class:`~rootfig.errors.RootfigWarning` where negative weights enter it, as
    far as can be told: where the total is negative, the efficiency outside
    ``[0, 1]``, or the passing, failing or all entries have a sum of squared
    weights above the square of their sum (non-negative weights never do). The
    sums cannot reveal every negative weight, so ``negative_weights``, one flag
    per bin, marks the bins in which an entry of ``total`` has one;
    :func:`rootfig.efficiency` knows them from filling. ``"normal"``
    propagates the sums to first order, which holds for signed weights too: it
    is ``nan`` (with the warning) only where the total is not positive, the
    efficiency lies outside ``[0, 1]`` or the variances cannot belong to a
    subset of the total (see
    :func:`~rootfig.histograms.binomial.normal_interval`).

    Raises
    ------
    BinningError
        If the binnings differ.
    ValueError
        If ``cl`` is not a confidence level between 0 and 1, for an unknown
        ``interval``, or a method of counts (Clopper-Pearson, Wilson,
        Agresti-Coull) for weighted histograms.
    """
    if not compatible_binning(passed, total):
        msg = "efficiency requires two one-dimensional histograms with identical bin edges"
        raise BinningError(msg)
    check_cl(cl)
    k = np.asarray(passed.values(), dtype=float)
    n = np.asarray(total.values(), dtype=float)
    vk = np.asarray(passed.variances(), dtype=float)
    vn = np.asarray(total.variances(), dtype=float)
    unweighted = is_unweighted(k.sum(), vk.sum()) and is_unweighted(n.sum(), vn.sum())
    method = resolve_interval(interval, unweighted, label)
    ok = n != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ok, k / n, np.nan)
    p, lower, upper = efficiency_interval(method, k, n, vk, vn, cl=cl, weighted=not unweighted)
    if method == "normal":
        problem = (
            "a negative total, an efficiency outside [0, 1] or a passed variance too large "
            "for a subset of the total"
        )
    else:  # binomial intervals need non-negative weights
        problem = (
            "negative weights (a negative total, an efficiency outside [0, 1] or a signed entry) "
            "or a positive total without a variance"
        )
        signed = _signed(k, vk, n, vn) | _signed(n - k, vn - vk, n, vn) | _signed(n, vn, n, vn)
        if negative_weights is not None:
            signed |= np.asarray(negative_weights, dtype=bool)
        lower, upper = np.where(signed, np.nan, lower), np.where(signed, np.nan, upper)
        # no posterior where weights are signed: the plain ratio, as for the other methods
        p = np.where(signed | np.isnan(p), ratio, p)
    undefined = int(np.count_nonzero(ok & np.isnan(lower)))
    if undefined:
        name = "Bayesian" if isinstance(method, Bayesian) else method
        warnings.warn(
            f"{label + ': ' if label else ''}{undefined} bin(s) hold {problem}; no "
            f"{name} confidence interval is drawn for them",
            RootfigWarning,
            stacklevel=2,
        )
    p, lower, upper = (np.where(ok, a, np.nan) for a in (p, lower, upper))
    # no entries at all, not weights that cancel: those leave the efficiency undefined
    empty = (n == 0) & (vn == 0)
    if show_empty and empty.any():
        if isinstance(method, Bayesian):  # the prior: the posterior of no entries
            prior = bayesian_interval(method, np.zeros(1), np.zeros(1), cl=cl)
            fill = [float(side[0]) for side in prior]
        else:
            fill = [0.0, 0.0, 1.0]
        p, lower, upper = (
            np.where(empty, f, a) for f, a in zip(fill, (p, lower, upper), strict=True)
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
