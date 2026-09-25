"""Confidence intervals of efficiencies: the methods of ROOT's ``TEfficiency``.

``"auto"`` is what ``TEfficiency`` gives (and ``TGraphAsymmErrors::Divide`` by
default, whose frequentist options also fall back to the normal approximation
for weighted histograms in ROOT 6.40): Clopper-Pearson for unweighted counts,
the normal approximation for weighted entries. Every other name is the method
ROOT means by it, for the entries ROOT allows it for; ``"wilson-effective"`` is
rootfig's own. The Bayesian intervals live in :mod:`~rootfig.histograms.bayesian`,
Feldman-Cousins in :mod:`~rootfig.histograms.feldman_cousins`.
"""

from __future__ import annotations

import typing
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from rootfig._typing import FloatArray
from rootfig.histograms.bayesian import Bayesian, bayesian_interval
from rootfig.histograms.feldman_cousins import feldman_cousins
from rootfig.histograms.intervals import ONE_SIGMA, check_cl, normal_quantile

__all__ = [
    "Bayesian",
    "EfficiencyInterval",
    "EfficiencyMethod",
    "Method",
    "agresti_coull",
    "check_interval",
    "clopper_pearson",
    "efficiency_interval",
    "feldman_cousins",
    "is_unweighted",
    "mid_p",
    "normal_interval",
    "resolve_interval",
    "wilson_interval",
]

EfficiencyMethod: TypeAlias = Literal[
    "auto",
    "clopper-pearson",
    "normal",
    "wilson",
    "wilson-effective",
    "agresti-coull",
    "feldman-cousins",
    "mid-p",
    "jeffreys",
    "uniform",
]
"""The named methods of an efficiency's confidence interval.

* ``"clopper-pearson"`` - the exact binomial interval of counts, never covering
  less than the requested probability; unweighted entries only.
* ``"normal"`` - the efficiency plus or minus ``z`` standard deviations of the
  weighted pass fraction, clipped to ``[0, 1]``: no width at 0 and 1.
* ``"wilson"`` - the Wilson score interval of counts; unweighted entries only.
* ``"agresti-coull"`` - the Agresti-Coull interval of counts: the normal
  approximation around the Wilson centre; unweighted entries only.
* ``"feldman-cousins"`` - the Neyman construction with Feldman and Cousins'
  likelihood-ratio ordering; unweighted entries only.
* ``"mid-p"`` - Lancaster's mid-P interval, a less conservative Clopper-Pearson;
  unweighted entries only.
* ``"jeffreys"``, ``"uniform"`` - the Bayesian intervals of the priors
  ``Beta(0.5, 0.5)`` and ``Beta(1, 1)`` (see :class:`Bayesian`, which also
  takes any other prior); weighted entries too.
* ``"wilson-effective"`` - the Wilson score interval of the effective entries
  ``(sum w)^2 / sum w^2``: rootfig's extension to weighted entries, for which
  ROOT's frequentist intervals fall back to the normal approximation. An
  approximation for non-negative weights, convenient but without guaranteed
  coverage for arbitrary weights. The same as ``"wilson"`` for counts.
* ``"auto"`` - ROOT's default: ``"clopper-pearson"`` for unweighted entries,
  ``"normal"`` otherwise.
"""

EfficiencyInterval: TypeAlias = EfficiencyMethod | Bayesian
"""How the confidence interval of an efficiency is computed: a named method or a prior."""

Method: TypeAlias = (
    Literal[
        "clopper-pearson",
        "normal",
        "wilson",
        "wilson-effective",
        "agresti-coull",
        "feldman-cousins",
        "mid-p",
    ]
    | Bayesian
)
"""A resolved :data:`EfficiencyInterval`: no ``"auto"``, the named priors as :class:`Bayesian`."""

_METHODS: tuple[str, ...] = typing.get_args(EfficiencyMethod)
_COUNTS_ONLY = ("clopper-pearson", "wilson", "agresti-coull", "feldman-cousins", "mid-p")
"""Methods of counts: ROOT refuses them for weighted entries and falls back to "normal"."""
_PRIORS = {"jeffreys": Bayesian(0.5, 0.5), "uniform": Bayesian(1.0, 1.0)}

_ROUND_OFF = 1e-9
"""Relative size below which a negative variance of the normal approximation is round-off."""

_UNWEIGHTED_TOLERANCE = 1e-12
"""Relative tolerance of ``TEfficiency``'s test for unweighted entries, ``sum w == sum w^2``.

ROOT uses it for double-precision histograms (``TH1D``), like rootfig's contents;
``TH1F`` gets ``1e-5``.
"""


def is_unweighted(sum_weights: float, sum_squares: float) -> bool:
    """Return True if entries summing to ``sum_weights`` count as unweighted, as ROOT decides.

    ``TEfficiency`` treats a histogram as unweighted when its sum of weights
    equals its sum of squared weights (relative ``1e-12`` for ``TH1D``), as it
    does when every entry has weight 1.
    """
    difference = abs(sum_weights - sum_squares)
    return difference <= 0.5 * _UNWEIGHTED_TOLERANCE * (abs(sum_weights) + abs(sum_squares))


def check_interval(interval: object) -> None:
    """Raise ``ValueError`` unless ``interval`` names an :data:`EfficiencyInterval`."""
    if not isinstance(interval, Bayesian) and interval not in _METHODS:
        msg = (
            f"interval must be one of {', '.join(map(repr, _METHODS))} or a rootfig.Bayesian "
            f"prior, got {interval!r}"
        )
        raise ValueError(msg)


def resolve_interval(interval: EfficiencyInterval, unweighted: bool, context: str = "") -> Method:
    """Return the method ``interval`` stands for: ``"auto"`` resolved like ``TEfficiency``.

    ROOT falls back to the normal approximation, with a warning, when a method
    of counts is asked for weighted entries; rootfig raises instead, so an
    explicit method never silently changes. The named priors become their
    :class:`Bayesian`.

    Raises
    ------
    ValueError
        For an unknown ``interval``, or a method of counts for weighted entries.
    """
    check_interval(interval)
    if isinstance(interval, Bayesian):
        return interval
    if interval == "auto":
        return "clopper-pearson" if unweighted else "normal"
    if interval in _PRIORS:
        return _PRIORS[interval]
    if interval in _COUNTS_ONLY and not unweighted:
        msg = (
            f"{context + ': ' if context else ''}interval={interval!r} needs unweighted "
            "entries; use 'normal' (ROOT's choice for weighted entries, also what 'auto' "
            "picks), a Bayesian interval ('jeffreys', 'uniform') or 'wilson-effective' "
            "(Wilson with the effective entries)"
        )
        raise ValueError(msg)
    return typing.cast("Method", interval)


def efficiency_interval(
    method: Method,
    passed: npt.ArrayLike,
    total: npt.ArrayLike,
    passed_variance: npt.ArrayLike,
    total_variance: npt.ArrayLike,
    *,
    cl: float = ONE_SIGMA,
    weighted: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the efficiency and its interval by a resolved ``method``: ``(value, lower, upper)``.

    The efficiency is ``passed / total`` (``nan`` for no total), except for a
    :class:`Bayesian` method, whose posterior sets it; that one scales the sums
    to effective entries when ``weighted``.

    Raises
    ------
    ValueError
        For ``"auto"`` and the named priors, which :func:`resolve_interval`
        turns into a method first, or an invalid ``cl``.
    """
    k = np.asarray(passed, dtype=float)
    n = np.asarray(total, dtype=float)
    if isinstance(method, Bayesian):
        return bayesian_interval(method, k, n, total_variance if weighted else None, cl)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.asarray(np.where(n != 0, k / n, np.nan), dtype=float)
    match method:
        case "clopper-pearson":
            bounds = clopper_pearson(k, n, cl)
        case "normal":
            bounds = normal_interval(k, n, passed_variance, total_variance, cl)
        case "wilson":
            bounds = wilson_interval(k, n, n, cl)  # counts: the variance is the count
        case "wilson-effective":
            bounds = wilson_interval(k, n, total_variance, cl)
        case "agresti-coull":
            bounds = agresti_coull(k, n, cl)
        case "feldman-cousins":
            bounds = feldman_cousins(k, n, cl)
        case "mid-p":
            bounds = mid_p(k, n, cl)
        case _:
            msg = f"efficiency_interval needs a resolved method, got {method!r}"  # type: ignore[unreachable]
            raise ValueError(msg)
    return (value, *bounds)


def _counts(passed: npt.ArrayLike, total: npt.ArrayLike) -> tuple[FloatArray, FloatArray, Any]:
    """Return ``passed``, ``total`` broadcast, and where they are a valid pair of counts."""
    k, n = np.broadcast_arrays(np.asarray(passed, dtype=float), np.asarray(total, dtype=float))
    return k, n, (n > 0) & (k >= 0) & (k <= n)


def clopper_pearson(
    passed: npt.ArrayLike, total: npt.ArrayLike, cl: float = ONE_SIGMA
) -> tuple[FloatArray, FloatArray]:
    """Return the Clopper-Pearson interval of ``passed`` of ``total`` counts: ``(lower, upper)``.

    The bounds are the efficiencies at which ``passed`` or more, and ``passed``
    or fewer, entries pass with probability ``(1 - cl) / 2``: beta quantiles
    (SciPy's inverse incomplete beta function), as
    ``TEfficiency::ClopperPearson`` computes them. The lower bound is 0 for no
    passing entry, the upper 1 when all pass. ``nan`` where the total is not
    positive or ``passed`` lies outside ``[0, total]``.
    """
    from scipy.special import betaincinv  # noqa: PLC0415 - 0.3 s to import, only when used

    tail = check_cl(cl)
    k, n, valid = _counts(passed, total)
    with np.errstate(invalid="ignore"):
        lower = np.where(k > 0, betaincinv(k, n - k + 1.0, tail), 0.0)
        upper = np.where(k < n, betaincinv(k + 1.0, n - k, 1.0 - tail), 1.0)
    return (
        np.asarray(np.where(valid, lower, np.nan), dtype=float),
        np.asarray(np.where(valid, upper, np.nan), dtype=float),
    )


def normal_interval(
    passed: npt.ArrayLike,
    total: npt.ArrayLike,
    passed_variance: npt.ArrayLike,
    total_variance: npt.ArrayLike,
    cl: float = ONE_SIGMA,
) -> tuple[FloatArray, FloatArray]:
    """Return the normal approximation of the efficiency ``passed / total``: ``(lower, upper)``.

    The efficiency ``p`` plus or minus ``z`` times the standard deviation of a
    weighted pass fraction (``z`` standard deviations span ``cl``: 1 for
    :data:`~rootfig.histograms.intervals.ONE_SIGMA`), ``sqrt(passed_variance
    (1 - 2 p) + total_variance p^2) / total`` (``sqrt(p (1 - p) / total)`` for
    counts), clipped to
    ``[0, 1]``, as ROOT's ``TEfficiency`` computes it for weighted entries. The
    variance is that of the ratio of the two sums to first order, so it also
    holds for signed weights. ``nan`` where the total is not positive or the
    efficiency lies outside ``[0, 1]``.

    The variance is ``passed_variance (1 - p)^2 + (total_variance -
    passed_variance) p^2``, never negative when the passing entries are a
    subset of all of them, whatever the signs of their weights. A variance
    below zero by more than round-off comes from sums that cannot be a subset
    (a passed variance above the total's) and gives ``nan``, as ROOT's square
    root of it does, rather than an interval of no width.
    """
    z = normal_quantile(cl)
    k = np.asarray(passed, dtype=float)
    n = np.asarray(total, dtype=float)
    vk = np.asarray(passed_variance, dtype=float)
    vn = np.asarray(total_variance, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n != 0, k / n, np.nan)
        first, second = vk * (1.0 - 2.0 * p), vn * p * p
        variance = first + second
        subset = variance >= -_ROUND_OFF * (np.abs(first) + np.abs(second))
        sigma = np.sqrt(np.maximum(variance, 0.0)) / n
        valid = (n > 0) & (p >= 0.0) & (p <= 1.0) & subset
        lower = np.where(valid, np.maximum(p - z * sigma, 0.0), np.nan)
        upper = np.where(valid, np.minimum(p + z * sigma, 1.0), np.nan)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def wilson_interval(
    passed: npt.ArrayLike,
    total: npt.ArrayLike,
    total_variance: npt.ArrayLike,
    cl: float = ONE_SIGMA,
) -> tuple[FloatArray, FloatArray]:
    """Return the Wilson score interval of the efficiency ``passed / total``: ``(lower, upper)``.

    ``passed`` sums the weights of a subset of the entries ``total`` sums, so the
    interval is binomial. For counts (``total_variance = total``) it is ROOT's
    ``TEfficiency::Wilson``. With weights, the effective entries of the total,
    ``total**2 / total_variance``, take the place of its count: the weighted
    pass fraction of entries passing with probability ``e`` has the variance
    ``e (1 - e) / n_eff``, and the score interval inverts it. Unlike the normal
    approximation it keeps a width at 0 and 1. The interval covers ``cl``, is
    clipped to ``[0, 1]`` and always contains the efficiency. It is ``nan``
    where the total is not positive, its variance not positive (no weights sum
    to a positive total with no variance) or the efficiency lies outside
    ``[0, 1]``.
    """
    z = normal_quantile(cl)
    k = np.asarray(passed, dtype=float)
    n = np.asarray(total, dtype=float)
    vn = np.asarray(total_variance, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n != 0, k / n, np.nan)
        n_eff = np.where(vn > 0, n**2 / vn, np.nan)
        z2 = z * z
        denominator = 1.0 + z2 / n_eff
        centre = (p + z2 / (2.0 * n_eff)) / denominator
        half = (z / denominator) * np.sqrt(p * (1.0 - p) / n_eff + z2 / (4.0 * n_eff**2))
        valid = (n > 0) & (p >= 0.0) & (p <= 1.0)
        # the Wilson interval contains p by construction; guard against round-off at 0 and 1
        lower = np.where(valid, np.minimum(np.clip(centre - half, 0.0, 1.0), p), np.nan)
        upper = np.where(valid, np.maximum(np.clip(centre + half, 0.0, 1.0), p), np.nan)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def agresti_coull(
    passed: npt.ArrayLike, total: npt.ArrayLike, cl: float = ONE_SIGMA
) -> tuple[FloatArray, FloatArray]:
    """Return the Agresti-Coull interval of ``passed`` of ``total`` counts: ``(lower, upper)``.

    The normal approximation around the Wilson centre, ``p~ = (passed + z^2 /
    2) / (total + z^2)``, of half width ``z sqrt(p~ (1 - p~) / (total + z^2))``,
    clipped to ``[0, 1]``, as ``TEfficiency::AgrestiCoull``. ``nan`` where the
    total is not positive or ``passed`` lies outside ``[0, total]``.
    """
    z = normal_quantile(cl)
    k, n, valid = _counts(passed, total)
    with np.errstate(divide="ignore", invalid="ignore"):
        centre = (k + 0.5 * z * z) / (n + z * z)
        half = z * np.sqrt(centre * (1.0 - centre) / (n + z * z))
        lower = np.where(valid, np.maximum(centre - half, 0.0), np.nan)
        upper = np.where(valid, np.minimum(centre + half, 1.0), np.nan)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def mid_p(
    passed: npt.ArrayLike, total: npt.ArrayLike, cl: float = ONE_SIGMA
) -> tuple[FloatArray, FloatArray]:
    """Return Lancaster's mid-P interval of ``passed`` of ``total`` counts: ``(lower, upper)``.

    Clopper-Pearson with half the probability of the observed count: the
    bounds are where ``P(N < passed) + P(N = passed) / 2`` is ``1 - (1 - cl) /
    2`` and ``(1 - cl) / 2``, extended to real counts through the beta
    function and, between 0 and 1 passing entries, interpolated linearly, as
    ``TEfficiency::MidPInterval``. The bounds are exact where ROOT's bisection
    stops at about 1e-9. ``nan`` where the total is not positive or ``passed``
    lies outside ``[0, total]``.
    """
    from scipy.optimize import brentq  # noqa: PLC0415 - imported only when used
    from scipy.special import betainc, betaln, xlog1py, xlogy  # noqa: PLC0415

    tail = check_cl(cl)
    k_all, n_all, valid = _counts(passed, total)

    def bound(k: float, n: float, upper: bool) -> float:
        if 0.0 < k < 1.0:  # between no and one passing entry: ROOT's linear interpolation
            none, one = bound(0.0, n, upper), bound(1.0, n, upper)
            return none + (one - none) * k
        if (upper and k == n) or (not upper and k == 0.0):
            return 1.0 if upper else 0.0

        def mid(p: float) -> float:  # P(N < k) + P(N = k) / 2, falling from 1 (or 1/2) to 0
            log = xlogy(k, p) + xlog1py(n - k, -p) - betaln(k + 1.0, n - k + 1.0)
            exactly = float(np.exp(log)) / (n + 1.0)
            below = float(betainc(n - k + 1.0, k, 1.0 - p)) if k >= 1.0 else 0.0
            return 0.5 * exactly + below - (tail if upper else 1.0 - tail)

        return float(brentq(mid, 0.0, 1.0, xtol=1e-15, rtol=4 * np.finfo(float).eps))

    lower = np.full(k_all.shape, np.nan)
    upper = np.full(k_all.shape, np.nan)
    for index in np.ndindex(valid.shape):
        if valid[index]:
            k, n = float(k_all[index]), float(n_all[index])
            lower[index], upper[index] = bound(k, n, upper=False), bound(k, n, upper=True)
    return lower, upper
