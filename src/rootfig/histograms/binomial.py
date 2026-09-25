"""Confidence intervals of efficiencies: Clopper-Pearson, the normal approximation and Wilson.

``"auto"`` is what ROOT's ``TEfficiency`` (and ``TGraphAsymmErrors::Divide``)
gives by default: Clopper-Pearson for unweighted counts, the normal
approximation for weighted entries. No SciPy: the Clopper-Pearson bounds are
beta quantiles, solved from the continued fraction of the incomplete beta
function.
"""

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from rootfig._typing import FloatArray

__all__ = [
    "EfficiencyInterval",
    "check_interval",
    "clopper_pearson",
    "efficiency_bounds",
    "is_unweighted",
    "normal_interval",
    "resolve_interval",
    "wilson_interval",
]

EfficiencyInterval: TypeAlias = Literal["auto", "clopper-pearson", "normal", "wilson"]
"""How the confidence interval of an efficiency is computed.

* ``"clopper-pearson"`` - the exact binomial interval of counts, never covering
  less than the requested probability; unweighted entries only.
* ``"normal"`` - the efficiency plus or minus ``z`` standard deviations of the
  weighted pass fraction, clipped to ``[0, 1]``: no width at 0 and 1.
* ``"wilson"`` - the Wilson score interval, with the effective entries
  ``(sum w)^2 / sum w^2`` for weighted entries.
* ``"auto"`` - ROOT's default: ``"clopper-pearson"`` for unweighted entries,
  ``"normal"`` otherwise.
"""

_UNWEIGHTED_TOLERANCE = 1e-5
"""Relative tolerance of ROOT's test for unweighted entries, ``sum w == sum w^2``."""


def is_unweighted(sum_weights: float, sum_squares: float) -> bool:
    """Return True if entries summing to ``sum_weights`` count as unweighted, as ROOT decides.

    ``TEfficiency`` treats a histogram as unweighted when its sum of weights
    equals its sum of squared weights (relative ``1e-5``), as it does when every
    entry has weight 1.
    """
    difference = abs(sum_weights - sum_squares)
    return difference <= 0.5 * _UNWEIGHTED_TOLERANCE * (abs(sum_weights) + abs(sum_squares))


def check_interval(interval: str) -> None:
    """Raise ``ValueError`` unless ``interval`` names an :data:`EfficiencyInterval`."""
    if interval not in ("auto", "clopper-pearson", "normal", "wilson"):
        msg = f"interval must be 'auto', 'clopper-pearson', 'normal' or 'wilson', got {interval!r}"
        raise ValueError(msg)


def resolve_interval(interval: str, unweighted: bool, context: str = "") -> EfficiencyInterval:
    """Return the method ``interval`` stands for: ``"auto"`` resolved like ``TEfficiency``.

    Raises
    ------
    ValueError
        For an unknown ``interval``, or ``"clopper-pearson"`` for weighted entries.
    """
    check_interval(interval)
    if interval == "auto":
        return "clopper-pearson" if unweighted else "normal"
    if interval == "clopper-pearson" and not unweighted:
        msg = (
            f"{context + ': ' if context else ''}interval='clopper-pearson' needs unweighted "
            "entries; use 'normal' (ROOT's choice for weighted entries, also what 'auto' "
            "picks) or 'wilson'"
        )
        raise ValueError(msg)
    return interval  # type: ignore[return-value]


def efficiency_bounds(
    interval: EfficiencyInterval,
    passed: npt.ArrayLike,
    total: npt.ArrayLike,
    passed_variance: npt.ArrayLike,
    total_variance: npt.ArrayLike,
    *,
    z: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Return the bounds of the resolved ``interval`` (not ``"auto"``): ``(lower, upper)``."""
    if interval == "clopper-pearson":
        return clopper_pearson(passed, total, z)
    if interval == "normal":
        return normal_interval(passed, total, passed_variance, total_variance, z)
    return wilson_interval(passed, total, total_variance, z)


def _tail(z: float) -> float:
    """Probability beyond ``z`` standard deviations on one side."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def clopper_pearson(
    passed: npt.ArrayLike, total: npt.ArrayLike, z: float = 1.0
) -> tuple[FloatArray, FloatArray]:
    """Return the Clopper-Pearson interval of ``passed`` of ``total`` counts: ``(lower, upper)``.

    The bounds are the efficiencies at which ``passed`` or more, and ``passed``
    or fewer, entries pass with the probability beyond ``z`` standard deviations
    on one side: beta quantiles, as ``TEfficiency::ClopperPearson`` computes
    them. The lower bound is 0 for no passing entry, the upper 1 when all pass.
    ``nan`` where the total is not positive or ``passed`` lies outside
    ``[0, total]``.
    """
    k = np.asarray(passed, dtype=float)
    n = np.asarray(total, dtype=float)
    k, n = np.broadcast_arrays(k, n)
    valid = (n > 0) & (k >= 0) & (k <= n)
    tail = _tail(z)
    lower = np.where(valid, 0.0, np.nan)
    upper = np.where(valid, 1.0, np.nan)
    some = valid & (k > 0)
    lower[some] = _beta_quantile(k[some], n[some] - k[some] + 1.0, tail, -z)
    short = valid & (k < n)
    upper[short] = _beta_quantile(k[short] + 1.0, n[short] - k[short], 1.0 - tail, z)
    return lower, upper


def normal_interval(
    passed: npt.ArrayLike,
    total: npt.ArrayLike,
    passed_variance: npt.ArrayLike,
    total_variance: npt.ArrayLike,
    z: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Return the normal approximation of the efficiency ``passed / total``: ``(lower, upper)``.

    The efficiency ``p`` plus or minus ``z`` times the standard deviation of a
    weighted pass fraction, ``sqrt(passed_variance (1 - 2 p) + total_variance
    p^2) / total`` (``sqrt(p (1 - p) / total)`` for counts), clipped to
    ``[0, 1]``, as ROOT's ``TEfficiency`` computes it for weighted entries. The
    variance is that of the ratio of the two sums to first order, so it also
    holds for signed weights. ``nan`` where the total is not positive or the
    efficiency lies outside ``[0, 1]``.
    """
    k = np.asarray(passed, dtype=float)
    n = np.asarray(total, dtype=float)
    vk = np.asarray(passed_variance, dtype=float)
    vn = np.asarray(total_variance, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n != 0, k / n, np.nan)
        sigma = np.sqrt(np.maximum(vk * (1.0 - 2.0 * p) + vn * p * p, 0.0)) / n
        valid = (n > 0) & (p >= 0.0) & (p <= 1.0)
        lower = np.where(valid, np.maximum(p - z * sigma, 0.0), np.nan)
        upper = np.where(valid, np.minimum(p + z * sigma, 1.0), np.nan)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def wilson_interval(
    passed: npt.ArrayLike, total: npt.ArrayLike, total_variance: npt.ArrayLike, z: float = 1.0
) -> tuple[FloatArray, FloatArray]:
    """Return the Wilson score interval of the efficiency ``passed / total``: ``(lower, upper)``.

    ``passed`` sums the weights of a subset of the entries ``total`` sums, so the
    interval is binomial. For counts it is ROOT's ``TEfficiency::Wilson``. With
    weights, the effective entries of the total, ``total**2 / total_variance``,
    take the place of its count: the weighted pass fraction of entries passing
    with probability ``e`` has the variance ``e (1 - e) / n_eff``, and the score
    interval inverts it. Unlike the normal approximation it keeps a width at 0
    and 1. The interval covers ``z`` standard deviations, is clipped to
    ``[0, 1]`` and always contains the efficiency. It is ``nan`` where the
    total is not positive or the efficiency lies outside ``[0, 1]``.
    """
    k = np.asarray(passed, dtype=float)
    n = np.asarray(total, dtype=float)
    vn = np.asarray(total_variance, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n != 0, k / n, np.nan)
        n_eff = np.where(vn > 0, n**2 / vn, n)
        z2 = z * z
        denominator = 1.0 + z2 / n_eff
        centre = (p + z2 / (2.0 * n_eff)) / denominator
        half = (z / denominator) * np.sqrt(p * (1.0 - p) / n_eff + z2 / (4.0 * n_eff**2))
        valid = (n > 0) & (p >= 0.0) & (p <= 1.0)
        # the Wilson interval contains p by construction; guard against round-off at 0 and 1
        lower = np.where(valid, np.minimum(np.clip(centre - half, 0.0, 1.0), p), np.nan)
        upper = np.where(valid, np.maximum(np.clip(centre + half, 0.0, 1.0), p), np.nan)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


# -- beta quantiles ----------------------------------------------------------------------

_TINY = 1e-300


def _log_beta(a: FloatArray, b: FloatArray) -> FloatArray:
    """``log B(a, b)`` per element."""
    lgamma = np.frompyfunc(math.lgamma, 1, 1)
    return np.asarray(lgamma(a) + lgamma(b) - lgamma(a + b), dtype=float)


def _continued_fraction(a: FloatArray, b: FloatArray, x: FloatArray) -> FloatArray:
    """Continued fraction of the incomplete beta function (modified Lentz).

    It converges quickly for ``x < (a + 1) / (a + b + 2)``; the caller uses the
    symmetry ``I_x(a, b) = 1 - I_{1-x}(b, a)`` beyond. Each element stops once
    a step changes it by less than ``1e-15``: after that its steps only
    scatter around 1 by round-off, and waiting for all of them to fall below
    at once can take thousands of iterations.
    """

    def clamp(value: FloatArray) -> FloatArray:
        return np.where(np.abs(value) < _TINY, _TINY, value)

    c = np.ones_like(x)
    d = 1.0 / clamp(1.0 - (a + b) * x / (a + 1.0))
    result = d
    active = np.ones(x.shape, dtype=bool)
    for m in range(1, 100_000):
        numerator = m * (b - m) * x / ((a + 2 * m - 1.0) * (a + 2 * m))
        d = 1.0 / clamp(1.0 + numerator * d)
        c = clamp(1.0 + numerator / c)
        step = d * c
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1.0))
        d = 1.0 / clamp(1.0 + numerator * d)
        c = clamp(1.0 + numerator / c)
        last = d * c
        result = np.where(active, result * step * last, result)
        active &= np.abs(last - 1.0) >= 1e-15
        if not active.any():
            break
    return result


def _beta_cdf(a: FloatArray, b: FloatArray, x: FloatArray, log_beta: FloatArray) -> FloatArray:
    """Regularised incomplete beta function ``I_x(a, b)`` for ``0 < x < 1``."""
    swap = x > (a + 1.0) / (a + b + 2.0)
    a_, b_ = np.where(swap, b, a), np.where(swap, a, b)
    x_ = np.where(swap, 1.0 - x, x)
    front = np.exp(a_ * np.log(x_) + b_ * np.log1p(-x_) - log_beta) / a_
    part = front * _continued_fraction(a_, b_, x_)
    return np.asarray(np.where(swap, 1.0 - part, part), dtype=float)


def _beta_start(
    a: FloatArray, b: FloatArray, q: float, deviations: float, log_beta: FloatArray
) -> FloatArray:
    """First guess of the ``q`` quantile of the beta distribution (Numerical Recipes).

    For ``a, b >= 1`` a normal approximation corrected for skewness
    (Abramowitz and Stegun 26.5.22), else the power law of the nearer tail.
    """
    upper_tail = -deviations  # the deviate beyond which the normal tail holds q
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        shape = (upper_tail**2 - 3.0) / 6.0
        h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0))
        asymmetry = 1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)
        w = upper_tail * np.sqrt(shape + h) / h - asymmetry * (shape + 5.0 / 6.0 - 2.0 / (3.0 * h))
        normal = a / (a + b * np.exp(2.0 * w))
        # the power laws x^a / (a B) and (1 - x)^b / (b B) of the two tails
        t = np.exp(a * np.log(a / (a + b)) - log_beta) / a
        u = np.exp(b * np.log(b / (a + b)) - log_beta) / b
        tails = np.where(
            q < t / (t + u),
            (a * (t + u) * q) ** (1.0 / a),
            1.0 - (b * (t + u) * (1.0 - q)) ** (1.0 / b),
        )
    x = np.where((a >= 1.0) & (b >= 1.0), normal, tails)
    return np.clip(np.nan_to_num(x, nan=0.5), 1e-300, 1.0 - 1e-16)


def _beta_quantile(a: FloatArray, b: FloatArray, q: float, deviations: float) -> FloatArray:
    """Return ``x`` with ``I_x(a, b) = q``, per element, by Halley's method within a bracket.

    The start is that of Numerical Recipes (``invbetai``), from the normal
    quantile ``deviations`` of ``q``; a step that leaves the bracket around the
    root is replaced by bisection. An element stops once its step falls below
    ``1e-12`` of ``x`` (or ``1 - x``): for large ``a + b`` the round-off of
    ``I_x`` leaves about ``1e-13`` of it undetermined.
    """
    if a.size == 0:
        return np.asarray(a, dtype=float)
    pairs, inverse = np.unique(np.stack([a, b], axis=1), axis=0, return_inverse=True)
    a, b = pairs[:, 0], pairs[:, 1]
    log_beta = _log_beta(a, b)
    x = _beta_start(a, b, q, deviations, log_beta)
    low, high = np.zeros_like(x), np.ones_like(x)
    active = np.ones(x.shape, dtype=bool)
    for _ in range(200):
        xa, aa, ba, lb = x[active], a[active], b[active], log_beta[active]
        excess = _beta_cdf(aa, ba, xa, lb) - q
        low[active] = np.where(excess < 0, xa, low[active])
        high[active] = np.where(excess > 0, xa, high[active])
        density = np.exp((aa - 1.0) * np.log(xa) + (ba - 1.0) * np.log1p(-xa) - lb)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = excess / density
            curvature = (aa - 1.0) / xa - (ba - 1.0) / (1.0 - xa)
            halley = xa - newton / (1.0 - 0.5 * np.minimum(1.0, newton * curvature))
        # converged: a step below 1e-12 of the nearer bound, or of the spacing of floats near 1
        still = np.abs(halley - xa) > np.maximum(1e-12 * np.minimum(xa, 1.0 - xa), 4e-16)
        inside = (halley > low[active]) & (halley < high[active])
        new = np.where(inside | ~still, halley, 0.5 * (low[active] + high[active]))
        x[active] = new
        active[active] = still & (excess != 0)
        if not active.any():
            break
    return np.asarray(x[inverse.ravel()], dtype=float)
