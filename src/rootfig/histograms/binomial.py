"""Confidence intervals of efficiencies: Clopper-Pearson, the normal approximation and Wilson.

``"auto"`` is what ROOT's ``TEfficiency`` (and ``TGraphAsymmErrors::Divide``)
gives by default: Clopper-Pearson for unweighted counts, the normal
approximation for weighted entries.
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
    """Return the bounds of the resolved ``interval``: ``(lower, upper)``.

    Raises
    ------
    ValueError
        For ``"auto"``, which :func:`resolve_interval` turns into a method first.
    """
    if interval == "clopper-pearson":
        return clopper_pearson(passed, total, z)
    if interval == "normal":
        return normal_interval(passed, total, passed_variance, total_variance, z)
    if interval == "wilson":
        return wilson_interval(passed, total, total_variance, z)
    msg = f"efficiency_bounds needs a resolved interval, got {interval!r}"
    raise ValueError(msg)


def _tail(z: float) -> float:
    """Probability beyond ``z`` standard deviations on one side."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def clopper_pearson(
    passed: npt.ArrayLike, total: npt.ArrayLike, z: float = 1.0
) -> tuple[FloatArray, FloatArray]:
    """Return the Clopper-Pearson interval of ``passed`` of ``total`` counts: ``(lower, upper)``.

    The bounds are the efficiencies at which ``passed`` or more, and ``passed``
    or fewer, entries pass with the probability beyond ``z`` standard deviations
    on one side: beta quantiles (SciPy's inverse incomplete beta function), as
    ``TEfficiency::ClopperPearson`` computes them. The lower bound is 0 for no
    passing entry, the upper 1 when all pass. ``nan`` where the total is not
    positive or ``passed`` lies outside ``[0, total]``.
    """
    from scipy.special import betaincinv  # noqa: PLC0415 - 0.3 s to import, only when used

    k, n = np.broadcast_arrays(np.asarray(passed, dtype=float), np.asarray(total, dtype=float))
    valid = (n > 0) & (k >= 0) & (k <= n)
    tail = _tail(z)
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
