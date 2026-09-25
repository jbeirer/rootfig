"""Confidence intervals of counts: Poisson (Garwood) for data, Wilson for efficiencies.

No SciPy: the Poisson bounds are quantiles of the gamma distribution with an
integer shape, which for counts up to :data:`EXACT_COUNTS` are solved exactly
from Poisson sums, and beyond from the Wilson-Hilferty approximation.
"""

from __future__ import annotations

import functools
import math
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from rootfig._typing import FloatArray

__all__ = [
    "DataErrors",
    "count_problem",
    "is_unit_counts",
    "poisson_errors",
    "poisson_interval",
    "wilson_interval",
]

DataErrors: TypeAlias = Literal["auto", "poisson", "sumw2"]
"""How the error bars of observed data are computed (``plot(data_errors=...)``).

* ``"poisson"`` - the Garwood interval of the counts (:func:`poisson_interval`).
* ``"sumw2"`` - ``sqrt(sum of squared weights)`` on both sides.
* ``"auto"`` - ``"poisson"`` for unit-weight counts (:func:`is_unit_counts`),
  ``"sumw2"`` otherwise.
"""

EXACT_COUNTS = 1000
"""Counts up to which the Garwood bounds are solved to machine precision.

Above, the Wilson-Hilferty approximation of the gamma quantile is within
``2e-5`` of the error bar at 1000 counts, and its error falls as ``1 / n``.
"""

_WHOLE = 1e-9
"""Relative tolerance within which an effective count is a whole number."""


def poisson_interval(counts: npt.ArrayLike, z: float = 1.0) -> tuple[FloatArray, FloatArray]:
    """Garwood's central interval of a Poisson mean for whole ``counts``: ``(lower, upper)``.

    The bounds cover ``z`` standard deviations of a normal distribution (68.27 %
    for ``z = 1``), half the remainder on each side: ``P(N >= n | lower)`` and
    ``P(N <= n | upper)`` are both ``(1 - coverage) / 2``. The lower bound of
    ``n = 0`` is 0, its upper bound ``-log((1 - coverage) / 2)`` (1.84).
    """
    n = np.asarray(counts, dtype=float)
    tail = 0.5 * math.erfc(z / math.sqrt(2.0))
    lower = np.where(n > 0, _gamma_quantile(np.maximum(n, 1.0), -z), 0.0)
    upper = _gamma_quantile(n + 1.0, z)
    exact = n <= EXACT_COUNTS
    if exact.any():
        wanted = n[exact]
        filled = wanted > 0
        low = lower[exact]
        low[filled] = _solve(wanted[filled], 1.0 - tail, low[filled])
        lower[exact] = low
        upper[exact] = _solve(wanted + 1.0, tail, upper[exact])
    return lower, upper


def _gamma_quantile(shape: FloatArray, z: float) -> FloatArray:
    """Return the Wilson-Hilferty quantile of a gamma distribution ``z`` deviations out."""
    return np.asarray(shape * (1.0 - 1.0 / (9.0 * shape) + z / (3.0 * np.sqrt(shape))) ** 3)


def _solve(shape: FloatArray, target: float, start: FloatArray) -> FloatArray:
    """Solve ``P(N < shape | x) = target`` for ``x``, per element, by Newton in ``log x``.

    ``P(N < shape | x)`` is the sum of the Poisson probabilities below the
    (whole) ``shape``; it falls with ``x`` at the rate of the probability of
    ``shape - 1``. Steps are capped at a factor ``e``, which keeps a poor start
    from overshooting.
    """
    if shape.size == 0:
        return np.asarray(start, dtype=float)
    # equal shapes have equal solutions: solve each once
    a, first, inverse = np.unique(
        np.rint(shape).astype(int), return_index=True, return_inverse=True
    )
    k = np.arange(int(a.max()))
    below = k < a[:, np.newaxis]
    log_factorial = _log_factorials(int(a.max()))
    log_x = np.log(np.maximum(np.asarray(start, dtype=float)[first], 1e-3))
    for _ in range(100):
        x = np.exp(log_x)
        log_terms = k * log_x[:, np.newaxis] - x[:, np.newaxis] - log_factorial[k]
        cumulative = np.sum(np.exp(log_terms), axis=1, where=below)
        density = np.exp((a - 1) * log_x - x - log_factorial[a - 1]) * x
        step = np.clip((cumulative - target) / density, -1.0, 1.0)
        log_x = log_x + step
        if np.all(np.abs(step) < 1e-13):
            break
    return np.asarray(np.exp(log_x)[inverse])


@functools.cache
def _log_factorials(size: int) -> FloatArray:
    """``log(k!)`` for ``k = 0, ..., size``."""
    return np.array([math.lgamma(k + 1.0) for k in range(size + 1)])


def poisson_errors(
    values: npt.ArrayLike, variances: npt.ArrayLike, z: float = 1.0
) -> tuple[FloatArray, FloatArray]:
    """Return the Garwood interval of scaled counts as ``(down, up)`` errors of ``values``.

    Each cell holds a count ``n`` scaled by a factor ``c``: ``values = c * n``
    and ``variances = c**2 * n``, so ``n = values**2 / variances`` and
    ``c = variances / values``. Unit-weight counts have ``c = 1``; normalising
    them, or dividing by bin widths, changes ``c`` but not ``n``. An empty cell
    takes the factor of the nearest filled one, the convention of mplhep and
    coffea (exact when one factor scales every cell), or 1 when none is filled.
    The contents must be scaled counts (see :func:`count_problem`).
    """
    values = np.asarray(values, dtype=float)
    variances = np.asarray(variances, dtype=float)
    flat_values, flat_variances = values.ravel(), variances.ravel()
    filled = np.flatnonzero(flat_values > 0)
    scale = np.ones_like(flat_values)
    if filled.size:
        positions = np.arange(flat_values.size)
        after = np.clip(np.searchsorted(filled, positions), 0, filled.size - 1)
        before = np.clip(after - 1, 0, filled.size - 1)
        left, right = filled[before], filled[after]
        nearest = np.where(np.abs(positions - left) <= np.abs(right - positions), left, right)
        scale = flat_variances[nearest] / flat_values[nearest]
    lower, upper = poisson_interval(np.rint(flat_values / scale), z)
    down = np.maximum(flat_values - scale * lower, 0.0).reshape(values.shape)
    up = np.maximum(scale * upper - flat_values, 0.0).reshape(values.shape)
    return down, up


def is_unit_counts(values: npt.ArrayLike, variances: npt.ArrayLike) -> bool:
    """Return True if every cell holds a whole number of unit-weight entries.

    That is, a non-negative whole number equal to its variance: what filling
    without weights gives, and what a stored ``TH1`` without ``Sumw2`` holding
    whole numbers reports (ROOT itself treats those as unweighted counts).
    """
    values = np.asarray(values, dtype=float)
    variances = np.asarray(variances, dtype=float)
    tolerance = _WHOLE * np.maximum(values, 1.0)
    return bool(
        np.all(np.isfinite(values))
        and np.all(values >= 0)
        and np.all(np.abs(values - np.rint(values)) <= tolerance)
        and np.all(np.abs(variances - values) <= tolerance)
    )


def count_problem(values: npt.ArrayLike, variances: npt.ArrayLike) -> str | None:
    """Say why ``values`` and ``variances`` are not scaled counts, or return ``None``.

    Scaled counts (see :func:`poisson_errors`) are non-negative, and a filled
    cell has a positive variance whose effective count ``values**2 /
    variances`` is a whole number; an empty cell has no variance.
    """
    values = np.asarray(values, dtype=float).ravel()
    variances = np.asarray(variances, dtype=float).ravel()
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(variances))):
        return "has non-finite contents"
    if np.any(values < 0):
        return "has negative contents (signed weights)"
    filled = values > 0
    if np.any(variances[~filled] != 0):
        return "has empty bins with a variance (positive and negative weights that cancel)"
    if np.any(variances[filled] <= 0):
        return "has filled bins without a variance"
    counts = values[filled] ** 2 / variances[filled]
    if np.any(np.abs(counts - np.rint(counts)) > _WHOLE * np.maximum(counts, 1.0)):
        return "is weighted (its effective counts are not whole numbers)"
    return None


def wilson_interval(
    passed: npt.ArrayLike, total: npt.ArrayLike, total_variance: npt.ArrayLike, z: float = 1.0
) -> tuple[FloatArray, FloatArray]:
    """Return the Wilson score interval of the efficiency ``passed / total``: ``(lower, upper)``.

    ``passed`` sums the weights of a subset of the entries ``total`` sums, so the
    interval is binomial, never that of two independent yields. With weights,
    the effective entries of the total, ``total**2 / total_variance``, take the
    place of its count: exact for one weight per entry, an approximation for
    others. The interval covers ``z`` standard deviations, is clipped to
    ``[0, 1]`` and always contains the efficiency. It is ``nan`` where the
    total is not positive or the efficiency lies outside ``[0, 1]``, which no
    binomial interval describes.
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
