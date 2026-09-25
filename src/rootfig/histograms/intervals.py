"""Poisson (Garwood) intervals of counts, for the error bars of data.

No SciPy: the bounds are quantiles of the gamma distribution with an integer
shape, which for counts up to :data:`EXACT_COUNTS` are solved exactly from
Poisson sums, and beyond from the Wilson-Hilferty approximation.
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
    "count_scale",
    "is_unit_counts",
    "poisson_errors",
    "poisson_interval",
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

Above, the Wilson-Hilferty approximation of the gamma quantile misses the exact
bound by ``1.8e-5`` of the error bar (``6e-4`` counts at 1001), and the
relative error falls as ``1 / n``.
"""

_WHOLE = 1e-9
"""Relative tolerance within which an effective count is a whole number."""


def poisson_interval(counts: npt.ArrayLike, z: float = 1.0) -> tuple[FloatArray, FloatArray]:
    """Garwood's central interval of a Poisson mean for whole ``counts``: ``(lower, upper)``.

    The bounds cover ``z`` standard deviations of a normal distribution (68.27 %
    for ``z = 1``), half the remainder on each side: ``P(N >= n | lower)`` and
    ``P(N <= n | upper)`` are both ``(1 - coverage) / 2``. The lower bound of
    ``n = 0`` is 0, its upper bound ``-log((1 - coverage) / 2)`` (1.84). This
    is ROOT's ``TH1::kPoisson`` interval of unweighted counts.

    Raises
    ------
    ValueError
        If a count is not a non-negative whole number or ``z`` is not a
        positive finite number.
    """
    if not (math.isfinite(z) and z > 0):
        msg = f"z must be a positive finite number of standard deviations, got {z!r}"
        raise ValueError(msg)
    n = np.asarray(counts, dtype=float)
    whole = np.rint(n)
    with np.errstate(invalid="ignore"):
        bad = ~(np.isfinite(n) & (n >= 0) & (np.abs(n - whole) <= _WHOLE * np.maximum(n, 1.0)))
    if bad.any():
        msg = f"counts must be non-negative whole numbers, got {float(n[bad].flat[0])!r}"
        raise ValueError(msg)
    n = whole
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


def count_scale(values: npt.ArrayLike, variances: npt.ArrayLike) -> FloatArray:
    """Return the factor ``c`` scaling the count of every cell: ``values = c * n``.

    A filled cell has ``c = variances / values`` (``variances = c**2 * n``); an
    empty one takes the factor of the nearest filled cell, or 1 when none is
    filled. The contents must be scaled counts (see :func:`count_problem`).
    """
    values = np.asarray(values, dtype=float)
    flat_values = values.ravel()
    flat_variances = np.asarray(variances, dtype=float).ravel()
    filled = np.flatnonzero(flat_values > 0)
    if not filled.size:
        return np.ones_like(values)
    positions = np.arange(flat_values.size)
    after = np.clip(np.searchsorted(filled, positions), 0, filled.size - 1)
    before = np.clip(after - 1, 0, filled.size - 1)
    left, right = filled[before], filled[after]
    nearest = np.where(np.abs(positions - left) <= np.abs(right - positions), left, right)
    return np.asarray(flat_variances[nearest] / flat_values[nearest], dtype=float).reshape(
        values.shape
    )


def poisson_errors(
    values: npt.ArrayLike,
    variances: npt.ArrayLike,
    z: float = 1.0,
    *,
    scale: npt.ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return the Garwood interval of scaled counts as ``(down, up)`` errors of ``values``.

    Each cell holds a count ``n`` scaled by a factor ``c`` (unit-weight counts:
    ``c = 1``; normalising them changes ``c`` but not ``n``), and takes the
    interval of ``n`` times ``c``, like its contents. A filled cell knows its
    factor, ``variances / values``; ``scale`` gives that of every cell, which
    an empty cell needs, else :func:`count_scale` infers it.
    """
    values = np.asarray(values, dtype=float)
    variances = np.asarray(variances, dtype=float)
    filled = values > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        own = variances / values
    guess = count_scale(values, variances) if scale is None else np.asarray(scale, dtype=float)
    factor = np.where(filled, own, guess)
    lower, upper = poisson_interval(np.rint(np.where(filled, values / factor, 0.0)), z)
    down = np.maximum(values - factor * lower, 0.0)
    up = np.maximum(factor * upper - values, 0.0)
    return np.asarray(down, dtype=float), np.asarray(up, dtype=float)


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
