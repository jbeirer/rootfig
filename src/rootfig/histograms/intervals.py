"""Poisson (Garwood) intervals of counts, for the error bars of data."""

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from rootfig._typing import FloatArray

__all__ = [
    "DataErrors",
    "check_z",
    "count_problem",
    "count_scale",
    "poisson_errors",
    "poisson_interval",
]

DataErrors: TypeAlias = Literal["sumw2", "poisson", "auto"]
"""How the error bars of observed data are computed (``plot(data_errors=...)``).

``None`` keeps each histogram's own model: ``sqrt(sum of squared weights)``, ROOT's
``TH1`` default, unless it carries the Poisson interval
(:attr:`~rootfig.histograms.Histogram.poisson`).

* ``"sumw2"`` - ``sqrt(sum of squared weights)`` on both sides, also for a
  histogram that carries the Poisson interval.
* ``"poisson"`` - the Garwood interval of the counts (:func:`poisson_interval`),
  ROOT's ``TH1::kPoisson``; unit-weight counts only (:func:`count_problem`).
* ``"auto"`` - ``"poisson"`` for unit-weight counts, ``sqrt(sum of squared
  weights)`` otherwise: the usual convention for data points (mplhep's).
"""

_WHOLE_ULPS = 4
"""Floating-point steps within which a count is a whole number, at any magnitude."""

_UNIT_WEIGHT = 1e-12
"""Relative tolerance within which a count equals its variance, compared as ``TMath::AreEqualRel``.

The tolerance of ``TEfficiency``'s test for unweighted ``TH1D`` contents, applied per bin.
"""


def _whole(numbers: FloatArray) -> npt.NDArray[np.bool_]:
    """Return where ``numbers`` are whole, to a few floating-point steps (``nan`` is not)."""
    with np.errstate(invalid="ignore"):
        step = np.spacing(np.maximum(np.abs(numbers), 1.0))
        return np.asarray(np.abs(numbers - np.rint(numbers)) <= _WHOLE_ULPS * step, dtype=bool)


def check_z(z: float) -> None:
    """Raise ``ValueError`` unless ``z`` (standard deviations) is positive and finite."""
    if not (math.isfinite(z) and z > 0):
        msg = f"z must be a positive finite number of standard deviations, got {z!r}"
        raise ValueError(msg)


def poisson_interval(counts: npt.ArrayLike, z: float = 1.0) -> tuple[FloatArray, FloatArray]:
    """Garwood's central interval of a Poisson mean for whole ``counts``: ``(lower, upper)``.

    The bounds cover ``z`` standard deviations of a normal distribution (68.27 %
    for ``z = 1``), half the remainder on each side: ``P(N >= n | lower)`` and
    ``P(N <= n | upper)`` are both ``(1 - coverage) / 2``. The lower bound of
    ``n = 0`` is 0, its upper bound ``-log((1 - coverage) / 2)`` (1.84). Both
    are gamma quantiles (SciPy's inverse incomplete gamma functions), as in
    ROOT's ``TH1::kPoisson`` interval of unweighted counts.

    Raises
    ------
    ValueError
        If a count is not a non-negative whole number or ``z`` is not a
        positive finite number.
    """
    check_z(z)
    n = np.asarray(counts, dtype=float)
    whole = np.rint(n)
    with np.errstate(invalid="ignore"):
        bad = ~(np.isfinite(n) & (n >= 0) & _whole(n))
    if bad.any():
        msg = f"counts must be non-negative whole numbers, got {float(n[bad].flat[0])!r}"
        raise ValueError(msg)
    from scipy.special import gammainccinv, gammaincinv  # noqa: PLC0415 - 0.3 s to import

    tail = 0.5 * math.erfc(z / math.sqrt(2.0))
    # P(N >= n | lower) = P(n, lower) and P(N <= n | upper) = Q(n + 1, upper), the regularised
    # lower and upper incomplete gamma functions
    lower = np.where(whole > 0, gammaincinv(np.maximum(whole, 1.0), tail), 0.0)
    upper = gammainccinv(whole + 1.0, tail)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def count_scale(ones: npt.ArrayLike, squares: npt.ArrayLike) -> FloatArray:
    """Return the factor ``c`` of a count in every cell from a record of one count per cell.

    The record starts as one unit count per cell and goes through every
    transformation of the contents, so a cell holds ``m c`` and ``m c**2`` for
    ``m`` merged cells: ``c = squares / ones``. A cell the record never had
    (added by a transformation) takes the factor of the nearest cell it has;
    none left means the contents were scaled by zero.
    """
    ones = np.asarray(ones, dtype=float)
    flat_ones = ones.ravel()
    known = np.flatnonzero(flat_ones != 0)
    if not known.size:
        return np.zeros_like(ones)
    factors = np.asarray(squares, dtype=float).ravel()[known] / flat_ones[known]
    positions = np.arange(flat_ones.size)
    after = np.clip(np.searchsorted(known, positions), 0, known.size - 1)
    before = np.clip(after - 1, 0, known.size - 1)
    nearest = np.where(
        np.abs(positions - known[before]) <= np.abs(known[after] - positions), before, after
    )
    return np.asarray(factors[nearest], dtype=float).reshape(ones.shape)


def poisson_errors(
    values: npt.ArrayLike, scale: npt.ArrayLike, z: float = 1.0
) -> tuple[FloatArray, FloatArray]:
    """Return the Garwood interval of scaled counts as ``(down, up)`` errors of ``values``.

    Each cell holds a count ``n`` times the factor ``scale`` (1 for unit-weight
    counts; normalising them changes the factor, not ``n``) and takes the
    interval of ``n`` times that factor, like its contents.
    """
    values = np.asarray(values, dtype=float)
    factor = np.asarray(scale, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        counts = np.rint(np.where(factor > 0, values / factor, 0.0))
    lower, upper = poisson_interval(counts, z)
    down = np.maximum(values - factor * lower, 0.0)
    up = np.maximum(factor * upper - values, 0.0)
    return np.asarray(down, dtype=float), np.asarray(up, dtype=float)


def count_problem(values: npt.ArrayLike, variances: npt.ArrayLike) -> str | None:
    """Say why ``values`` and ``variances`` are not unit-weight counts, or return ``None``.

    Unit-weight counts are non-negative whole numbers equal to their variances:
    what filling without weights gives, and what a stored ``TH1`` without
    ``Sumw2`` holding whole numbers reports (ROOT itself treats those as
    unweighted counts). Only they are known to be counts: the sums ``(sum w,
    sum w**2)`` of weighted or scaled entries cannot tell counts scaled by one
    factor from unequal weights (``[1, 1, 4]`` sums like two entries of weight
    3), so their Poisson interval is not the counts'. Nor can the contents tell
    counts from weighted entries that are all empty; a
    :class:`~rootfig.histograms.Histogram` filled with weights knows that it was.
    """
    values = np.asarray(values, dtype=float).ravel()
    variances = np.asarray(variances, dtype=float).ravel()
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(variances))):
        return "has non-finite contents"
    if np.any(values < 0):
        return "has negative contents (signed weights)"
    if np.any(np.abs(variances - values) > 0.5 * _UNIT_WEIGHT * (values + np.abs(variances))):
        return "is weighted or scaled (its variances differ from its contents)"
    if not np.all(_whole(values)):
        return "has contents that are not whole numbers"
    return None
