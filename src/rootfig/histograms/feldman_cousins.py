"""The Feldman-Cousins interval of an efficiency, as ROOT's ``TEfficiency::FeldmanCousins``."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from rootfig._typing import FloatArray
from rootfig.histograms.intervals import ONE_SIGMA, check_cl

__all__ = ["feldman_cousins"]

_SIGMAS = 12.0
"""Half width, in standard deviations, of the counts that can matter to an acceptance region."""


def _accepts(k: float, n: int, rho: float, cl: float) -> bool:
    """Return whether ``k`` of ``n`` lies in the Feldman-Cousins acceptance region at ``rho``.

    The region collects counts ``x`` in decreasing order of the likelihood
    ratio ``P(x | rho) / P(x | x / n)`` until they hold ``cl`` of the
    probability; ``k`` lies in it when it lies between its smallest and
    largest count, as ROOT tests it (so a real ``k`` has an interval too).
    Counts beyond ``_SIGMAS`` standard deviations hold none of the probability
    to double precision and are left out.
    """
    from scipy.special import gammaln, xlog1py, xlogy  # noqa: PLC0415 - imported only when used

    spread = _SIGMAS * math.sqrt(n * rho * (1.0 - rho)) + _SIGMAS
    first, last = max(0, math.floor(n * rho - spread)), min(n, math.ceil(n * rho + spread))
    if not first <= k <= last:
        return False
    x = np.arange(first, last + 1, dtype=float)
    log_probability = (
        gammaln(n + 1.0) - gammaln(x + 1.0) - gammaln(n - x + 1.0)
        + xlogy(x, rho) + xlog1py(n - x, -rho)
    )  # fmt: skip
    best = x / n
    log_ratio = xlogy(x, rho) - xlogy(x, best) + xlog1py(n - x, -rho) - xlog1py(n - x, -best)
    order = np.argsort(-log_ratio, kind="stable")
    held = np.cumsum(np.exp(log_probability[order]))
    # add counts while the region holds less than cl, as ROOT does: the one reaching it is in
    size = int(np.searchsorted(held, cl, side="left")) + 1
    region = x[order[:size]]
    return bool(region.min() <= k <= region.max())


def _inside(k: float, n: int, cl: float) -> float | None:
    """Return an efficiency whose region holds ``k``: ``k / n``, else the nearest on a grid."""
    best = k / n
    if _accepts(k, n, best, cl):
        return best
    grid = np.linspace(0.0, 1.0, 1001)
    for rho in grid[np.argsort(np.abs(grid - best))]:
        if _accepts(k, n, float(rho), cl):
            return float(rho)
    return None


def _edge(k: float, n: int, inside: float, outside: float, cl: float) -> float:
    """Bisect between an efficiency whose region holds ``k`` and one whose region does not."""
    for _ in range(200):
        middle = 0.5 * (inside + outside)
        if middle in (inside, outside):
            break
        if _accepts(k, n, middle, cl):
            inside = middle
        else:
            outside = middle
    return inside


def feldman_cousins(
    passed: npt.ArrayLike, total: npt.ArrayLike, cl: float = ONE_SIGMA
) -> tuple[FloatArray, FloatArray]:
    """Return the Feldman-Cousins interval of ``passed`` of ``total`` counts: ``(lower, upper)``.

    The Neyman construction with the likelihood-ratio ordering of Feldman and
    Cousins: the efficiencies whose acceptance region (see :func:`_accepts`)
    holds the observed count. The total is truncated to a whole number, as
    ROOT does. The bounds are found by bisection to double precision, where
    ROOT's stops at about 1e-9 (so it reports 9.3e-10 rather than 0 for no
    passing entry). ``nan`` where the total is not positive or ``passed``
    lies outside ``[0, total]``.
    """
    check_cl(cl)
    k_all, n_all = np.broadcast_arrays(
        np.asarray(passed, dtype=float), np.trunc(np.asarray(total, dtype=float))
    )
    lower = np.full(k_all.shape, np.nan)
    upper = np.full(k_all.shape, np.nan)
    for index in np.ndindex(k_all.shape):
        k, n = k_all[index], n_all[index]
        if not (np.isfinite(k) and np.isfinite(n) and n > 0 and 0 <= k <= n):
            continue
        whole_n = int(n)
        inside = _inside(float(k), whole_n, cl)  # k / n for a whole k: its ratio is 1, the largest
        if inside is None:
            continue
        lower[index] = 0.0 if k == 0 else _edge(float(k), whole_n, inside, 0.0, cl)
        upper[index] = 1.0 if k == whole_n else _edge(float(k), whole_n, inside, 1.0, cl)
    return lower, upper
