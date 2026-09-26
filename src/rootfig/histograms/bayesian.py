"""Bayesian intervals of efficiencies: the posterior of a Beta prior, as ROOT's ``TEfficiency``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rootfig._typing import FloatArray
from rootfig.histograms.intervals import ONE_SIGMA, check_cl

__all__ = ["Bayesian", "bayesian_interval"]


@dataclass(frozen=True)
class Bayesian:
    """The Bayesian interval of an efficiency with a ``Beta(alpha, beta)`` prior.

    ``passed`` of ``total`` entries make the posterior ``Beta(passed + alpha,
    total - passed + beta)``; with weights, the sums are scaled to the effective
    entries of the total first (``sum w / sum w^2``), as ROOT's ``TEfficiency``
    does. The efficiency is the posterior mean, or its mode with ``mode=True``,
    and the interval the central one, or the shortest with ``shortest=True``
    (ROOT's ``kPosteriorMode`` and ``kShortestInterval``). ``shortest`` follows
    ``mode`` unless given, as ``TEfficiency::SetPosteriorMode`` also sets the
    shortest interval. ``interval="jeffreys"`` is ``Bayesian(0.5, 0.5)``,
    ``"uniform"`` is ``Bayesian(1, 1)``.

    Raises
    ------
    ValueError
        Unless ``alpha`` and ``beta`` are positive finite numbers.
    """

    alpha: float = 1.0
    beta: float = 1.0
    mode: bool = False
    shortest: bool | None = None

    def __post_init__(self) -> None:
        for name in ("alpha", "beta"):
            value = getattr(self, name)
            number = isinstance(value, int | float) and not isinstance(value, bool)
            if not (number and 0 < value < np.inf):
                msg = f"Bayesian {name} must be a positive finite number, got {value!r}"
                raise ValueError(msg)
        if self.shortest is None:
            object.__setattr__(self, "shortest", self.mode)


def _posterior(
    prior: Bayesian,
    passed: FloatArray,
    total: FloatArray,
    total_variance: FloatArray | None,
) -> tuple[FloatArray, FloatArray]:
    """Return the posterior's Beta parameters; weighted sums are scaled to effective entries."""
    norm: FloatArray | float = 1.0
    if total_variance is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(total_variance > 0, total / total_variance, np.nan)
    return passed * norm + prior.alpha, (total - passed) * norm + prior.beta


def _mode(a: FloatArray, b: FloatArray) -> FloatArray:
    """Return the mode of ``Beta(a, b)``, at an end as ``TEfficiency::BetaMode`` puts it."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = (a - 1.0) / (a + b - 2.0)
    edge = np.where(a < b, 0.0, np.where(a > b, 1.0, 0.5))
    return np.asarray(np.where((a <= 1.0) | (b <= 1.0), edge, inner), dtype=float)


def _shortest(a: float, b: float, cl: float) -> tuple[float, float]:
    """Return the shortest interval holding ``cl`` of ``Beta(a, b)``, with ROOT's end cases."""
    from scipy.optimize import brentq  # noqa: PLC0415 - imported only when used
    from scipy.special import betainc, betaincinv, xlog1py, xlogy  # noqa: PLC0415

    mode = float(_mode(np.array(a), np.array(b)))
    if mode == 0.0:
        return 0.0, float(betaincinv(a, b, cl))
    if mode == 1.0:
        return float(betaincinv(a, b, 1.0 - cl)), 1.0
    if a == b and a <= 1.0:  # no shortest interval: the central one, as ROOT
        tail = check_cl(cl)
        return float(betaincinv(a, b, tail)), float(betaincinv(a, b, 1.0 - tail))

    def log_density(x: float) -> float:
        return float(xlogy(a - 1.0, x) + xlog1py(b - 1.0, -x))

    def upper(low: float) -> float:
        return float(betaincinv(a, b, min(betainc(a, b, low) + cl, 1.0)))

    # a and b above 1: the density rises to the mode and falls, and the shortest interval
    # holding cl ends where it is equally high on both sides
    top = float(betaincinv(a, b, 1.0 - cl))
    low = brentq(
        lambda x: log_density(x) - log_density(upper(x)),
        0.0,
        top,
        xtol=1e-15,
        rtol=4 * np.finfo(float).eps,
    )
    return float(low), upper(float(low))


def bayesian_interval(
    prior: Bayesian,
    passed: npt.ArrayLike,
    total: npt.ArrayLike,
    total_variance: npt.ArrayLike | None = None,
    cl: float = ONE_SIGMA,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the posterior efficiency and its interval at ``cl``: ``(value, lower, upper)``.

    ``total_variance`` (the sum of squared weights of the total) scales weighted
    sums to effective entries; ``None`` takes the counts as they are. With no
    entries the posterior is the prior. ``nan`` where the posterior is not a
    Beta distribution (negative weights) or a weighted total has no variance.
    """
    from scipy.special import betaincinv  # noqa: PLC0415 - 0.3 s to import, only when used

    tail = check_cl(cl)
    k = np.asarray(passed, dtype=float)
    n = np.asarray(total, dtype=float)
    variance = None if total_variance is None else np.asarray(total_variance, dtype=float)
    k, n = np.broadcast_arrays(k, n)
    a, b = _posterior(prior, k, n, variance)
    valid = (a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b)
    a, b = np.where(valid, a, 1.0), np.where(valid, b, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = _mode(a, b) if prior.mode else a / (a + b)
    if prior.shortest:
        pairs = zip(a.ravel(), b.ravel(), strict=True)
        bounds = [_shortest(float(x), float(y), cl) for x, y in pairs]
        lower = np.array([low for low, _ in bounds], dtype=float).reshape(a.shape)
        upper = np.array([high for _, high in bounds], dtype=float).reshape(a.shape)
    else:
        lower, upper = betaincinv(a, b, tail), betaincinv(a, b, 1.0 - tail)
    nan = np.nan
    return (
        np.asarray(np.where(valid, value, nan), dtype=float),
        np.asarray(np.where(valid, lower, nan), dtype=float),
        np.asarray(np.where(valid, upper, nan), dtype=float),
    )
