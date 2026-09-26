"""Uncertainties of a normalised shape, whose own total fluctuates with its bins.

A histogram ``x`` normalised to its own visible total ``S`` becomes ``y = g x / S``
(``g`` the target, or one over the bin size for a density). To first order its
covariance is ``J V J^T`` with the Jacobian ``J_ij = g_i (delta_ij / S - x_i / S^2)``
(the second term for visible bins ``j`` only) and ``V`` the diagonal of the
variances: every bin is anti-correlated with the others, and a bin holding the
fraction ``p`` of ``N`` counts has the variance ``p (1 - p) / N`` of a binomial
fraction. For counts rootfig takes the exact binomial interval instead.
"""

from __future__ import annotations

import numpy as np

from rootfig._typing import FloatArray
from rootfig.histograms.binomial import clopper_pearson

__all__ = ["shape_bounds", "shape_covariance_matrix", "shape_variances"]


def shape_variances(
    values: FloatArray, variances: FloatArray, gain: FloatArray, visible: np.ndarray
) -> FloatArray:
    """Return the variances of every cell of ``gain * values / S`` to first order.

    ``S`` sums the ``visible`` cells; a flow cell is divided by it without
    entering it. All arrays have the shape of the cells.
    """
    total = float(values[visible].sum())
    fraction = values / total
    spread = float(variances[visible].sum())
    inner = variances - 2.0 * fraction * variances * visible + fraction**2 * spread
    return np.asarray((gain / total) ** 2 * np.maximum(inner, 0.0), dtype=float)


def shape_covariance_matrix(
    values: FloatArray, variances: FloatArray, gain: FloatArray | float
) -> FloatArray:
    """Return the covariance of ``gain * values / values.sum()`` (visible bins, flattened)."""
    x = np.ravel(values)
    v = np.ravel(variances)
    g = np.broadcast_to(np.ravel(gain) if np.ndim(gain) else gain, x.shape)
    total = float(x.sum())
    fraction = x / total
    spread = float(v.sum())
    inner = (
        np.diag(v)
        - np.outer(fraction, v)
        - np.outer(v, fraction)
        + spread * np.outer(fraction, fraction)
    )
    return np.asarray(np.outer(g, g) * inner / total**2, dtype=float)


def shape_bounds(
    counts: FloatArray, gain: FloatArray, visible: np.ndarray, cl: float
) -> tuple[FloatArray, FloatArray] | None:
    """Return the Clopper-Pearson ``(down, up)`` errors of counts normalised to their total.

    A visible bin of ``n`` of the ``N`` visible counts is the binomial fraction
    ``n / N``: its interval at ``cl`` is Clopper-Pearson's, times ``gain``. The
    flow cells are left at 0 for the caller, and ``None`` is returned where
    there are no visible counts.
    """
    total = float(counts[visible].sum())
    if total <= 0:
        return None
    lower, upper = clopper_pearson(counts, total, cl)
    fraction = counts / total
    down = np.where(visible, gain * (fraction - lower), 0.0)
    up = np.where(visible, gain * (upper - fraction), 0.0)
    return np.asarray(down, dtype=float), np.asarray(up, dtype=float)
