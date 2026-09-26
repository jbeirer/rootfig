"""What the statistical errors of a :class:`~rootfig.histograms.Histogram` rest on."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from rootfig._typing import Hist

__all__ = ["Provenance", "error_sides", "one_count"]

_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0  # multiples of it are distinct modulo 1


@dataclass(frozen=True)
class Provenance:
    """The record behind a histogram's statistical errors, changed with its contents.

    Attributes
    ----------
    unit
        One unit count per cell, set iff the contents are known counts. Put
        through every transformation of the contents, a cell holds ``m c`` and
        ``m c**2`` for ``m`` merged counts of factor ``c``, so every cell's
        factor is known, empty cells included (see
        :func:`~rootfig.histograms.intervals.count_scale`). Dropped once a
        cell merges counts of different factors.
    weighted
        The contents were filled with weights, scaled or transformed, so they
        are never unit-weight counts, however they look.
    errors
        Errors given as ``stat_errors``: the ``(down, up)`` sides as the
        variances of two histograms, so scaling squares the factor and merging
        cells adds them in quadrature.
    """

    unit: Hist | None = None
    weighted: bool = False
    errors: tuple[Hist, Hist] | None = None

    def through(self, transform: Callable[[Hist], Hist], *, factor: float = 1.0) -> Provenance:
        """Return the record put through ``transform``, linear in the cells like the contents.

        ``factor`` is the transformation's overall factor, where it has one: a
        negative factor turns an interval over, so the given sides swap.
        """
        unit, weighted = self.unit, self.weighted
        if unit is not None:
            moved = transform(unit)
            unit = None if _mixes_factors(unit, moved, transform) else moved
            weighted |= unit is None  # no longer counts times one factor per cell
        errors = self.errors
        if errors is not None:
            down, up = transform(errors[0]), transform(errors[1])
            errors = (up, down) if factor < 0 else (down, up)
        return Provenance(unit, weighted, errors)


def _mixes_factors(unit: Hist, moved: Hist, transform: Callable[[Hist], Hist]) -> bool:
    """Say whether ``transform`` merged counts of different factors into one cell of ``unit``.

    A cell merging counts ``m_k`` of factors ``c_k`` holds ``sum m c`` and
    ``sum m c**2``. Weighting the cells before the merge by distinct positive
    numbers keeps the factor ``sum m c**2 / sum m c`` if every ``c_k`` is the
    same and, barring coincidences of the weights, only then.
    """
    probe = unit.copy()
    view: Any = probe.view(flow=True)
    weights = 1.0 + np.modf(np.arange(view.value.size) * _GOLDEN)[0].reshape(view.value.shape)
    view.value = view.value * weights
    view.variance = view.variance * weights
    probe = transform(probe)
    return not np.allclose(
        probe.variances(flow=True) * moved.values(flow=True),
        probe.values(flow=True) * moved.variances(flow=True),
        rtol=1e-9,
        atol=0.0,
    )


def one_count(histogram: Hist) -> Hist:
    """Return one unit count in every cell of ``histogram``: a new :attr:`Provenance.unit`."""
    unit = histogram.copy()
    view: Any = unit.view(flow=True)
    view.value = 1.0
    view.variance = 1.0
    return unit


def error_sides(
    histogram: Hist, errors: tuple[npt.ArrayLike, npt.ArrayLike], label: str
) -> tuple[Hist, Hist]:
    """Hold ``(down, up)`` errors as the variances of two histograms shaped like ``histogram``.

    Raises
    ------
    ValueError
        Unless both sides hold a non-negative finite error per bin or per cell
        with the flow bins.
    """
    try:
        down, up = (np.asarray(side, dtype=float) for side in errors)
    except (TypeError, ValueError):
        msg = f"histogram {label!r}: stat_errors must be a (down, up) pair of arrays"
        raise ValueError(msg) from None
    visible = np.shape(histogram.values(flow=False))
    cells = np.shape(histogram.values(flow=True))
    sides = []
    for side in (down, up):
        if side.shape not in (visible, cells):
            msg = (
                f"histogram {label!r}: stat_errors need one error per bin {visible} or per "
                f"cell with the flow bins {cells}, got {side.shape}"
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(side) & (side >= 0)):
            msg = f"histogram {label!r}: stat_errors must be non-negative and finite"
            raise ValueError(msg)
        held = histogram.copy()
        view: Any = held.view(flow=True)
        if side.shape == cells:
            view.variance = side**2
        else:
            view.variance = 0.0
            inner: Any = held.view(flow=False)
            inner.variance = side**2
        sides.append(held)
    return sides[0], sides[1]
