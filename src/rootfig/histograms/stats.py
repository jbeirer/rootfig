"""Summary statistics and correlations of the (unbinned) values that fill a histogram."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rootfig._typing import FloatArray
from rootfig.errors import SelectionError

if TYPE_CHECKING:
    from rootfig.selection import Columns

__all__ = ["Summary", "correlation_matrix", "describe_table", "summarize"]


@dataclass(frozen=True)
class Summary:
    """Weighted summary statistics of one column of values.

    All moments use the weights as frequency weights. ``sem`` is the standard
    error of the mean, ``std / sqrt(n_eff)`` with the Kish effective sample size
    ``n_eff = (sum w)^2 / sum w^2``.

    With negative weights (e.g. NLO simulation) the weighted second moment can
    be negative; the standard deviation, its error and the skewness are then
    ``nan`` (the histogram itself is unaffected).
    """

    entries: int
    sum_weights: float
    mean: float
    std: float
    sem: float
    skewness: float
    minimum: float
    maximum: float
    n_events: int = 0
    n_selected_events: int = 0
    n_missing: int = 0
    n_nonfinite: int = 0
    per_object: bool = False

    @property
    def rms(self) -> float:
        """Alias for :attr:`std` (ROOT calls the standard deviation "RMS")."""
        return self.std

    @property
    def effective_entries(self) -> float:
        """Kish effective sample size, equal to ``entries`` for unit weights."""
        return self.sum_weights**2 / self._sum_w2 if self._sum_w2 else 0.0

    _sum_w2: float = 0.0

    def format(self, precision: int = 4, *, include_entries: bool = True) -> str:
        r"""Return a compact multi-line text, e.g. for a statistics box on a plot.

        Uses matplotlib math text for the symbols so it renders with the plot
        fonts (``$\mu$``, ``$\sigma$``).
        """
        lines = []
        if include_entries:
            lines.append(f"N = {self.entries}")
        lines.append(f"$\\mu$ = {_fmt(self.mean, precision)}")
        lines.append(f"$\\sigma$ = {_fmt(self.std, precision)}")
        return "\n".join(lines)


def _fmt(value: float, precision: int) -> str:
    if not math.isfinite(value):
        return "nan"
    if value != 0 and (abs(value) < 10.0 ** -(precision - 1) or abs(value) >= 10.0**precision):
        return f"{value:.{max(precision - 1, 1)}e}"
    return f"{value:.{precision}g}"


def summarize(columns: Columns, index: int = 0) -> Summary:
    """Compute :class:`Summary` statistics for column ``index`` of ``columns``."""
    values = columns.arrays[index]
    weights = columns.effective_weights()
    n = int(values.size)
    common: dict[str, Any] = {
        "entries": n,
        "n_events": columns.n_events,
        "n_selected_events": columns.n_selected_events,
        "n_missing": columns.n_missing,
        "n_nonfinite": columns.n_nonfinite,
        "per_object": columns.per_object,
    }
    sum_w = float(weights.sum())
    sum_w2 = float((weights**2).sum())
    if n == 0 or sum_w == 0:
        nan = float("nan")
        return Summary(
            sum_weights=sum_w,
            mean=nan,
            std=nan,
            sem=nan,
            skewness=nan,
            minimum=nan,
            maximum=nan,
            _sum_w2=sum_w2,
            **common,
        )
    mean = float(np.average(values, weights=weights))
    centred = values - mean
    variance = float(np.average(centred**2, weights=weights))
    # negative weights can make the weighted second moment negative: undefined, not an error
    std = math.sqrt(variance) if variance >= 0 else float("nan")
    n_eff = sum_w**2 / sum_w2 if sum_w2 else 0.0
    sem = std / math.sqrt(n_eff) if n_eff > 0 else float("nan")
    third = float(np.average(centred**3, weights=weights))
    skewness = third / std**3 if std > 0 else float("nan")
    return Summary(
        sum_weights=sum_w,
        mean=mean,
        std=std,
        sem=sem,
        skewness=skewness,
        minimum=float(values.min()),
        maximum=float(values.max()),
        _sum_w2=sum_w2,
        **common,
    )


def correlation_matrix(columns: Columns) -> FloatArray:
    """Return the (weighted) Pearson correlation matrix of all columns in ``columns``.

    Raises
    ------
    SelectionError
        If fewer than two columns or fewer than two entries (with non-zero
        weight) are available, or weights are negative.
    """
    if len(columns.arrays) < 2:
        msg = "a correlation matrix needs at least two variables"
        raise SelectionError(msg)
    if columns.n_entries < 2:
        msg = "a correlation matrix needs at least two entries after selection"
        raise SelectionError(msg)
    matrix = np.vstack(columns.arrays)
    weights = None if columns.weights is None else columns.weights
    if weights is not None:
        if np.any(weights < 0):
            msg = "correlation matrices with negative weights are not supported"
            raise SelectionError(msg)
        sum_w = float(weights.sum())
        sum_w2 = float((weights**2).sum())
        # numpy's weighted covariance divides by sum(w) * (1 - 1/n_eff): with the Kish
        # effective sample size n_eff = sum(w)^2 / sum(w^2) at or below one it is undefined.
        if sum_w <= 0 or sum_w**2 <= sum_w2:
            msg = (
                "a correlation matrix needs at least two entries with non-zero weight "
                "after selection"
            )
            raise SelectionError(msg)
    covariance = np.atleast_2d(np.cov(matrix, aweights=weights))
    diagonal = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.outer(diagonal, diagonal)
    correlation[~np.isfinite(correlation)] = np.nan
    np.fill_diagonal(correlation, np.where(diagonal > 0, 1.0, np.nan))
    return np.asarray(correlation, dtype=float)


def describe_table(summaries: Sequence[tuple[str, Summary]], precision: int = 4) -> str:
    """Format ``(label, summary)`` pairs as an aligned plain-text table."""
    header = ["", "entries", "mean", "std", "sem", "skew", "min", "max"]
    rows = [header]
    for label, s in summaries:
        rows.append(
            [
                label,
                str(s.entries),
                _fmt(s.mean, precision),
                _fmt(s.std, precision),
                _fmt(s.sem, precision),
                _fmt(s.skewness, precision),
                _fmt(s.minimum, precision),
                _fmt(s.maximum, precision),
            ]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    return "\n".join(
        "  ".join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]) for i, cell in enumerate(row)
        )
        for row in rows
    )
