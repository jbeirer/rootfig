"""Histogram normalisation."""

from __future__ import annotations

import functools
import warnings
from typing import Any, Literal, TypeAlias

import numpy as np

from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError, RootfigWarning
from rootfig.histograms.build import Histogram, as_weight_storage

__all__ = ["NormalizeSpec", "normalization_label", "normalize", "normalize_hist"]

NormalizeSpec: TypeAlias = bool | Literal["unity", "density", "width"] | float | int | None
"""How to normalise a histogram.

* ``False``/``None`` - raw sums of weights.
* ``True`` or ``"unity"`` - scale so the visible bins sum to one.
* ``"density"`` - scale so the integral over the visible range is one
  (contents divided by bin width and total).
* ``"width"`` - divide each bin by its width (``Events / GeV``), no rescaling.
* a number - scale so the visible bins sum to that number.
"""


def _mode(spec: NormalizeSpec) -> str | float | None:
    if spec is None or spec is False:
        return None
    if spec is True:
        return "unity"
    if isinstance(spec, str):
        if spec not in ("unity", "density", "width"):
            msg = (
                "normalize must be True/False, 'unity', 'density', 'width' or a number, "
                f"got {spec!r}"
            )
            raise BinningError(msg)
        return spec
    if isinstance(spec, int | float):
        if not np.isfinite(spec) or spec <= 0:
            msg = f"normalize target must be a positive finite number, got {spec!r}"
            raise BinningError(msg)
        return float(spec)
    msg = f"unsupported normalize specification {spec!r}"  # type: ignore[unreachable]
    raise BinningError(msg)


def _bin_sizes(histogram: Hist) -> FloatArray:
    """Bin sizes including the flow cells, which take the width of the neighbouring bin."""
    widths = []
    for axis in histogram.axes:
        width = np.asarray(axis.widths, dtype=float)
        if axis.traits.underflow:
            width = np.r_[width[0], width]
        if axis.traits.overflow:
            width = np.r_[width, width[-1]]
        widths.append(width)
    return np.asarray(functools.reduce(np.multiply.outer, widths), dtype=float)


def normalize_hist(histogram: Hist, spec: NormalizeSpec) -> Hist:
    """Return a normalised copy of ``histogram`` with ``Weight`` storage.

    Flow bins are scaled by the same factor as the visible bins for the
    rescaling modes, and divided by the neighbouring visible bin size for the
    per-width modes so they stay comparable when drawn. Histograms with a plain
    count storage are converted (see :func:`~rootfig.histograms.as_weight_storage`).
    """
    mode = _mode(spec)
    histogram = as_weight_storage(histogram)
    if mode is None:
        return histogram.copy()
    total = float(histogram.values(flow=False).sum())
    if mode in ("density", "width"):
        result = histogram.copy()
        sizes = _bin_sizes(histogram)
        view: Any = result.view(flow=True)
        view.value /= sizes
        view.variance /= sizes**2
        if mode == "density":
            if total <= 0:
                _warn_empty()
                return result
            result = result / total
        return result
    if total <= 0:
        _warn_empty()
        return histogram.copy()
    target = 1.0 if mode == "unity" else float(mode)
    return histogram * (target / total)


def _warn_empty() -> None:
    warnings.warn(
        "histogram has no entries in the visible range; normalisation skipped",
        RootfigWarning,
        stacklevel=4,
    )


def normalize(histogram: Histogram, spec: NormalizeSpec) -> Histogram:
    """Return a normalised copy of a :class:`Histogram`, recording the mode."""
    mode = _mode(spec)
    if mode is None:
        return histogram
    label = normalization_label(spec)
    return histogram.with_(hist=normalize_hist(histogram.hist, spec), normalization=label)


def normalization_label(spec: NormalizeSpec) -> str | None:
    """Short description of a normalisation, used as the y-axis label."""
    mode = _mode(spec)
    if mode is None:
        return None
    if mode == "unity":
        return "Normalised to unity"
    if mode == "density":
        return "Density"
    if mode == "width":
        return "Events / unit"
    return f"Normalised to {mode:g}"
