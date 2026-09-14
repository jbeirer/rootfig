"""Histogram normalisation."""

from __future__ import annotations

import functools
import warnings
from typing import Any, Literal, TypeAlias

import numpy as np

from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError, RootfigWarning, SystematicError
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

    The rescaling modes divide by the signed sum of the visible bins, so the
    bins sum to the target even when negative weights dominate (the shape then
    flips sign, with a warning). A histogram whose visible bins sum to zero,
    because it is empty or because positive and negative weights cancel, is
    returned unchanged with a warning.
    """
    return _normalize_hist(histogram, spec)[0]


def _normalize_hist(histogram: Hist, spec: NormalizeSpec) -> tuple[Hist, bool]:
    """Normalise ``histogram``; also report whether the requested mode was applied."""
    mode = _mode(spec)
    histogram = as_weight_storage(histogram)
    if mode is None:
        return histogram.copy(), False
    if mode in ("density", "width"):
        factor = 1.0
        if mode == "density":
            scale = _scale_factor(histogram, 1.0)
            if scale is None:
                return histogram.copy(), False
            factor = scale
        result = histogram.copy()
        sizes = _bin_sizes(histogram)
        view: Any = result.view(flow=True)
        view.value /= sizes
        view.variance /= sizes**2
        return (result * factor if factor != 1.0 else result), True
    target = 1.0 if mode == "unity" else float(mode)
    scale = _scale_factor(histogram, target)
    if scale is None:
        return histogram.copy(), False
    return histogram * scale, True


def _scale_factor(histogram: Hist, target: float) -> float | None:
    """Factor scaling the visible bins to sum to ``target``, or ``None`` (with a warning)."""
    total = float(histogram.values(flow=False).sum())
    if total == 0.0:
        if float(np.asarray(histogram.variances(flow=False), dtype=float).sum()) == 0.0:
            _warn("histogram has no entries in the visible range; normalisation skipped")
        else:
            _warn(
                "histogram weights sum to zero in the visible range (positive and negative "
                "weights cancel); normalisation skipped"
            )
        return None
    if not np.isfinite(total):
        _warn(
            f"histogram total is not finite ({total}) in the visible range; normalisation skipped"
        )
        return None
    if total < 0:
        _warn(
            f"histogram has a negative total ({total:g}) in the visible range; normalising to "
            f"{target:g} flips its sign"
        )
    return target / total


def _warn(message: str) -> None:
    warnings.warn(message, RootfigWarning, stacklevel=5)


def normalize(histogram: Histogram, spec: NormalizeSpec) -> Histogram:
    """Return a normalised copy of a :class:`Histogram`, recording the mode.

    ``normalization`` is set only when the normalisation was actually applied;
    a histogram that could not be normalised (see :func:`normalize_hist`) keeps
    ``None`` so labels do not claim a scaling that did not happen.

    Systematic variations are normalised the same way as the nominal histogram,
    each by its own total, so a normalised plot shows how a variation changes the
    shape; a pure normalisation uncertainty drops out of the rescaling modes
    (``unity``, ``density`` and numeric targets). ``width`` retains it.
    If the nominal cannot be normalised, all variations stay unchanged too.
    A variation that cannot be normalised when the nominal can raises
    :class:`~rootfig.errors.SystematicError`, avoiding a mixture of raw and
    normalised contents in the uncertainty.
    """
    mode = _mode(spec)
    if mode is None:
        return histogram
    result, applied = _normalize_hist(histogram.hist, spec)
    if not applied:
        return histogram.replace(hist=result, normalization=None)
    label = normalization_label(spec)
    variations = {}
    for name, pair in histogram.variations.items():
        normalized = []
        for direction, varied in zip(("up", "down"), pair, strict=True):
            shifted, shift_applied = _normalize_hist(varied, spec)
            if not shift_applied:
                msg = (
                    f"histogram {histogram.label!r}: systematic {name!r} {direction} cannot "
                    "be normalised; its visible total must be finite and non-zero"
                )
                raise SystematicError(msg)
            normalized.append(shifted)
        variations[name] = (normalized[0], normalized[1])
    return histogram.replace(hist=result, normalization=label, variations=variations)


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
