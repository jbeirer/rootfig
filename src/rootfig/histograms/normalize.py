"""Histogram normalisation."""

from __future__ import annotations

import functools
import warnings
from typing import Any, Literal, TypeAlias

import numpy as np

from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError, RootfigWarning, SystematicError
from rootfig.histograms.build import Histogram, _error_sides, as_weight_storage
from rootfig.histograms.intervals import count_scale
from rootfig.histograms.shape import shape_bounds, shape_covariance_matrix, shape_variances

__all__ = [
    "NormalizeSpec",
    "NormalizeUncertainty",
    "normalization_label",
    "normalize",
    "normalize_hist",
    "shape_covariance",
]

NormalizeSpec: TypeAlias = bool | Literal["unity", "density", "width"] | float | int | None
"""How to normalise a histogram.

* ``False``/``None`` - raw sums of weights.
* ``True`` or ``"unity"`` - scale so the visible bins sum to one.
* ``"density"`` - scale so the integral over the visible range is one
  (contents divided by bin width and total).
* ``"width"`` - divide each bin by its width (``Events / GeV``), no rescaling.
* a number - scale so the visible bins sum to that number.
"""


NormalizeUncertainty: TypeAlias = Literal["scale", "shape"]
"""What normalising to a histogram's own total does to its statistical uncertainty.

* ``"scale"`` - the factor is taken as a constant, as ``TH1::Scale`` does: every
  bin keeps its relative uncertainty.
* ``"shape"`` - the total fluctuates with the bins, which are then
  anti-correlated: a bin's variance is propagated to first order through the
  division by the total (``p (1 - p) / N`` for a fraction ``p`` of ``N``
  counts), and counts with a Poisson interval get the Clopper-Pearson interval
  of their fraction of the total, at the same confidence level. See
  :func:`shape_covariance` for the correlations.
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

    Variances are multiplied by the square of the factor, which is taken as a
    constant although the rescaling modes derive it from the histogram's own
    total: the bins keep their relative uncertainties, and the correlation the
    shared total brings (a normalised shape's covariance) is not represented.

    The rescaling modes divide by the signed sum of the visible bins, so the
    bins sum to the target even when negative weights dominate (the shape then
    flips sign, with a warning). A histogram whose visible bins sum to zero,
    because it is empty or because positive and negative weights cancel, is
    returned unchanged with a warning.
    """
    return _normalize_hist(histogram, spec)[0]


def _normalize_hist(
    histogram: Hist, spec: NormalizeSpec, factor: float | None = None
) -> tuple[Hist, bool, float]:
    """Normalise ``histogram``: the result, whether the mode was applied, and the factor.

    ``factor`` imposes the overall factor instead of deriving it from the
    histogram's own total (a record that must follow another histogram).
    """
    mode = _mode(spec)
    histogram = as_weight_storage(histogram)
    if mode is None:
        return histogram.copy(), False, 1.0
    if mode in ("density", "width"):
        scale = 1.0
        if mode == "density":
            derived = _scale_factor(histogram, 1.0) if factor is None else factor
            if derived is None:
                return histogram.copy(), False, 1.0
            scale = derived
        result = histogram.copy()
        sizes = _bin_sizes(histogram)
        view: Any = result.view(flow=True)
        view.value /= sizes
        view.variance /= sizes**2
        return (result * scale if scale != 1.0 else result), True, scale
    target = 1.0 if mode == "unity" else float(mode)
    rescale = _scale_factor(histogram, target) if factor is None else factor
    if rescale is None:
        return histogram.copy(), False, 1.0
    return histogram * rescale, True, rescale


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


def _rescales(mode: str | float | None) -> bool:
    """Whether ``mode`` divides by the histogram's own total."""
    return mode is not None and mode != "width"


def _check_uncertainty(uncertainty: str, mode: str | float | None) -> None:
    if uncertainty not in ("scale", "shape"):
        msg = f"the normalisation uncertainty must be 'scale' or 'shape', got {uncertainty!r}"
        raise ValueError(msg)
    if uncertainty == "shape" and not _rescales(mode):
        msg = (
            "a shape uncertainty needs a normalisation to the histogram's own total "
            f"(True, 'unity', 'density' or a number), got {mode!r}"
        )
        raise ValueError(msg)


def _visible(histogram: Hist) -> np.ndarray:
    """Return which cells (flow cells included) are visible bins."""
    marker = histogram.copy()
    view: Any = marker.view(flow=True)
    view.value = 0.0
    inner: Any = marker.view(flow=False)
    inner.value = 1.0
    return np.asarray(marker.values(flow=True) == 1.0, dtype=bool)


def _gain(histogram: Hist, mode: str | float, *, flow: bool) -> FloatArray | float:
    """Return the factor of ``values / total`` a rescaling mode applies: target or 1 / size."""
    if mode == "density":
        sizes = _bin_sizes(histogram)
        if not flow:
            sizes = sizes[_visible(histogram)].reshape(np.shape(histogram.values(flow=False)))
        return np.asarray(1.0 / sizes, dtype=float)
    return 1.0 if mode == "unity" else float(mode)


def shape_covariance(histogram: Histogram | Hist, spec: NormalizeSpec = True) -> FloatArray:
    """Return the covariance matrix of the visible bins of ``histogram`` normalised by ``spec``.

    The histogram's own total fluctuates with its bins, so they are
    anti-correlated (see :data:`NormalizeUncertainty`): the first-order
    covariance ``J V J^T`` of ``normalize(histogram, spec)``, the bins
    flattened in C order. Its diagonal holds the variances that
    ``normalize(..., uncertainty="shape")`` gives a histogram of summed weights.

    Raises
    ------
    ValueError
        Unless ``spec`` normalises to the histogram's own total.
    """
    mode = _mode(spec)
    _check_uncertainty("shape", mode)
    assert mode is not None
    hist_ = histogram.hist if isinstance(histogram, Histogram) else as_weight_storage(histogram)
    values = np.asarray(hist_.values(flow=False), dtype=float)
    variances = np.asarray(hist_.variances(flow=False), dtype=float)
    return shape_covariance_matrix(values, variances, _gain(hist_, mode, flow=False))


def _with_shape_errors(original: Histogram, normalized: Histogram, mode: str | float) -> Histogram:
    """Give ``normalized`` the uncertainty of a shape (see :data:`NormalizeUncertainty`)."""
    values = original.values(flow=True)
    visible = _visible(original.hist)
    gain = np.broadcast_to(_gain(original.hist, mode, flow=True), values.shape)
    if original.poisson and original._unit is not None:
        unit = original._unit
        factor = count_scale(unit.values(flow=True), np.asarray(unit.variances(flow=True)))
        known = factor[visible]
        if known.size and np.all(known > 0) and np.allclose(known, known.flat[0], rtol=1e-12):
            counts = np.rint(np.where(factor > 0, values / np.where(factor > 0, factor, 1.0), 0.0))
            bounds = shape_bounds(counts, gain, visible, original._cl)
            if bounds is not None:
                flow = np.sqrt(
                    shape_variances(values, original.variances(flow=True), gain, visible)
                )
                sides = (np.where(visible, bounds[0], flow), np.where(visible, bounds[1], flow))
                return normalized.replace(
                    poisson=False,
                    _unit=None,
                    _errors=_error_sides(normalized.hist, sides, normalized.label),
                )
    if original._errors is not None:
        down, up = (
            np.sqrt(shape_variances(values, np.asarray(side.variances(flow=True)), gain, visible))
            for side in original._errors
        )
        sides = (down, up)
        return normalized.replace(
            poisson=False,
            _unit=None,
            _errors=_error_sides(normalized.hist, sides, normalized.label),
        )
    result = normalized.hist.copy()
    view: Any = result.view(flow=True)
    view.variance = shape_variances(values, original.variances(flow=True), gain, visible)
    return normalized.replace(hist=result, poisson=False, _weighted=True)


def normalize(
    histogram: Histogram, spec: NormalizeSpec, *, uncertainty: NormalizeUncertainty = "scale"
) -> Histogram:
    """Return a normalised copy of a :class:`Histogram`, recording the mode.

    ``uncertainty`` says what normalising to the histogram's own total does to
    its statistical uncertainty (see :data:`NormalizeUncertainty`): ``"scale"``
    keeps every bin's relative uncertainty, as ``TH1::Scale``; ``"shape"``
    lets the total fluctuate with the bins.

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
    _check_uncertainty(uncertainty, mode)
    if mode is None:
        return histogram
    result, applied, factor = _normalize_hist(histogram.hist, spec)
    if not applied:
        return histogram.replace(
            hist=result,
            normalization=None,
            _unit=histogram._unit,
            _weighted=histogram._weighted,
            _errors=histogram._errors,
        )
    label = normalization_label(spec)
    variations = {}
    for name, pair in histogram.variations.items():
        normalized = []
        for direction, varied in zip(("up", "down"), pair, strict=True):
            shifted, shift_applied, _ = _normalize_hist(varied, spec)
            if not shift_applied:
                msg = (
                    f"histogram {histogram.label!r}: systematic {name!r} {direction} cannot "
                    "be normalised; its visible total must be finite and non-zero"
                )
                raise SystematicError(msg)
            normalized.append(shifted)
        variations[name] = (normalized[0], normalized[1])
    # the record of counts and the given errors follow the nominal's factor
    unit = histogram._unit
    if unit is not None:
        unit = _normalize_hist(unit, spec, factor=factor)[0]
    errors = histogram._errors
    if errors is not None:
        errors = (
            _normalize_hist(errors[0], spec, factor=factor)[0],
            _normalize_hist(errors[1], spec, factor=factor)[0],
        )
    rescaled = histogram.replace(
        hist=result,
        normalization=label,
        variations=variations,
        _unit=unit,
        _weighted=True,
        _errors=errors,
    )
    if uncertainty == "shape":
        assert mode is not None
        return _with_shape_errors(histogram, rescaled, mode)
    return rescaled


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
