"""Drawing one-dimensional histograms: overlays, stacks and data points."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import hist
import mplhep as hep
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes

from rootfig.errors import BinningError
from rootfig.histograms.build import Histogram
from rootfig.histograms.ratio import compatible_binning
from rootfig.model.samples import HistType
from rootfig.model.style import Style
from rootfig.plotting.style import color_cycle

__all__ = [
    "DATA_STYLE",
    "Drawn",
    "FlowSpec",
    "draw_histograms",
    "envelope",
    "fold_flow_bins",
    "label_flow_bins",
    "show_flow_bins",
]

FlowSpec: TypeAlias = Literal["hint", "show", "sum", "none"]
"""How under/overflow is shown (mplhep ``flow``): small arrows hinting at flow content
(``"hint"``), extra bins (``"show"``), added to the edge bins (``"sum"``), or ignored."""

DATA_STYLE: dict[str, Any] = {"color": "black", "marker": "o", "markersize": 5, "capsize": 0}
"""Default appearance of data points."""


@dataclass(frozen=True)
class Drawn:
    """What :func:`draw_histograms` produced.

    ``colors`` maps legend labels to colours (for the statistics box);
    ``histogram_colors`` holds the colour of every input histogram in order, so
    callers can identify histograms with duplicate labels.
    """

    artists: list[Artist]
    labels: list[str]
    ymin: float
    ymax: float
    ymin_positive: float
    colors: dict[str, str]
    histogram_colors: list[str]


def _errors(histogram: Histogram) -> np.ndarray:
    return histogram.errors()


def _histplot(*args: Any, **kwargs: Any) -> Any:
    """Call ``mplhep.histplot`` without its scipy-less Poisson-interval warning.

    mplhep evaluates automatic uncertainties even for ``yerr=False`` and warns
    when scipy is absent; the errors are never drawn in that case, so the
    warning is noise for rootfig users.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Integer weights indicate poissonian data")
        return hep.histplot(*args, **kwargs)


def show_flow_bins(histograms: Sequence[Histogram]) -> tuple[list[Histogram], tuple[bool, bool]]:
    """Turn under/overflow into extra visible bins, consistently for all histograms.

    mplhep's ``flow="show"`` adds a flow bin to each histogram only when *that*
    histogram has content there, so a stack or overlay mixing histograms with and
    without overflow ends up with mismatched bins (a crash for stacks, misaligned
    axes otherwise). rootfig therefore materialises the flow bins itself: a bin is
    added on a side when *any* histogram has content there, with mplhep's width
    convention (the larger of 5 % of the range and the mean bin width), and every
    histogram gets the same edges. The flow bins of the returned histograms are empty.

    Returns
    -------
    histograms, (underflow_shown, overflow_shown)
    """
    if not histograms:
        return [], (False, False)
    _require_same_binning(histograms, "flow='show'")
    edges = histograms[0].edges
    under = any(_has_content(h.underflow, h.underflow_variance) for h in histograms)
    over = any(_has_content(h.overflow, h.overflow_variance) for h in histograms)
    if not (under or over):
        return list(histograms), (False, False)
    width = max(0.05 * float(edges[-1] - edges[0]), float(np.mean(np.diff(edges))))
    new_edges = edges
    if under:
        new_edges = np.r_[edges[0] - width, new_edges]
    if over:
        new_edges = np.r_[new_edges, edges[-1] + width]
    result: list[Histogram] = []
    for histogram in histograms:
        axis = hist.axis.Variable(new_edges, label=histogram.axis.label)
        new = hist.Hist(axis, storage=hist.storage.Weight())
        values, variances = histogram.values(), histogram.variances()
        if under:
            values = np.r_[histogram.underflow, values]
            variances = np.r_[histogram.underflow_variance, variances]
        if over:
            values = np.r_[values, histogram.overflow]
            variances = np.r_[variances, histogram.overflow_variance]
        view = new.view()
        view.value = values
        view.variance = variances
        result.append(histogram.with_(hist=new))
    return result, (under, over)


def fold_flow_bins(histograms: Sequence[Histogram]) -> list[Histogram]:
    """Add the under/overflow of each histogram to its first/last visible bin.

    This is mplhep's ``flow="sum"`` done once, up front, so that ratios,
    significances, stack totals, statistical bands and axis limits are all
    computed from the bins that are drawn. The flow bins of the returned
    histograms are empty (values and variances).
    """
    result: list[Histogram] = []
    for histogram in histograms:
        if not (
            _has_content(histogram.underflow, histogram.underflow_variance)
            or _has_content(histogram.overflow, histogram.overflow_variance)
        ):
            result.append(histogram)
            continue
        new = histogram.hist.copy()
        view: Any = new.view(flow=True)
        traits = histogram.axis.traits
        first, last = (1 if traits.underflow else 0), (-2 if traits.overflow else -1)
        if traits.underflow:
            view.value[first] += view.value[0]
            view.variance[first] += view.variance[0]
            view.value[0] = view.variance[0] = 0.0
        if traits.overflow:
            view.value[last] += view.value[-1]
            view.variance[last] += view.variance[-1]
            view.value[-1] = view.variance[-1] = 0.0
        result.append(histogram.with_(hist=new))
    return result


def _has_content(value: float, variance: float) -> bool:
    """Return True if a flow bin holds any weight (also negative or cancelling to zero)."""
    return bool(value != 0.0 or variance > 0.0)


def _require_same_binning(histograms: Sequence[Histogram], what: str) -> None:
    first = histograms[0].hist
    for histogram in histograms[1:]:
        if not compatible_binning(first, histogram.hist):
            msg = (
                f"{what} needs histograms with identical bin edges; {histograms[0].label!r} "
                f"and {histogram.label!r} differ"
            )
            raise BinningError(msg)


def label_flow_bins(ax: Axes, edges: np.ndarray, *, under: bool, over: bool) -> None:
    """Label the flow bins made by :func:`show_flow_bins` ``<low`` and ``>high`` on ``ax``."""
    low = edges[1] if under else edges[0]
    high = edges[-2] if over else edges[-1]
    # drop the tick at the boundary edge itself: the flow label sits right next to it
    ticks = [
        float(t)
        for t in ax.get_xticks()
        if low <= t <= high
        and not (under and np.isclose(t, low))
        and not (over and np.isclose(t, high))
    ]
    labels = [f"{t:g}" for t in ticks]
    if under:
        ticks.insert(0, float((edges[0] + edges[1]) / 2))
        labels.insert(0, f"<{low:g}")
    if over:
        ticks.append(float((edges[-2] + edges[-1]) / 2))
        labels.append(f">{high:g}")
    ax.set_xticks(ticks, labels)


def envelope(histograms: Sequence[Histogram], *, stack: bool) -> tuple[np.ndarray, np.ndarray]:
    """Bin edges and the highest drawn value (content plus uncertainty) per bin.

    Used to keep legends and labels clear of the histograms. For stacks the
    stack total counts; otherwise the maximum over all histograms. Overlaid
    histograms may have different binnings: the envelope is then evaluated on
    the union of all edges.
    """
    mc = [h for h in histograms if not h.is_data]
    tops: list[tuple[np.ndarray, np.ndarray]] = []
    if stack and mc:
        total = _stack_total(mc)
        tops.append((total.edges, total.values() + total.errors()))
    else:
        tops.extend((h.edges, h.values() + h.errors()) for h in mc)
    tops.extend((h.edges, h.values() + h.errors()) for h in histograms if h.is_data)
    edges = np.unique(np.concatenate([e for e, _ in tops] or [histograms[0].edges]))
    if not tops:
        return edges, np.zeros(len(edges) - 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    heights = np.full(centers.size, -np.inf)
    for own_edges, top in tops:
        index = np.searchsorted(own_edges, centers, side="right") - 1
        inside = (index >= 0) & (index < top.size)
        sampled = np.where(inside, top[np.clip(index, 0, top.size - 1)], -np.inf)
        heights = np.maximum(heights, sampled)
    heights = np.where(np.isfinite(heights), heights, 0.0)
    return edges, np.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)


def _stack_total(histograms: Sequence[Histogram]) -> Histogram:
    total = histograms[0].hist.copy()
    for histogram in histograms[1:]:
        total = total + histogram.hist
    return Histogram(total, label="Total", normalization=histograms[0].normalization)


def draw_histograms(
    histograms: Sequence[Histogram],
    ax: Axes,
    *,
    style: Style,
    stack: bool = False,
    histtype: HistType | None = None,
    errorbars: bool | None = None,
    flow: FlowSpec = "hint",
    stack_uncertainty: bool = True,
    alpha: float | None = None,
) -> Drawn:
    """Draw ``histograms`` on ``ax``.

    Histograms flagged ``is_data`` are always drawn as black points with error
    bars on top; the others are overlaid (default) or stacked.

    Parameters
    ----------
    histograms
        Histograms to draw, in legend order.
    ax
        Target axes.
    style
        Style providing the colour cycle.
    stack
        Stack the non-data histograms (filled).
    histtype
        Default drawing type for non-data histograms without their own
        ``histtype``; ``None`` means ``"fill"`` for stacks and ``"step"``
        otherwise.
    errorbars
        Draw statistical error bars on non-data histograms. ``None`` draws them
        only for ``"errorbar"`` histtypes.
    flow
        Under/overflow display, see :data:`FlowSpec`.
    stack_uncertainty
        Draw a hatched band for the statistical uncertainty of the stack total.
    alpha
        Opacity for filled histograms (default 1 for stacks, 0.4 for overlays).
    """
    artists: list[Artist] = []
    labels: list[str] = []
    ranges: list[tuple[float, float, float]] = []

    flow_shown = (False, False)
    if flow == "show":
        histograms, flow_shown = show_flow_bins(histograms)
        flow = "none"
    elif flow == "sum":
        histograms = fold_flow_bins(histograms)
        flow = "none"

    data = [h for h in histograms if h.is_data]
    mc = [h for h in histograms if not h.is_data]
    colors = _assign_colors(mc, style)
    used_colors: dict[str, str] = {h.label: c for h, c in zip(mc, colors, strict=True)}
    used_colors.update({h.label: (h.color or "black") for h in data})
    by_histogram = {id(h): c for h, c in zip(mc, colors, strict=True)}
    by_histogram.update({id(h): (h.color or "black") for h in data})
    histogram_colors = [by_histogram[id(h)] for h in histograms]

    if mc and stack:
        _require_same_binning(mc, "a stack")
        fill_alpha = 1.0 if alpha is None else alpha
        stacked = _histplot(
            [h.hist for h in mc],
            ax=ax,
            stack=True,
            histtype="fill",
            label=[h.label for h in mc],
            color=colors,
            flow=flow,
            yerr=False,
            alpha=fill_alpha,
            edgecolor="black",
            linewidth=0.5,
        )
        artists.extend(_flatten_artists(stacked))
        labels.extend(h.label for h in mc)
        total = _stack_total(mc)
        values, errors = total.values(), total.errors()
        ranges.append(_range(values, errors))
        if stack_uncertainty and np.any(errors > 0):
            band = _histplot(
                total.hist,
                ax=ax,
                histtype="band",
                yerr=errors,
                flow=flow,
                facecolor="none",
                edgecolor="black",
                hatch="////",
                linewidth=0.0,
                alpha=0.5,
                label="Stat. unc.",
            )
            artists.extend(_flatten_artists(band))
            labels.append("Stat. unc.")
    elif mc:
        default_type: HistType = histtype or "step"
        for histogram, color in zip(mc, colors, strict=True):
            kind: HistType = histogram.histtype or default_type
            errors = _errors(histogram)
            show_errors = errorbars if errorbars is not None else kind == "errorbar"
            kwargs: dict[str, Any] = {
                "ax": ax,
                "histtype": kind,
                "label": histogram.label,
                "color": color,
                "flow": flow,
                "yerr": errors if show_errors else False,
            }
            if kind == "fill":
                kwargs["alpha"] = 0.45 if alpha is None else alpha
                kwargs["edgecolor"] = color
                kwargs["linewidth"] = 1.0
            elif kind == "errorbar":
                kwargs.update({"marker": "o", "markersize": 4, "capsize": 0})
            elif kind == "step" and show_errors:
                kwargs["linewidth"] = 1.6
            drawn = _histplot(histogram.hist, **kwargs)
            artists.extend(_flatten_artists(drawn))
            labels.append(histogram.label)
            ranges.append(_range(histogram.values(), errors if show_errors else None))

    for histogram in data:
        errors = _errors(histogram)
        drawn = _histplot(
            histogram.hist,
            ax=ax,
            histtype="errorbar",
            yerr=errors,
            xerr=False,
            label=histogram.label,
            flow=flow,
            **{**DATA_STYLE, **({"color": histogram.color} if histogram.color else {})},
        )
        artists.extend(_flatten_artists(drawn))
        labels.append(histogram.label)
        ranges.append(_range(histogram.values(), errors))

    if any(flow_shown):
        label_flow_bins(ax, histograms[0].edges, under=flow_shown[0], over=flow_shown[1])

    if ranges:
        ymin = min(r[0] for r in ranges)
        ymax = max(r[1] for r in ranges)
        positives = [r[2] for r in ranges if np.isfinite(r[2])]
        ymin_positive = min(positives) if positives else float("nan")
    else:
        ymin, ymax, ymin_positive = 0.0, 0.0, float("nan")
    return Drawn(artists, labels, ymin, ymax, ymin_positive, used_colors, histogram_colors)


def _assign_colors(histograms: Sequence[Histogram], style: Style) -> list[str]:
    cycle = iter(color_cycle(max(len(histograms), 1), style))
    return [h.color if h.color else next(cycle) for h in histograms]


def _range(values: np.ndarray, errors: np.ndarray | None) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 0.0, float("nan"))
    if errors is not None and errors.shape == values.shape:
        upper = finite + errors[np.isfinite(values)]
        lower = finite - errors[np.isfinite(values)]
    else:
        upper = lower = finite
    positive = finite[finite > 0]
    return (
        float(lower.min()),
        float(upper.max()),
        float(positive.min()) if positive.size else float("nan"),
    )


def _flatten_artists(result: Any) -> list[Artist]:
    """Mplhep returns lists of named tuples of artists; flatten to a list of artists."""
    found: list[Artist] = []
    if result is None:
        return found
    if isinstance(result, Artist):
        return [result]
    if isinstance(result, tuple) and hasattr(result, "_fields"):
        for field_name in result._fields:
            found.extend(_flatten_artists(getattr(result, field_name)))
        return found
    if isinstance(result, list | tuple):
        for item in result:
            found.extend(_flatten_artists(item))
    return found
