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

from rootfig._storage import is_category
from rootfig._typing import FloatArray
from rootfig.errors import BinningError
from rootfig.histograms.build import Histogram, compatible_binning
from rootfig.histograms.systematics import sum_histograms, uncertainty
from rootfig.model.samples import HistType
from rootfig.model.style import Style
from rootfig.plotting.style import color_cycle, foreground

__all__ = [
    "DATA_STYLE",
    "Drawn",
    "FlowSpec",
    "StackSpec",
    "band_label",
    "draw_histograms",
    "envelope",
    "fold_flow_bins",
    "in_view",
    "label_flow_bins",
    "require_same_binning",
    "show_flow_bins",
    "split_stack",
]

FlowSpec: TypeAlias = Literal["hint", "show", "sum", "none"]
"""How under/overflow is shown (mplhep ``flow``): small arrows hinting at flow content
(``"hint"``), extra bins (``"show"``), added to the edge bins (``"sum"``), or ignored."""

StackSpec: TypeAlias = bool | str | Sequence[str]
"""Stack all non-data histograms, none, or those carrying the given legend labels."""


def split_stack(
    histograms: Sequence[Histogram], stack: StackSpec
) -> tuple[list[Histogram], list[Histogram], list[Histogram]]:
    """Return stacked, overlaid and observed histograms, each in input order.

    Labels select every non-data histogram carrying them. Unknown labels and
    labels belonging only to observed data raise ``ValueError``; selectors other
    than a bool, a string or a sequence of strings raise ``TypeError``.
    """
    if isinstance(stack, bool):
        names = [h.label for h in histograms if not h.is_data] if stack else []
    elif isinstance(stack, str):
        names = [stack]
    elif isinstance(stack, Sequence) and all(isinstance(name, str) for name in stack):
        names = list(stack)
    else:
        msg = f"stack= must be True, False, a label or a list of labels, got {stack!r}"
        raise TypeError(msg)
    labels = [h.label for h in histograms]
    simulated = {h.label for h in histograms if not h.is_data}
    for name in names:
        if name in labels and name not in simulated:
            msg = (
                f"stack= names {name!r}, which is observed data; "
                "data is drawn as points and never stacked"
            )
            raise ValueError(msg)
    missing = [name for name in names if name not in labels]
    if missing:
        description = (
            "is not the label of a drawn histogram"
            if len(missing) == 1
            else "are not labels of drawn histograms"
        )
        msg = (
            f"stack= names {', '.join(repr(name) for name in missing)}, which {description} "
            f"(labels: {labels}); a Group is stacked by its own label, "
            "not by those of its components"
        )
        raise ValueError(msg)
    selected = set(names)
    return (
        [h for h in histograms if not h.is_data and h.label in selected],
        [h for h in histograms if not h.is_data and h.label not in selected],
        [h for h in histograms if h.is_data],
    )


DATA_STYLE: dict[str, Any] = {"marker": "o", "markersize": 5, "capsize": 0}
"""Default appearance of data points, drawn in the style's ink colour unless a sample sets one."""


def band_label(*, systematics: bool) -> str:
    """Label of an uncertainty band: statistical, or statistical and systematic."""
    return "Stat. + syst. unc." if systematics else "Stat. unc."


@dataclass(frozen=True)
class Drawn:
    """What :func:`draw_histograms` produced.

    ``colors`` holds one colour per input histogram, in input order, so
    duplicate labels stay distinct. ``stack`` is the summed histogram used for
    the stack's uncertainty band, including variations, or ``None``.
    """

    artists: list[Artist]
    labels: list[str]
    ymin: float
    ymax: float
    ymin_positive: float
    colors: list[str]
    stack: Histogram | None


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
    added on a side when *any* histogram (or systematic variation) has content
    there, with mplhep's width convention (the larger of 5 % of the range and the
    mean bin width), and every histogram gets the same edges. The flow bins of the
    returned histograms are empty.

    Returns
    -------
    histograms, (underflow_shown, overflow_shown)
    """
    if not histograms:
        return [], (False, False)
    require_same_binning(histograms, "flow='show'")
    _reject_category_flow(histograms, "flow='show'")
    edges = histograms[0].edges
    under = any(_flow_content(h, 0) for h in histograms)
    over = any(_flow_content(h, -1) for h in histograms)
    if not (under or over):
        return list(histograms), (False, False)
    width = max(0.05 * float(edges[-1] - edges[0]), float(np.mean(np.diff(edges))))
    new_edges = edges
    if under:
        new_edges = np.r_[edges[0] - width, new_edges]
    if over:
        new_edges = np.r_[new_edges, edges[-1] + width]

    def expand(h: Any) -> Any:
        axis = hist.axis.Variable(new_edges, name=h.axes[0].name, label=h.axes[0].label)
        new = hist.Hist(axis, storage=hist.storage.Weight())
        traits = h.axes[0].traits
        visible = slice(1 if traits.underflow else 0, -1 if traits.overflow else None)
        view = new.view()
        for field, flow_cells in (
            ("value", h.values(flow=True)),
            ("variance", h.variances(flow=True)),
        ):
            cells = np.asarray(flow_cells, dtype=float)
            low = [cells[0] if traits.underflow else 0.0] if under else []
            high = [cells[-1] if traits.overflow else 0.0] if over else []
            setattr(view, field, np.r_[low, cells[visible], high])
        return new

    def expand_sizes(h: Histogram) -> FloatArray | None:
        """Lay out the sizes the cells of ``h`` were divided by like ``expand``'s cells."""
        if h._sizes is None:
            return None
        traits = h.axis.traits
        cells = h._sizes
        inner = cells[1 if traits.underflow else 0 : -1 if traits.overflow else None]
        low = [cells[0] if traits.underflow else inner[0]] if under else []
        high = [cells[-1] if traits.overflow else inner[-1]] if over else []
        shown = np.r_[low, inner, high]
        # the new flow cells take their neighbours'
        return np.asarray(np.r_[shown[0], shown, shown[-1]], dtype=float)

    return [h.map_hists(expand, _sizes=expand_sizes(h)) for h in histograms], (under, over)


def fold_flow_bins(histograms: Sequence[Histogram]) -> list[Histogram]:
    """Add the under/overflow of each histogram to its first/last visible bin.

    This is mplhep's ``flow="sum"`` done once, up front, so that ratios,
    significances, stack totals, statistical bands and axis limits are all
    computed from the bins that are drawn. Systematic variations are folded the
    same way. The flow bins of the returned histograms are empty (values and
    variances).
    """
    _reject_category_flow(histograms, "flow='sum'")

    def fold(h: Any) -> Any:
        new = h.copy()
        view: Any = new.view(flow=True)
        traits = h.axes[0].traits
        first, last = (1 if traits.underflow else 0), (-2 if traits.overflow else -1)
        if traits.underflow:
            view.value[first] += view.value[0]
            view.variance[first] += view.variance[0]
            view.value[0] = view.variance[0] = 0.0
        if traits.overflow:
            view.value[last] += view.value[-1]
            view.variance[last] += view.variance[-1]
            view.value[-1] = view.variance[-1] = 0.0
        return new

    return [
        h.map_hists(fold) if _flow_content(h, 0) or _flow_content(h, -1) else h for h in histograms
    ]


def _flow_content(histogram: Histogram, side: int) -> bool:
    """Return True if the flow bin on ``side`` (0 under, -1 over) holds weight anywhere.

    The nominal histogram and every systematic variation count, and a bin whose
    weights cancel to zero still has content (its variance is positive).
    """
    traits = histogram.axis.traits
    if not (traits.underflow if side == 0 else traits.overflow):
        return False
    hists = [histogram.hist, *(h for pair in histogram.variations.values() for h in pair)]
    return any(
        _has_content(
            float(np.asarray(h.values(flow=True))[side]),
            float(np.asarray(h.variances(flow=True))[side]),
        )
        for h in hists
    )


def _has_content(value: float, variance: float) -> bool:
    """Return True if a flow bin holds any weight (also negative or cancelling to zero)."""
    return bool(value != 0.0 or variance > 0.0)


def _reject_category_flow(histograms: Sequence[Histogram], what: str) -> None:
    """Refuse ``what`` for a category axis whose flow bin holds entries.

    The overflow of a category axis collects the entries of categories the axis
    does not list. They have no place beyond the last category, so neither an
    extra bin nor folding them into that category represents them; the arrow of
    ``flow="hint"`` only says they exist.
    """
    for histogram in histograms:
        if is_category(histogram.axis) and (
            _flow_content(histogram, 0) or _flow_content(histogram, -1)
        ):
            msg = (
                f"{what} needs numeric bins: the flow bins of the category axis of "
                f"{histogram.label!r} hold entries of categories it does not list "
                f"({list(histogram.axis)}), which have no bin beyond the last category. Use "
                "flow='hint' or flow='none', or add the category to the histogram"
            )
            raise BinningError(msg)


def require_same_binning(histograms: Sequence[Histogram], what: str, *, flow: bool = False) -> None:
    """Refuse histograms ``what`` must combine bin by bin but which bin differently.

    Fewer than two histograms are always compatible. ``flow`` also asks for the
    same under- and overflow bins, which summing them needs; drawing them does
    not, and ``flow="show"`` gives them the flow bins itself.
    """
    if not histograms:
        return
    first = histograms[0].hist
    for histogram in histograms[1:]:
        if not compatible_binning(first, histogram.hist):
            msg = (
                f"{what} needs histograms with identical bin edges; {histograms[0].label!r} "
                f"and {histogram.label!r} differ"
            )
            raise BinningError(msg)
        if flow and not _same_flow_bins(first, histogram.hist):
            msg = (
                f"{what} needs histograms with the same flow bins; {histograms[0].label!r} "
                f"and {histogram.label!r} differ (hist.axis.Regular(..., underflow=, overflow=))"
            )
            raise BinningError(msg)


def _same_flow_bins(a: Any, b: Any) -> bool:
    """Whether two one-dimensional histograms have their flow bins alike."""
    return (a.axes[0].traits.underflow, a.axes[0].traits.overflow) == (
        b.axes[0].traits.underflow,
        b.axes[0].traits.overflow,
    )


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


def envelope(histograms: Sequence[Histogram], *, stack: StackSpec) -> tuple[np.ndarray, np.ndarray]:
    """Bin edges and the highest drawn value (content plus uncertainty) per bin.

    Used to keep legends and labels clear of the histograms. The uncertainty
    includes systematic variations, which are drawn as bands. The stack total,
    every overlaid histogram and every data histogram count separately. Overlaid
    histograms may have different binnings: the envelope is evaluated on the
    union of all edges.
    """
    stacked, overlaid, data = split_stack(histograms, stack)
    tops: list[tuple[np.ndarray, np.ndarray]] = []
    if stacked:
        tops.append(_top(sum_histograms(stacked)))
    tops.extend(_top(h) for h in overlaid)
    tops.extend(_top(h) for h in data)
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


def _top(histogram: Histogram) -> tuple[np.ndarray, np.ndarray]:
    return histogram.edges, histogram.values() + uncertainty(histogram).total_up


def draw_histograms(
    histograms: Sequence[Histogram],
    ax: Axes,
    *,
    style: Style,
    stack: StackSpec = False,
    histtype: HistType | None = None,
    errorbars: bool | None = None,
    flow: FlowSpec = "hint",
    stack_uncertainty: bool = True,
    alpha: float | None = None,
    view: Sequence[tuple[float, float]] | None = None,
) -> Drawn:
    """Draw ``histograms`` on ``ax``.

    Histograms flagged ``is_data`` are always drawn as points with error
    bars on top; the others are overlaid (default) or stacked.

    Parameters
    ----------
    histograms
        Histograms to draw, in input order.
    ax
        Target axes.
    style
        Style providing the colour cycle.
    stack
        ``True`` stacks every non-data histogram; a label or sequence of labels
        stacks those histograms. ``False`` or an empty sequence overlays all.
        Stacks keep input order, first at the bottom, followed by their band,
        overlays in input order, then data. Colours are assigned over all
        non-data histograms before splitting; explicit colours do not consume
        cycle entries. Stacked histograms are always filled.
    histtype
        Default drawing type for overlaid histograms without their own
        ``histtype``; ``None`` means ``"step"``. Ignored for stacked histograms,
        as is a histogram's own ``histtype``.
    errorbars
        Draw statistical error bars on overlaid histograms. ``None`` draws them
        only for ``"errorbar"`` histtypes. Ignored for stacked histograms.
    flow
        Under/overflow display, see :data:`FlowSpec`.
    stack_uncertainty
        Draw a hatched band for the uncertainty of the stack total: statistical,
        and systematic where the histograms carry variations. Overlaid
        histograms with variations get a light band in their own colour.
    alpha
        Opacity for filled histograms (default 1 for stacks, 0.45 for overlays).
    view
        The x windows whose bins set ``ymin``/``ymax``/``ymin_positive``.
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

    stacked, overlaid, data = split_stack(histograms, stack)
    mc = [h for h in histograms if not h.is_data]
    colors = _assign_colors(mc, style)
    by_histogram = {id(h): c for h, c in zip(mc, colors, strict=True)}
    by_histogram.update({id(h): (h.color or foreground()) for h in data})
    histogram_colors = [by_histogram[id(h)] for h in histograms]

    total = None
    if stacked:
        added_artists, added_labels, added_ranges, total = _draw_stack(
            stacked,
            ax,
            colors=[by_histogram[id(h)] for h in stacked],
            flow=flow,
            alpha=alpha,
            view=view,
            stack_uncertainty=stack_uncertainty,
        )
        artists.extend(added_artists)
        labels.extend(added_labels)
        ranges.extend(added_ranges)
    for histogram in overlaid:
        added_artists, added_labels, added_ranges = _draw_overlay(
            histogram,
            ax,
            color=by_histogram[id(histogram)],
            flow=flow,
            histtype=histtype,
            errorbars=errorbars,
            alpha=alpha,
            view=view,
        )
        artists.extend(added_artists)
        labels.extend(added_labels)
        ranges.extend(added_ranges)
    for histogram in data:
        added_artists, added_labels, added_ranges = _draw_data(histogram, ax, flow=flow, view=view)
        artists.extend(added_artists)
        labels.extend(added_labels)
        ranges.extend(added_ranges)

    if any(flow_shown):
        label_flow_bins(ax, histograms[0].edges, under=flow_shown[0], over=flow_shown[1])

    if ranges:
        ymin = min(r[0] for r in ranges)
        ymax = max(r[1] for r in ranges)
        positives = [r[2] for r in ranges if np.isfinite(r[2])]
        ymin_positive = min(positives) if positives else float("nan")
    else:
        ymin, ymax, ymin_positive = 0.0, 0.0, float("nan")
    return Drawn(artists, labels, ymin, ymax, ymin_positive, histogram_colors, total)


def _draw_stack(
    mc: Sequence[Histogram],
    ax: Axes,
    *,
    colors: list[str],
    flow: FlowSpec,
    alpha: float | None,
    stack_uncertainty: bool,
    view: Sequence[tuple[float, float]] | None,
) -> tuple[list[Artist], list[str], list[tuple[float, float, float]], Histogram]:
    """Draw the filled stack and its total uncertainty band."""
    artists: list[Artist] = []
    labels: list[str] = []
    ranges: list[tuple[float, float, float]] = []
    require_same_binning(mc, "a stack")
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
        edgecolor=foreground(),
        linewidth=0.5,
    )
    artists.extend(_flatten_artists(stacked))
    labels.extend(h.label for h in mc)
    total = sum_histograms(mc)
    summary = uncertainty(total)
    down, up = summary.total_down, summary.total_up
    ranges.append(_range(total.values(), (down, up), total.edges, view))
    if stack_uncertainty and (np.any(up > 0) or np.any(down > 0)):
        label = band_label(systematics=summary.has_systematics)
        band = _histplot(
            total.hist,
            ax=ax,
            histtype="band",
            yerr=[down, up] if summary.has_systematics else up,
            flow=flow,
            facecolor="none",
            edgecolor=foreground(),
            hatch="////",
            linewidth=0.0,
            alpha=0.5,
            label=label,
        )
        artists.extend(_flatten_artists(band))
        labels.append(label)
    return artists, labels, ranges, total


def _draw_overlay(
    histogram: Histogram,
    ax: Axes,
    *,
    color: str,
    flow: FlowSpec,
    histtype: HistType | None,
    errorbars: bool | None,
    alpha: float | None,
    view: Sequence[tuple[float, float]] | None,
) -> tuple[list[Artist], list[str], list[tuple[float, float, float]]]:
    """Draw one overlay with its errors and systematic band."""
    artists: list[Artist] = []
    labels: list[str] = []
    ranges: list[tuple[float, float, float]] = []
    default_type: HistType = histtype or "step"
    kind: HistType = histogram.histtype or default_type
    errors = histogram.errors()
    show_errors = errorbars if errorbars is not None else kind == "errorbar"
    kwargs: dict[str, Any] = {
        "ax": ax,
        "histtype": kind,
        "label": histogram.label,
        "color": color,
        "flow": flow,
        "yerr": list(errors) if show_errors else False,
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
    ranges.append(
        _range(histogram.values(), errors if show_errors else None, histogram.edges, view)
    )
    if histogram.variations:
        summary = uncertainty(histogram)
        bounds = (summary.total_down, summary.total_up)
        band = _histplot(
            histogram.hist,
            ax=ax,
            histtype="band",
            yerr=list(bounds),
            flow=flow,
            facecolor=color,
            edgecolor="none",
            hatch="",
            linewidth=0.0,
            alpha=0.3,
        )
        artists.extend(_flatten_artists(band))
        ranges.append(_range(histogram.values(), bounds, histogram.edges, view))

    return artists, labels, ranges


def _draw_data(
    histogram: Histogram,
    ax: Axes,
    *,
    flow: FlowSpec,
    view: Sequence[tuple[float, float]] | None,
) -> tuple[list[Artist], list[str], list[tuple[float, float, float]]]:
    """Draw one observed histogram as points with error bars."""
    artists: list[Artist] = []
    labels: list[str] = []
    ranges: list[tuple[float, float, float]] = []
    errors = histogram.errors()
    drawn = _histplot(
        histogram.hist,
        ax=ax,
        histtype="errorbar",
        yerr=list(errors),
        xerr=False,
        label=histogram.label,
        flow=flow,
        **{**DATA_STYLE, "color": histogram.color or foreground()},
    )
    artists.extend(_flatten_artists(drawn))
    labels.append(histogram.label)
    ranges.append(_range(histogram.values(), errors, histogram.edges, view))

    return artists, labels, ranges


def _assign_colors(histograms: Sequence[Histogram], style: Style) -> list[str]:
    cycle = iter(color_cycle(max(len(histograms), 1), style))
    return [h.color if h.color else next(cycle) for h in histograms]


def in_view(edges: np.ndarray, view: Sequence[tuple[float, float]] | None) -> np.ndarray:
    """Return a mask of bins overlapping any x window, or every bin for ``None``."""
    if view is None:
        return np.ones(len(edges) - 1, dtype=bool)
    visible = np.zeros(len(edges) - 1, dtype=bool)
    for low, high in view:
        visible |= (edges[:-1] < high) & (edges[1:] > low)
    return visible


def _range(
    values: np.ndarray,
    errors: np.ndarray | tuple[np.ndarray, np.ndarray] | None,
    edges: np.ndarray,
    view: Sequence[tuple[float, float]] | None,
) -> tuple[float, float, float]:
    """Lowest and highest drawn value and the smallest positive content.

    ``errors`` is symmetric or a ``(down, up)`` pair.
    """
    visible = np.isfinite(values) & in_view(edges, view)
    finite = values[visible]
    if finite.size == 0:
        return (0.0, 0.0, float("nan"))
    down, up = errors if isinstance(errors, tuple) else (errors, errors)
    if down is not None and up is not None and up.shape == values.shape:
        upper = finite + up[visible]
        lower = finite - down[visible]
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
