"""Figure layout and axis finishing shared by all plot types."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.ticker import MaxNLocator

from rootfig.model.style import Style

__all__ = [
    "AxesLike",
    "Layout",
    "apply_xbreak",
    "break_segments",
    "finish_axes",
    "fit_ylabel",
    "make_figure",
    "ylabel_for",
]

AxesLike: TypeAlias = Axes | tuple[Axes, Axes] | Sequence[Axes] | None
"""Where to draw: nothing (new figure), one axes, or ``(main, ratio)`` axes."""

RATIO_HEIGHT_FRACTION = 0.3
"""Height of the ratio panel relative to the main panel."""

RATIO_LABEL_MIN_SCALE = 0.6
"""Smallest y label size of a lower panel, relative to the style's label size."""

BREAK_GAP = 0.04
LAYOUT_PAD = 0.04  # inches between the canvas edge and the outermost artist
"""Horizontal gap between the two segments of a broken x axis (figure width fraction)."""


@dataclass
class Layout:
    """The axes of a figure: main panel, optional ratio panel, optional right segments.

    With a broken x axis every panel exists twice: ``main``/``main_right`` and
    ``ratio``/``ratio_right`` show the left and right segments.
    """

    fig: Figure
    main: Axes
    ratio: Axes | None = None
    main_right: Axes | None = None
    ratio_right: Axes | None = None

    @property
    def main_axes(self) -> tuple[Axes, ...]:
        """The main-panel axes, left to right."""
        return (self.main,) if self.main_right is None else (self.main, self.main_right)

    @property
    def ratio_axes(self) -> tuple[Axes, ...]:
        """The ratio-panel axes, left to right (empty without a ratio panel)."""
        if self.ratio is None:
            return ()
        return (self.ratio,) if self.ratio_right is None else (self.ratio, self.ratio_right)

    @property
    def is_broken(self) -> bool:
        """True if the x axis is split into two segments."""
        return self.main_right is not None

    @property
    def legend_axes(self) -> Axes:
        """Where the legend and statistics box go: the right-most main axes."""
        return self.main_right if self.main_right is not None else self.main

    @property
    def xlabel_axes(self) -> Axes:
        """Where the x label goes: the bottom right axes."""
        bottom = self.ratio_axes or self.main_axes
        return bottom[-1]


def make_figure(
    style: Style,
    *,
    ratio: bool,
    ax: AxesLike = None,
    figsize: tuple[float, float] | None = None,
    break_widths: tuple[float, float] | None = None,
) -> Layout:
    """Create (or reuse) the figure and axes for a plot.

    Parameters
    ----------
    style
        Provides the default figure size.
    ratio
        Add a ratio panel below the main panel, sharing the x axis.
    ax
        Existing axes to draw into: one ``Axes``, or ``(main, ratio)``. Not
        supported together with ``break_widths``.
    figsize
        Figure size in inches; defaults to the style's, enlarged for a ratio panel.
    break_widths
        Relative widths of the left and right segments of a broken x axis.
        ``None`` for an ordinary single x axis.
    """
    if ax is not None:
        if break_widths is not None:
            msg = "a broken x axis (xbreak) cannot be drawn into existing axes; leave ax=None"
            raise ValueError(msg)
        if isinstance(ax, Axes):
            if ratio:
                msg = "a ratio panel needs two axes: pass ax=(main_ax, ratio_ax)"
                raise ValueError(msg)
            return Layout(_figure_of(ax), ax)
        axes = tuple(ax)
        if len(axes) != 2 or not all(isinstance(a, Axes) for a in axes):
            msg = "ax must be a single Axes or a pair (main_ax, ratio_ax)"
            raise ValueError(msg)
        main, lower = axes
        return Layout(_figure_of(main), main, lower if ratio else None)

    size = figsize or style.figsize
    if size is None:
        width, height = plt.rcParams["figure.figsize"]
        size = (width, height * (1 + RATIO_HEIGHT_FRACTION * 0.85)) if ratio else (width, height)
    # Constrained layout fits labels, legends and colour bars into the canvas, so a saved
    # figure has exactly the requested size and every plot type shares one shape.
    engine = ConstrainedLayoutEngine(w_pad=LAYOUT_PAD, h_pad=LAYOUT_PAD)
    fig = plt.figure(figsize=size, layout=engine)
    rows = 2 if ratio else 1
    columns = 2 if break_widths is not None else 1
    grid = fig.add_gridspec(
        rows,
        columns,
        height_ratios=[1.0, RATIO_HEIGHT_FRACTION] if ratio else None,
        width_ratios=list(break_widths) if break_widths is not None else None,
        hspace=0.06,
        wspace=BREAK_GAP,
    )
    main = fig.add_subplot(grid[0, 0])
    layout = Layout(fig, main)
    _pin_tick_label_size(main)
    if columns == 2:
        layout.main_right = fig.add_subplot(grid[0, 1], sharey=main)
        _pin_tick_label_size(layout.main_right)
    if ratio:
        layout.ratio = fig.add_subplot(grid[1, 0], sharex=main)
        _pin_tick_label_size(layout.ratio)
        main.tick_params(axis="x", labelbottom=False)
        if columns == 2:
            assert layout.main_right is not None
            layout.ratio_right = fig.add_subplot(
                grid[1, 1], sharex=layout.main_right, sharey=layout.ratio
            )
            _pin_tick_label_size(layout.ratio_right)
            layout.main_right.tick_params(axis="x", labelbottom=False)
    return layout


def _pin_tick_label_size(ax: Axes) -> None:
    """Store the style's tick label size on the axes.

    Matplotlib's automatic tick locator estimates how many labels fit from the
    label size, which it reads from rcParams at draw time unless the axes carry
    an explicit ``labelsize``. Figures are drawn and saved after the style
    context has ended, so without this the locator assumes the default 10 pt
    labels and crowds the axis whenever the style uses a larger font.
    """
    x_size = FontProperties(size=plt.rcParams["xtick.labelsize"]).get_size_in_points()
    y_size = FontProperties(size=plt.rcParams["ytick.labelsize"]).get_size_in_points()
    ax.tick_params(axis="x", which="both", labelsize=x_size)
    ax.tick_params(axis="y", which="both", labelsize=y_size)


def _figure_of(ax: Axes) -> Figure:
    figure = ax.get_figure(root=True)
    if not isinstance(figure, Figure):  # pragma: no cover - matplotlib always sets it
        msg = "axes is not attached to a figure"
        raise ValueError(msg)
    return figure


# --------------------------------------------------------------------------------------
# Broken x axis
# --------------------------------------------------------------------------------------


def break_segments(
    outer: tuple[float, float],
    xbreak: tuple[float, float],
    *,
    logx: bool = False,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Split ``outer`` at ``xbreak`` into two x ranges and their relative widths.

    Returns ``(left_range, right_range, (left_width, right_width))``. Widths
    are proportional to the spans (in log space for ``logx``).

    Raises
    ------
    ValueError
        If the break does not satisfy ``outer[0] < a < b < outer[1]``.
    """
    try:
        a, b = (float(v) for v in xbreak)
    except (TypeError, ValueError) as exc:
        msg = f"xbreak must be a pair (a, b), got {xbreak!r}"
        raise ValueError(msg) from exc
    low, high = outer
    if not (low < a < b < high):
        msg = f"xbreak must satisfy {low:g} < a < b < {high:g}, got ({a:g}, {b:g})"
        raise ValueError(msg)
    if logx:
        if low <= 0:
            msg = "a broken logarithmic x axis needs a positive lower limit"
            raise ValueError(msg)
        spans = (math.log10(a) - math.log10(low), math.log10(high) - math.log10(b))
    else:
        spans = (a - low, high - b)
    total = spans[0] + spans[1]
    widths = (max(spans[0] / total, 0.15), max(spans[1] / total, 0.15))
    return (low, a), (b, high), widths


def apply_xbreak(
    left: Axes,
    right: Axes,
    left_range: tuple[float, float],
    right_range: tuple[float, float],
    *,
    mark_size: float = 0.012,
) -> None:
    """Style a pair of axes as the two segments of one broken x axis.

    Sets the x limits, removes the facing spines and ticks, and draws diagonal
    cut marks at the break. The right axes keeps no y tick labels (it shares
    y with the left one).
    """
    left.set_xlim(*left_range)
    right.set_xlim(*right_range)
    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    left.tick_params(axis="y", which="both", right=False)
    right.tick_params(axis="y", which="both", left=False, labelleft=False)
    right.set_ylabel("")
    # Drop the tick labels touching the break so "120" and "200" cannot collide.
    if left.get_xscale() == "linear":
        left.xaxis.set_major_locator(MaxNLocator(nbins="auto", prune="upper"))
    if right.get_xscale() == "linear":
        right.xaxis.set_major_locator(MaxNLocator(nbins="auto", prune="lower"))

    # Diagonal marks: scale the horizontal extent so both look alike despite
    # the different axes widths.
    fig = left.get_figure(root=True)
    width_left = left.get_position().width
    width_right = right.get_position().width
    reference = max(width_left, width_right) or 1.0
    d = mark_size
    for axes, x, scale in (
        (left, 1.0, reference / width_left),
        (right, 0.0, reference / width_right),
    ):
        dx = d * scale * 0.6
        kwargs: dict[str, Any] = {
            "transform": axes.transAxes,
            "color": "black",
            "clip_on": False,
            "linewidth": 1.0,
        }
        axes.plot((x - dx, x + dx), (-d, +d), **kwargs)
        axes.plot((x - dx, x + dx), (1 - d, 1 + d), **kwargs)
    del fig


# --------------------------------------------------------------------------------------
# Labels and limits
# --------------------------------------------------------------------------------------


def ylabel_for(
    *,
    normalization: str | None,
    unit: str | None,
    widths: np.ndarray | None,
    per_object: bool,
) -> str:
    """Default y-axis label: ``Events``, ``Entries / 2 GeV``, ``Normalised to unity``, ..."""
    if normalization is not None:
        if normalization == "Events / unit":
            return f"Entries / {unit}" if unit else "Entries / unit"
        return normalization
    noun = "Entries" if per_object else "Events"
    if widths is not None and widths.size and np.allclose(widths, widths[0]):
        width = float(widths[0])
        rounded = float(f"{width:.3g}")
        is_round = math.isclose(rounded, width, rel_tol=1e-9, abs_tol=0.0)
        if is_round and unit:
            return f"{noun} / {width:g} {unit}"
        if is_round and not math.isclose(width, 1.0):
            return f"{noun} / {width:g}"
    return noun


def finish_axes(
    ax: Axes,
    *,
    data_range: tuple[float, float],
    xlabel: str | None,
    ylabel: str | None,
    xlim: tuple[float, float] | None,
    ylim: tuple[float | None, float | None] | None,
    logx: bool,
    logy: bool,
    headroom: float = 1.45,
    log_headroom: float = 30.0,
    ymin_linear: Literal["zero", "auto"] = "zero",
) -> None:
    """Apply labels, scales and limits, leaving headroom for legends and labels.

    Parameters
    ----------
    data_range
        ``(min, max)`` of the drawn values (including error bars). For log
        axes the minimum should be the smallest positive value.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel, loc="right")
    if ylabel is not None:
        ax.set_ylabel(ylabel, loc="top")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)

    low, high = data_range
    if not (math.isfinite(low) and math.isfinite(high)):
        low, high = (1.0, 1.0) if logy else (0.0, 0.0)
    if logy:
        bottom = low / 2.0 if low > 0 else 0.5
        top = high * log_headroom if high > 0 else 10.0
    else:
        span = high - low
        bottom = 0.0 if (ymin_linear == "zero" and low >= 0) else low - 0.05 * span
        top = high + (headroom - 1.0) * (high - bottom) if high > bottom else bottom + 1.0
    if ylim is not None:
        user_low, user_high = ylim
        bottom = bottom if user_low is None else user_low
        top = top if user_high is None else user_high
    if top <= bottom:
        top = bottom + 1.0
    ax.set_ylim(bottom, top)


def _renderer(fig: Any) -> Any:
    """Return a renderer for measuring artists, or ``None`` if the backend has none."""
    get = getattr(fig.canvas, "get_renderer", None)
    if callable(get):
        return get()
    private = getattr(fig, "_get_renderer", None)
    return private() if callable(private) else None


def _balanced_wrap(text: str) -> str:
    """Break ``text`` at the space that leaves the two lines most even in length."""
    words = text.split()
    if len(words) < 2:
        return text
    split = min(
        range(1, len(words)),
        key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))),
    )
    return " ".join(words[:split]) + "\n" + " ".join(words[split:])


def fit_ylabel(
    ax: Axes, *, min_scale: float = RATIO_LABEL_MIN_SCALE, fraction: float = 0.98
) -> None:
    """Shrink (and if necessary wrap) the y label of ``ax`` until it fits the panel.

    A rotated y label is bounded by the *height* of its axes, and a lower panel is
    a fraction of the main one, so a label inherited at the main panel's size —
    ``"Ratio to <sample>"`` is easily twice as tall as the ratio panel — runs into
    the panel above and off the canvas. Constrained layout does not help: it
    reserves width for a y label, never height.

    The label is measured against the drawn panel and shrunk to fit, down to
    ``min_scale`` of its current size; if that is not enough it is wrapped onto
    two lines (never mathtext, which must not be broken) and shrunk again. A very
    long label in a very short panel stops at the floor rather than becoming
    unreadable. A label that already fits is left untouched, as is a figure whose
    backend cannot measure artists.
    """
    label = ax.yaxis.label
    text = label.get_text()
    fig = ax.get_figure(root=True)
    if not text or fig is None:
        return
    fig.canvas.draw()  # constrained layout sizes the axes at draw time
    renderer = _renderer(fig)
    if renderer is None:
        return

    def size() -> float:
        # get_fontsize() may be a named size ("large"); ask for the resolved points
        return float(label.get_fontproperties().get_size_in_points())

    def overflow() -> float:
        available = ax.get_window_extent(renderer).height * fraction
        if available <= 0:  # pragma: no cover - degenerate axes
            return 0.0
        return float(label.get_window_extent(renderer).height / available)

    floor = size() * min_scale
    over = overflow()
    if over <= 1.0:
        return
    if size() / over >= floor:
        label.set_fontsize(size() / over)
        return
    if "$" not in text:
        label.set_text(_balanced_wrap(text))
        over = overflow()
    if over > 1.0:
        label.set_fontsize(max(size() / over, floor))


def overlay_artists(ax: Axes, legend: Artist | None) -> list[Artist]:
    """Return the legend, anchored text boxes and axes-relative texts drawn on ``ax``.

    These are the things that must not cover the histograms: the legend, the
    experiment label, the statistics box and free text lines.
    """
    found: list[Artist] = [legend] if legend is not None else []
    found.extend(a for a in ax.artists if isinstance(a, AnchoredOffsetbox))
    found.extend(t for t in ax.texts if t.get_transform() == ax.transAxes)
    return found


def raise_ylim_above(
    axes: Sequence[Axes],
    obstacles: Sequence[Artist],
    *,
    edges: np.ndarray,
    heights: np.ndarray,
    logy: bool,
    floating: Sequence[Artist] = (),
    margin: float = 1.05,
) -> None:
    """Raise the upper y limit of ``axes`` until ``obstacles`` clear the histograms.

    ``heights`` is the tallest drawn value per bin (see
    :func:`~rootfig.plotting.hist1d.envelope`). ``obstacles`` are anchored in axes
    coordinates (a legend with a fixed location, the experiment label, text boxes):
    they keep their place while the data shrinks beneath them, so for each one the
    histogram maximum under its horizontal extent is compared with its lower edge
    and, if it would cover the histogram, the shared y range is stretched until
    the edge sits ``margin`` above the histogram. ``floating`` artists (a legend
    with ``loc="best"``) are relocated by matplotlib at draw time; for those only
    the size matters and room is made for them in the upper left *or* upper right
    corner, whichever needs less. Nothing happens if the backend cannot measure
    artists.
    """
    if not axes or (not obstacles and not floating) or heights.size == 0:
        return
    renderer = _renderer(axes[0].figure)
    if renderer is None:
        return
    new_top: float | None = None

    def consider(ax: Axes, fraction: float, lo: float, hi: float) -> None:
        nonlocal new_top
        bottom, top = ax.get_ylim()
        if logy and bottom <= 0:
            return
        # an obstacle low in the axes cannot be helped by more headroom
        if not 0.25 <= fraction < 1.0:
            return
        covered = (edges[1:] > lo) & (edges[:-1] < hi)
        if not covered.any():
            return
        needed = float(heights[covered].max()) * margin
        if not math.isfinite(needed) or needed <= 0 or (logy and needed <= bottom):
            return
        if logy:
            at_fraction = bottom * (top / bottom) ** fraction
            required = bottom * (needed / bottom) ** (1.0 / fraction)
        else:
            at_fraction = bottom + fraction * (top - bottom)
            required = bottom + (needed - bottom) / fraction
        if at_fraction < needed and math.isfinite(required):
            new_top = required if new_top is None else max(new_top, required)

    for ax in axes:
        x_low, x_high = sorted(ax.get_xlim())
        to_axes = ax.transAxes.inverted()
        to_data = ax.transData.inverted()

        def x_at(fraction: float, ax: Axes = ax, to_data: Any = to_data) -> float:
            return float(to_data.transform(ax.transAxes.transform((fraction, 0.0)))[0])

        for artist in obstacles:
            bbox = _extent(artist, renderer)
            if bbox is None:
                continue
            fraction = float(to_axes.transform((bbox.x0, bbox.y0))[1])
            span = to_data.transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y0]])[:, 0]
            consider(ax, fraction, max(min(span), x_low), min(max(span), x_high))
        for artist in floating:
            bbox = _extent(artist, renderer)
            if bbox is None:
                continue
            (fx0, fy0), (fx1, fy1) = to_axes.transform(bbox.get_points())
            width, height = fx1 - fx0, fy1 - fy0
            pad = 0.02
            fraction = 1.0 - height - pad
            corners = [(x_at(1.0 - width - pad), x_high), (x_low, x_at(width + pad))]
            candidates: list[float | None] = []
            for lo, hi in corners:
                before = new_top
                new_top = None
                consider(ax, fraction, lo, hi)
                candidates.append(new_top)
                new_top = before
            required = [c for c in candidates if c is not None]
            if len(required) == len(corners):  # neither corner is free: make room in one
                best = min(required)
                new_top = best if new_top is None else max(new_top, best)
    if new_top is not None:
        for ax in axes:
            ax.set_ylim(top=new_top)


def _extent(artist: Artist, renderer: Any) -> Any:
    try:
        bbox = artist.get_window_extent(renderer)
    except (AttributeError, RuntimeError, ValueError):
        return None
    if bbox.width <= 0 or bbox.height <= 0:
        return None
    return bbox
