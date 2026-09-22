"""Legends, statistics boxes and free text on axes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib as mpl
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend
from matplotlib.offsetbox import AnchoredText
from matplotlib.text import Text

from rootfig.histograms.build import Histogram
from rootfig.model.style import Style
from rootfig.plotting.style import foreground, legend_location

__all__ = ["add_legend", "add_stats_box", "add_text"]

_LOCATIONS = {
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
}


def add_legend(ax: Axes, style: Style, *, loc: str | None = None) -> Legend | None:
    """Add a legend with data entries first, then the other histograms in drawing order.

    mplhep already registers stacked histograms top-of-stack first, so the
    legend order matches what the eye sees. Returns ``None`` when the style
    disables legends or nothing is labelled.
    """
    location = loc or legend_location(style)
    if location is None:
        return None
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    ordered = _data_first(list(zip(handles, labels, strict=True)), ax)
    kwargs: dict[str, Any] = {"loc": location, **dict(style.legend_kwargs)}
    return ax.legend([h for h, _ in ordered], [lab for _, lab in ordered], **kwargs)


def _data_first(entries: list[tuple[Artist, str]], ax: Axes) -> list[tuple[Artist, str]]:
    """Move error-bar containers (data points) to the front of the legend."""
    data_labels = {
        c.get_label() for c in ax.containers if isinstance(c, mpl.container.ErrorbarContainer)
    }
    data = [e for e in entries if e[1] in data_labels]
    rest = [e for e in entries if e[1] not in data_labels]
    return data + rest


def add_stats_box(
    ax: Axes,
    histograms: Sequence[Histogram],
    *,
    loc: str = "auto",
    precision: int = 4,
    include_entries: bool = True,
    colors: Sequence[str] | None = None,
    legend: Legend | None = None,
) -> list[Text]:
    """Add ``N``, mean and standard deviation for each histogram with statistics.

    With ``loc="auto"`` the box sits in the upper right corner, directly below
    the legend if there is one there (the classic ROOT layout). Any matplotlib
    legend location string places it in that corner instead. One block of text
    is drawn per histogram. ``colors`` supplies one colour per input histogram;
    a length mismatch raises ``ValueError``.
    """
    if colors is not None and len(colors) != len(histograms):
        msg = f"got {len(colors)} colours for {len(histograms)} histograms"
        raise ValueError(msg)
    blocks: list[tuple[str, str]] = []
    for index, histogram in enumerate(histograms):
        if histogram.stats is None:
            continue
        body = histogram.stats.format(precision, include_entries=include_entries)
        text = f"{histogram.label}\n{body}" if len(histograms) > 1 else body
        color = colors[index] if colors is not None else foreground()
        blocks.append((text, color))
    if not blocks:
        return []
    if loc != "auto" and loc not in _LOCATIONS:
        msg = f"unknown stats box location {loc!r}; use 'auto' or a matplotlib legend location"
        raise ValueError(msg)

    x, y, ha, va = 0.97, 0.96, "right", "top"
    if loc == "auto" and legend is not None:
        bbox = _axes_bbox(ax, legend)
        if bbox is not None:
            x, y = bbox[2], bbox[1] - 0.02
    elif loc != "auto":
        x = 0.03 if "left" in loc else 0.5 if "center" in loc and "right" not in loc else 0.97
        ha = "left" if "left" in loc else "center" if x == 0.5 else "right"
        if "lower" in loc:
            y, va = 0.04, "bottom"
            blocks = blocks[::-1]
        elif loc in ("center", "center left", "center right", "right"):
            y, va = 0.5, "center"

    fontsize = FontProperties(size=mpl.rcParams["legend.fontsize"]).get_size_in_points() * 0.9
    texts: list[Text] = []
    fig = ax.get_figure(root=True)
    renderer = _renderer(ax)
    for text, color in blocks:
        artist = ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color,
            family="monospace",
            linespacing=1.25,
        )
        texts.append(artist)
        if renderer is not None and fig is not None:
            extent = artist.get_window_extent(renderer).transformed(ax.transAxes.inverted())
            step = extent.height + 0.02
            y = y - step if va == "top" else y + step
    return texts


def _renderer(ax: Axes) -> Any:
    fig = ax.get_figure(root=True)
    if fig is None:  # pragma: no cover - axes always belong to a figure
        return None
    canvas = fig.canvas
    if hasattr(canvas, "get_renderer"):
        return canvas.get_renderer()
    try:  # pragma: no cover - backend dependent
        canvas.draw()
        return getattr(canvas, "renderer", None)
    except Exception:
        return None


def _axes_bbox(ax: Axes, artist: Artist) -> tuple[float, float, float, float] | None:
    """Return ``(x0, y0, x1, y1)`` of ``artist`` in axes coordinates, if computable."""
    renderer = _renderer(ax)
    if renderer is None:
        return None
    extent = artist.get_window_extent(renderer).transformed(ax.transAxes.inverted())
    return (float(extent.x0), float(extent.y0), float(extent.x1), float(extent.y1))


def add_text(
    ax: Axes, text: str | Sequence[str], *, loc: str = "upper left", **kwargs: Any
) -> AnchoredText:
    """Add free text (one string or several lines) anchored inside ``ax``."""
    body = text if isinstance(text, str) else "\n".join(text)
    prop = {"fontsize": mpl.rcParams["legend.fontsize"], **kwargs.pop("prop", {})}
    box = AnchoredText(body, loc=loc, frameon=False, prop=prop, pad=0.3, borderpad=0.6, **kwargs)
    ax.add_artist(box)
    return box
