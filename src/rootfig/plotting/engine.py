"""The layout engine of rootfig's figures: constrained layout, then the x labels.

Constrained layout sizes every axes each time a figure is drawn. Whether an x
label fits beside its axis' offset text depends on that size, so it is decided
there too, right after the layout, at every draw: a figure that is saved, shown
or resized places its labels for the size it has then.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.transforms import ScaledTranslation, Transform

__all__ = ["OFFSET_TEXT_GAP_EM", "PlotLayoutEngine", "clear_offset_text", "renderer_of"]

OFFSET_TEXT_GAP_EM = 0.5
"""Horizontal gap between an x label and the offset text beside it, in units of the
offset text's font size."""


def renderer_of(fig: Any) -> Any:
    """Return a renderer for measuring artists, or ``None`` if the backend has none."""
    get = getattr(fig.canvas, "get_renderer", None)
    if callable(get):
        return get()
    private = getattr(fig, "_get_renderer", None)
    return private() if callable(private) else None


@dataclass
class _ClearXLabel:
    """The x label of ``ax`` kept clear of the axis' offset text.

    ``transform`` and ``labelpad`` are the label's own, before any move, so every
    placement starts from them and a label can move back as well as away.
    """

    ax: Axes
    transform: Transform
    labelpad: float
    below: bool = False

    def place(self, renderer: Any) -> bool:
        """Put the label beside or below the offset text; return whether it changed line.

        Matplotlib puts both at the right end of the axis, a label with
        ``loc="right"`` ending there and the offset text (``1e-6``) below the tick
        labels, and moves neither. The label moves left until it ends
        :data:`OFFSET_TEXT_GAP_EM` before the offset text or, where it would then
        start left of the axes, goes below the offset text: constrained layout
        reserves the height of an x label but never its width, so a label pushed
        past the axes could leave the canvas. Both moves are in points, so they
        hold at any dpi. mplhep's ``xlabel_sci_adjust`` moves the label only when
        the formatter uses an additive offset, but the order of magnitude is shown
        without one too (``axes.formatter.useoffset: False``).
        """
        axis = self.ax.xaxis
        label, offset = axis.label, axis.get_offset_text()
        fig = self.ax.get_figure(root=True)
        label.set_transform(self.transform)
        below, height = False, 0.0
        shown = label.get_visible() and label.get_text()
        if fig is not None and shown and offset.get_visible() and offset.get_text():
            points = 72.0 / fig.dpi
            label_box = label.get_window_extent(renderer)
            offset_box = offset.get_window_extent(renderer)
            height = offset_box.height * points
            # both sit a pad below the tick labels; the label's own pad may clear it
            same_line = self.labelpad < axis.OFFSETTEXTPAD + height
            if same_line and label_box.x1 > offset_box.x0 and offset_box.x1 > label_box.x0:
                gap = OFFSET_TEXT_GAP_EM * offset.get_fontproperties().get_size_in_points()
                shift = (label_box.x1 - offset_box.x0) * points + gap
                if label_box.x0 - shift / points >= self.ax.get_window_extent(renderer).x0:
                    beside = ScaledTranslation(-shift / 72.0, 0.0, fig.dpi_scale_trans)
                    label.set_transform(self.transform + beside)
                else:
                    below = True
        # below: the label keeps its own pad under the offset text
        axis.labelpad = self.labelpad + (axis.OFFSETTEXTPAD + height if below else 0.0)
        changed = below != self.below
        self.below = below
        return changed


def clear_offset_text(ax: Axes, renderer: Any) -> None:
    """Keep the x label of ``ax`` clear of the axis' offset text, once, as laid out now.

    For a figure without a :class:`PlotLayoutEngine` (axes the caller made), which
    places its x labels at every draw instead. A label already clear stays where
    it is.
    """
    axis = ax.xaxis
    _ClearXLabel(ax, axis.label.get_transform(), axis.labelpad).place(renderer)


class PlotLayoutEngine(ConstrainedLayoutEngine):
    """Constrained layout, then the x labels kept clear of their offset texts.

    Every draw lays the figure out and then places each registered x label
    beside or below its axis' offset text for the width its axes have now
    (:meth:`keep_clear`). A label that changes line changes the height the layout
    must reserve, so the figure is laid out once more then; a label moved beside
    changes nothing constrained layout measures.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._xlabels: list[_ClearXLabel] = []

    def keep_clear(self, ax: Axes) -> None:
        """Keep the x label of ``ax`` clear of its offset text at every draw."""
        if all(entry.ax is not ax for entry in self._xlabels):
            axis = ax.xaxis
            self._xlabels.append(_ClearXLabel(ax, axis.label.get_transform(), axis.labelpad))

    def execute(self, fig: Figure) -> None:
        """Lay out ``fig``, then place its registered x labels for that layout."""
        super().execute(fig)
        renderer = renderer_of(fig) if self._xlabels else None
        if renderer is None:
            return
        moved = [entry.place(renderer) for entry in self._xlabels]  # every label, no shortcut
        if any(moved):
            super().execute(fig)
