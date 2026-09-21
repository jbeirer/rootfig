"""Finishing plots once their figure is laid out: what depends on the final size of the axes.

Constrained layout places the axes when a figure is drawn, so every step that
measures a plot against its axes (the headroom above the histograms, the y label
of a lower panel, the x label under the offset text, the experiment label) runs
on the finished figure, after a layout pass. A pass costs the whole figure, and a
page of a :class:`~rootfig.PlotBook` holds many plots, which constrained layout
places independently of each other; so the plots of one figure are finished
together, in a fixed number of passes, rather than each laying out the page again.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import ScaledTranslation

from rootfig.plotting.figure import fit_ylabel, lay_out
from rootfig.plotting.style import align_experiment_labels

__all__ = [
    "OFFSET_TEXT_GAP_EM",
    "Finish",
    "clear_offset_text",
    "finish_figure",
    "finishing_together",
]


@dataclass(frozen=True)
class Finish:
    """What one plot does once its figure is laid out.

    ``main`` carries the experiment label (``right``, the right segment of a broken
    x axis, its luminosity text), ``panels`` are the lower panels whose y label is
    fitted to their height, ``xlabel`` is the axes whose x label must clear the
    axis' offset text (``1e-6`` at the right end, where a label with
    ``loc="right"`` ends too), and ``headroom`` raises the y limit until legends and labels
    clear the histograms (:func:`~rootfig.plotting.raise_ylim_above` with its
    arguments bound; ``None`` when the limit is the caller's).
    """

    main: Axes
    right: Axes | None = None
    panels: tuple[Axes, ...] = ()
    xlabel: Axes | None = None
    headroom: Callable[[], None] | None = None


OFFSET_TEXT_GAP_EM = 0.5
"""Horizontal gap between an x label and the offset text beside it, in units of the
offset text's font size."""

_collected: WeakKeyDictionary[Figure, list[Finish]] = WeakKeyDictionary()


def finish_figure(fig: Figure, plots: Sequence[Finish]) -> None:
    """Finish ``plots``, all drawn on ``fig``, against the figure's layout.

    Call once drawing is complete and the style context has ended, as the figure is
    shown and saved from there. Layout passes serve every plot: the lower panels'
    y labels are fitted, the experiment labels are aligned
    (:func:`~rootfig.plotting.align_experiment_labels`, whose passes are shared
    too), and then each plot's headroom is raised, so it measures the labels where
    they end up. A raised limit can change the y axis' offset text and hence the
    axes geometry, so one more shared layout/headroom pass follows a raise, then
    the experiment labels are realigned to dodge the offset text. Last, the x labels
    are kept clear of their offset texts (:func:`clear_offset_text`), beside them
    where the settled width of the axes leaves room and below them otherwise.
    Within :func:`finishing_together` for ``fig`` the plots are collected instead.
    Nothing is measured if the backend cannot measure artists.
    """
    if fig in _collected:
        _collected[fig].extend(plots)
        return
    if any(p.headroom is not None or p.panels or p.xlabel is not None for p in plots):
        if lay_out(fig) is None:
            return
        for plot in plots:
            for panel in plot.panels:
                fit_ylabel(panel, laid_out=True)
    labels = [(plot.main, plot.right) for plot in plots]
    align_experiment_labels(labels)
    raised = False
    for plot in plots:
        if plot.headroom is not None:
            top = plot.main.get_ylim()[1]
            plot.headroom()
            raised = raised or plot.main.get_ylim()[1] != top
    if raised:
        if lay_out(fig) is None:
            return
        for plot in plots:
            if plot.headroom is not None:
                plot.headroom()
        align_experiment_labels(labels)
    xlabels = [
        plot.xlabel
        for plot in plots
        if plot.xlabel is not None
        and plot.xlabel.get_xlabel()
        and plot.xlabel.xaxis.get_offset_text().get_text()
    ]
    if xlabels:
        renderer = lay_out(fig)
        if renderer is None:
            return
        for ax in xlabels:
            clear_offset_text(ax, renderer)


@contextmanager
def finishing_together(fig: Figure) -> Iterator[None]:
    """Finish the plots drawn onto ``fig`` in the block together, once it completes.

    The plotting functions finish their figure before returning; drawing several
    plots onto the cells of one page, each would lay out the whole page again, so
    what they finish on ``fig`` is collected here and finished in one go at the end
    of the outermost block. A block that raises discards only its own plots.
    """
    parent = _collected.get(fig)
    plots: list[Finish] = []
    _collected[fig] = plots
    try:
        yield
    finally:
        if parent is None:
            del _collected[fig]
        else:
            _collected[fig] = parent
    finish_figure(fig, plots)


def clear_offset_text(ax: Axes, renderer: Any) -> None:
    """Keep the x label of ``ax`` clear of the axis' offset text.

    Matplotlib puts both at the right end of the axis, a label with
    ``loc="right"`` ending there and the offset text (``1e-6``) below the tick
    labels, and moves neither. The label moves left until it ends
    :data:`OFFSET_TEXT_GAP_EM` before the offset text, or, where it would then
    start left of the axes, below the offset text instead: constrained layout
    reserves the height of an x label but never its width, so a label pushed past
    the axes could leave the canvas. Both moves are offsets in points, which hold
    when the figure is resized or saved at another dpi. mplhep's
    ``xlabel_sci_adjust`` moves the label only when the formatter uses an additive
    offset, but the order of magnitude is shown without one too
    (``axes.formatter.useoffset: False``). A label that does not overlap is left
    where it is, so finishing again changes nothing.
    """
    axis = ax.xaxis
    label, offset = axis.label, axis.get_offset_text()
    fig = ax.get_figure(root=True)
    if fig is None or not (label.get_visible() and offset.get_visible()):
        return
    label_box = label.get_window_extent(renderer)
    offset_box = offset.get_window_extent(renderer)
    if not label_box.overlaps(offset_box):
        return
    points = 72.0 / fig.dpi
    gap = OFFSET_TEXT_GAP_EM * offset.get_fontproperties().get_size_in_points()
    shift = (label_box.x1 - offset_box.x0) * points + gap
    if label_box.x0 - shift / points >= ax.get_window_extent(renderer).x0:
        beside = ScaledTranslation(-shift / 72.0, 0.0, fig.dpi_scale_trans)
        label.set_transform(label.get_transform() + beside)
    else:
        # the pad below the tick labels grows by what the label shares with the
        # offset text, and by the pad again to keep that gap below the offset text
        axis.labelpad += (label_box.y1 - offset_box.y0) * points + axis.labelpad
