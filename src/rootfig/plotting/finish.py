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
from weakref import WeakKeyDictionary

import mplhep as hep
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rootfig.plotting.figure import fit_ylabel, lay_out, without_redraw
from rootfig.plotting.style import align_experiment_labels

__all__ = ["Finish", "finish_figure", "finishing_together"]


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


_collected: WeakKeyDictionary[Figure, list[Finish]] = WeakKeyDictionary()


def finish_figure(fig: Figure, plots: Sequence[Finish]) -> None:
    """Finish ``plots``, all drawn on ``fig``, against the figure's layout.

    Call once drawing is complete and the style context has ended, as the figure is
    shown and saved from there. One layout pass serves every plot: the x labels move
    left of their offset texts (``mplhep.xlabel_sci_adjust``), each plot's
    headroom is raised and its lower panels' y labels are fitted, and last the
    experiment labels are aligned
    (:func:`~rootfig.plotting.align_experiment_labels`, whose passes are shared
    too). Within :func:`finishing_together` for ``fig`` the plots are collected
    instead. Nothing is measured if the backend cannot measure artists.
    """
    if fig in _collected:
        _collected[fig].extend(plots)
        return
    if any(p.headroom is not None or p.panels or p.xlabel is not None for p in plots):
        if lay_out(fig) is None:
            return
        for plot in plots:
            if plot.xlabel is not None:
                with without_redraw(fig):  # laid out above
                    hep.xlabel_sci_adjust(plot.xlabel)
            if plot.headroom is not None:
                plot.headroom()
            for panel in plot.panels:
                fit_ylabel(panel, laid_out=True)
    align_experiment_labels([(plot.main, plot.right) for plot in plots])


@contextmanager
def finishing_together(fig: Figure) -> Iterator[None]:
    """Finish the plots drawn onto ``fig`` in the block together, once it completes.

    The plotting functions finish their figure before returning; drawing several
    plots onto the cells of one page, each would lay out the whole page again, so
    what they finish on ``fig`` is collected here and finished in one go at the end
    of the block. A block that raises finishes nothing.
    """
    _collected[fig] = []
    try:
        yield
        plots = _collected[fig]
    finally:
        del _collected[fig]
    finish_figure(fig, plots)
