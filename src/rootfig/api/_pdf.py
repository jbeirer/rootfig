"""Drawing the tasks of a :class:`~rootfig.PlotBook` onto the pages of one PDF.

The book prepares its tasks as for any output (:meth:`PlotBook._prepared_tasks`:
batched reads, one preparation for the variants that only change the drawing);
here each is drawn into a cell of the current page with
:func:`~rootfig.api.plots1d.draw_plot` and the page is written and closed before
the next one is begun, so one page figure exists at a time.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rootfig.plotting import Plot, figure_size, style_context
from rootfig.plotting.figure import close_figures_since, open_figure_ids
from rootfig.plotting.pages import Page, make_page, multipage_pdf, page_size

if TYPE_CHECKING:
    from rootfig.api.batch import PlotTask


class PreparedTask(Protocol):
    """A task of a book with its histograms prepared, as the book's iterator yields them."""

    @property
    def task(self) -> PlotTask: ...

    def draw(self, **extra: Any) -> Plot:
        """Draw the task with its own keywords plus ``extra``; errors carry the task's note."""
        ...


def is_complex(kwargs: Mapping[str, Any]) -> bool:
    """Whether ``plot(**kwargs)`` draws a lower panel or a broken x axis, which need room."""
    return bool(kwargs.get("ratio", False)) or kwargs.get("xbreak") is not None


def cell_size(tasks: Sequence[PlotTask]) -> tuple[float, float]:
    """Return the cell size for ``tasks``: the largest width and height of their own figures.

    Each task's figure size is what :func:`~rootfig.plot` gives it on its own
    (its style's, taller with a ratio panel), so a cell shows its plot at that
    size; one size for every cell of the document keeps the pages alike.
    """
    sizes = []
    for task in tasks:
        try:
            with style_context(task.kwargs.get("style")) as st:
                sizes.append(figure_size(st, ratio=bool(task.kwargs.get("ratio", False))))
        except Exception as exc:
            exc.add_note(f"while sizing PlotBook task {task.describe()}")
            raise
    return (max(width for width, _ in sizes), max(height for _, height in sizes))


def page_sizes(
    tasks: Sequence[PlotTask], pages: Sequence[Page], figsize: tuple[float, float] | None
) -> list[tuple[float, float]]:
    """Return the figure size of every page of the document, in order.

    An explicit ``figsize`` is the size of each page, and the styles of the tasks
    are left alone; otherwise a page is its grid of cells of :func:`cell_size`.
    """
    if figsize is not None:
        return [figsize] * len(pages)
    inches = cell_size(tasks)
    return [page_size(page.grid, inches) for page in pages]


def write_pdf(
    prepared: Iterator[PreparedTask],
    tasks: Sequence[PlotTask],
    pages: Sequence[Page],
    *,
    target: Path,
    figsize: tuple[float, float] | None,
    savefig_kwargs: Mapping[str, Any],
) -> None:
    """Draw ``tasks`` onto ``pages`` and write them to ``target``.

    ``prepared`` yields the tasks in the order of ``tasks``, which ``pages`` split
    into consecutive runs; the tasks of a page are drawn in reading order into the
    cells of one figure of the size :func:`page_sizes` gives the page, created
    under the style of the page's first task. The document is written
    through :func:`~rootfig.plotting.pages.multipage_pdf`: an existing ``target``
    is replaced only once every page is written. A ``metadata`` among
    ``savefig_kwargs`` describes the document, not a page, so it goes to the
    writer; the rest is passed to every :meth:`PdfPages.savefig`. A failing task
    raises with its note (see :meth:`PlotBook.plots`), a failing write with a note
    naming the page and the file; the page figure is closed either way.
    """
    kwargs = {"facecolor": "auto", "edgecolor": "auto", **savefig_kwargs}
    metadata = kwargs.pop("metadata", None)
    sizes = page_sizes(tasks, pages, figsize)
    position = 0
    with multipage_pdf(target, metadata=metadata) as pdf:
        for number, (page, size) in enumerate(zip(pages, sizes, strict=True), start=1):
            first = tasks[position]
            position += page.count
            before = open_figure_ids()
            try:
                fig, cells = make_page(size, page.grid, style=first.kwargs.get("style"))
                for cell in cells[: page.count]:
                    next(prepared).draw(cell=cell)
                try:
                    pdf.savefig(fig, **kwargs)
                except Exception as exc:
                    exc.add_note(f"while writing PlotBook PDF page {number} to {str(target)!r}")
                    raise
            finally:
                close_figures_since(before)
