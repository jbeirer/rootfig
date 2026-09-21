"""Drawing the tasks of a :class:`~rootfig.PlotBook` onto the pages of one PDF.

The book prepares its tasks as for any output (:meth:`PlotBook._prepared_tasks`:
batched reads, one preparation for the variants that only change the drawing);
here each is drawn into a cell of the current page with
:func:`~rootfig.api.plots1d.draw_plot` and the page is written and closed before
the next one is begun, so one page figure exists at a time.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from matplotlib.colors import to_hex

from rootfig.plotting import Plot, figure_size, style_context
from rootfig.plotting.figure import close_figures_since, open_figure_ids
from rootfig.plotting.pages import (
    Page,
    background_color,
    make_page,
    multipage_pdf,
    page_size,
)

if TYPE_CHECKING:
    from rootfig.api.batch import PlotTask


class PreparedTask(Protocol):
    """A task of a book with its histograms prepared, as the book's iterator yields them."""

    @property
    def task(self) -> PlotTask: ...

    def draw(self, **extra: Any) -> Plot:
        """Draw the task with its own keywords plus ``extra``; errors carry the task's note."""
        ...


@contextmanager
def task_note(task: PlotTask, doing: str) -> Iterator[None]:
    """Add a note naming ``task`` to any error from the block; the type is kept.

    ``doing`` says what was being done (``"sizing"``), so a failure outside
    :meth:`PlotBook.plots`' own drawing step still names the task at fault.
    """
    try:
        yield
    except Exception as exc:
        exc.add_note(f"while {doing} PlotBook task {task.describe()}")
        raise


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
        with task_note(task, "sizing"), style_context(task.kwargs.get("style")) as st:
            sizes.append(figure_size(st, ratio=bool(task.kwargs.get("ratio", False))))
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


def check_page_backgrounds(tasks: Sequence[PlotTask], pages: Sequence[Page]) -> None:
    """Raise if the plots sharing a page want different page backgrounds.

    A page is one figure and so has one background, which the first plot's style
    gives it (:func:`write_pdf`). The labels, ticks and legends of the others are
    drawn outside their axes, on that background rather than their own, so a dark
    plot beside a light one loses its white labels against the light page. There
    is no per-cell figure to fix this with, so such a book is refused before
    anything is drawn and pointed at one document per style. Styles that compare
    equal are not resolved at all: a book with one style throughout, which is the
    usual one, costs nothing here.

    Raises
    ------
    ValueError
        Two plots of one page resolve to different backgrounds.
    """
    position = 0
    for number, page in enumerate(pages, start=1):
        own = tasks[position : position + page.count]
        position += page.count
        style = own[0].kwargs.get("style")
        if all(task.kwargs.get("style") == style for task in own[1:]):
            continue
        first, *rest = ((task, _page_background(task)) for task in own)
        clash = next(((task, colour) for task, colour in rest if colour != first[1]), None)
        if clash is None:
            continue
        msg = (
            f"the plots of PDF page {number} ask for different page backgrounds: "
            f"{first[0].describe()} wants {to_hex(first[1], keep_alpha=True)} and "
            f"{clash[0].describe()} wants {to_hex(clash[1], keep_alpha=True)}; one page is one "
            f"figure and has one background, so write a document per style, e.g. "
            f"book.select(variants={clash[0].variant_id!r}).save_pdf(...)"
        )
        raise ValueError(msg)


def _page_background(task: PlotTask) -> tuple[float, float, float, float]:
    """Return the page background ``task``'s own style asks for."""
    with task_note(task, "resolving the page background of"):
        return background_color(task.kwargs.get("style"))


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
    check_page_backgrounds(tasks, pages)
    sizes = page_sizes(tasks, pages, figsize)
    position = 0
    with multipage_pdf(target, metadata=metadata) as pdf:
        for number, (page, size) in enumerate(zip(pages, sizes, strict=True), start=1):
            first = tasks[position]
            position += page.count
            before = open_figure_ids()
            try:
                with task_note(first, f"opening PDF page {number} for"):
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
