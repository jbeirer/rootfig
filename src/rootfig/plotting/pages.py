"""Several plots on one page: the grid, the page figure and a multipage PDF written safely.

A page is a figure with an outer grid whose cells each hold one plot, built by
:func:`~rootfig.plotting.make_figure` with ``cell=`` so that a cell has the same
panels as a figure of its own. The pages of a document are planned before
anything is drawn (:func:`plan_pages`) and written one after another
(:func:`multipage_pdf`).
"""

from __future__ import annotations

import math
import os
import secrets
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec

from rootfig.model.style import StyleLike
from rootfig.plotting.engine import PlotLayoutEngine
from rootfig.plotting.figure import LAYOUT_PAD
from rootfig.plotting.style import style_context

__all__ = [
    "Grid",
    "LayoutSpec",
    "Page",
    "auto_grid",
    "background_color",
    "cell_room",
    "check_figsize",
    "check_layout",
    "make_page",
    "multipage_pdf",
    "page_size",
    "pdf_target",
    "plan_pages",
]

Grid: TypeAlias = tuple[int, int]
"""``(rows, columns)`` of plots on a page."""

LayoutSpec: TypeAlias = Literal["auto"] | tuple[int, int]
"""How plots are arranged on pages: chosen from their number and kind, or a fixed grid."""

MAX_PLAIN_CELLS = 6
"""Most plots without a lower panel or broken axis on one automatically laid out page."""

MAX_COMPLEX_CELLS = 4
"""Most plots on one automatically laid out page when one of them has a lower panel or a
broken axis, which need the room."""

PAGE_GAP = 0.16
"""Inches between neighbouring cells of a page.

A cell is laid out like a figure of its own, with its outermost artists
:data:`~rootfig.plotting.figure.LAYOUT_PAD` inside its edges, so neighbouring
plots are this plus twice that apart.
"""

_PDF_SUFFIX = ".pdf"


@dataclass(frozen=True)
class Page:
    """One page of a document: its ``(rows, columns)`` grid and how many cells hold a plot.

    The plots take the first ``count`` cells in reading order.
    """

    grid: Grid
    count: int


def auto_grid(count: int, *, complex: bool) -> Grid:
    """Return the grid for ``count`` plots on a page laid out automatically.

    One plot fills the page, two share a row, three or four a ``2 x 2`` grid and
    more a ``2 x 3`` grid; ``complex`` (a lower panel or a broken axis among
    them) stops at ``2 x 2``. Readable rather than dense: the page is sized for
    the grid, so a plot in a cell keeps the size it has on a figure of its own.
    """
    if count <= 1:
        return (1, 1)
    if count == 2:
        return (1, 2)
    if count <= 4 or complex:
        return (2, 2)
    return (2, 3)


def plan_pages(complexity: Sequence[bool], layout: LayoutSpec) -> list[Page]:
    """Split plots into pages, in order; ``complexity`` flags the plots needing room.

    With ``layout="auto"`` a page takes up to :data:`MAX_PLAIN_CELLS` plots, or
    :data:`MAX_COMPLEX_CELLS` when one of the plots that would go on it has a
    lower panel or a broken axis, and its grid follows the number it takes
    (:func:`auto_grid`), so the last page adapts to what is left: seven plain
    plots are a ``2 x 3`` page and a ``1 x 1`` one. An explicit ``(rows,
    columns)`` is the grid of every page; the cells of the last one that have
    no plot stay empty.
    """
    total = len(complexity)
    if layout != "auto":
        rows, columns = layout
        per_page = rows * columns
        return [
            Page((rows, columns), min(per_page, total - start))
            for start in range(0, total, per_page)
        ]
    pages: list[Page] = []
    start = 0
    while start < total:
        is_complex = any(complexity[start : start + MAX_PLAIN_CELLS])
        count = min(total - start, MAX_COMPLEX_CELLS if is_complex else MAX_PLAIN_CELLS)
        pages.append(Page(auto_grid(count, complex=is_complex), count))
        start += count
    return pages


def check_layout(layout: object) -> LayoutSpec:
    """Return ``layout`` as ``"auto"`` or a ``(rows, columns)`` pair, else raise ``ValueError``."""
    if isinstance(layout, str) and layout == "auto":
        return "auto"
    if (
        isinstance(layout, Sequence)
        and not isinstance(layout, str)
        and len(layout) == 2
        and all(_is_count(value) for value in layout)
    ):
        return (int(layout[0]), int(layout[1]))
    msg = f"layout must be 'auto' or a (rows, columns) pair of positive integers, got {layout!r}"
    raise ValueError(msg)


def check_figsize(figsize: object) -> tuple[float, float] | None:
    """Return ``figsize`` as a ``(width, height)`` pair of inches, ``None`` as is, else raise."""
    if figsize is None:
        return None
    if (
        isinstance(figsize, Sequence)
        and not isinstance(figsize, str)
        and len(figsize) == 2
        and all(_is_length(value) for value in figsize)
    ):
        return (float(figsize[0]), float(figsize[1]))
    msg = f"figsize must be a (width, height) pair of positive numbers in inches, got {figsize!r}"
    raise ValueError(msg)


def _is_count(value: object) -> bool:
    """Whether ``value`` is a positive integer; ``True`` is an integer but not a count."""
    return not isinstance(value, bool) and isinstance(value, int | np.integer) and bool(value > 0)


def _is_length(value: object) -> bool:
    """Whether ``value`` is a finite positive number."""
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float | np.integer | np.floating)
        and math.isfinite(value)
        and bool(value > 0)
    )


def pdf_target(path: str | os.PathLike[str]) -> Path:
    """Return the PDF file ``path`` names: ``.pdf`` is appended to a name without a suffix.

    Raises
    ------
    IsADirectoryError
        ``path`` is an existing directory or ends with a separator: the document
        needs a file name.
    ValueError
        ``path`` has a suffix other than ``.pdf`` (compared ignoring case).
    """
    target = Path(path)
    if str(path).endswith(("/", os.sep)) or target.is_dir():
        example = str(target / "plots.pdf")
        msg = f"{str(path)!r} is a directory; name the PDF file to write, e.g. {example!r}"
        raise IsADirectoryError(msg)
    if not target.suffix:
        target = target.with_name(target.name + _PDF_SUFFIX)
    elif target.suffix.lower() != _PDF_SUFFIX:
        msg = (
            f"a multipage PDF ends in .pdf, got {str(path)!r}; leave the suffix out to have it "
            f"added, or write one file per plot in another format with save(directory, "
            f"formats=[{target.suffix[1:]!r}])"
        )
        raise ValueError(msg)
    if target.is_dir():
        msg = f"{str(target)!r} is a directory; name the PDF file to write"
        raise IsADirectoryError(msg)
    return target


def page_size(grid: Grid, cell: tuple[float, float]) -> tuple[float, float]:
    """Return the size of a page of ``grid`` cells of ``cell`` inches each.

    The cell size is the size the plots would have on figures of their own
    (:func:`~rootfig.plotting.figure_size`), the same for every page of a
    document, so a plot keeps its usual size and pages differ only by their grid;
    :data:`PAGE_GAP` separates the cells.
    """
    rows, columns = grid
    return (columns * cell[0] + (columns - 1) * PAGE_GAP, rows * cell[1] + (rows - 1) * PAGE_GAP)


def cell_room(size: tuple[float, float], grid: Grid) -> tuple[float, float]:
    """Return the size of each cell of a page of ``size`` inches with ``grid``.

    Raises
    ------
    ValueError
        The gaps between the cells leave them no room.
    """
    rows, columns = grid
    width = (size[0] - (columns - 1) * PAGE_GAP) / columns
    height = (size[1] - (rows - 1) * PAGE_GAP) / rows
    if width <= 0 or height <= 0:
        gaps = page_size(grid, (0.0, 0.0))
        msg = (
            f"figsize={size} leaves no room for a {rows} x {columns} grid of plots: the "
            f"{PAGE_GAP} in gaps between the cells alone take ({gaps[0]:g}, {gaps[1]:g}) "
            f"inches; give the page more than that, or a smaller layout"
        )
        raise ValueError(msg)
    return (width, height)


def background_color(style: StyleLike = None) -> tuple[float, float, float, float]:
    """Return the figure background ``style`` asks for, as RGBA.

    Resolved rather than compared as written, so that ``"white"`` and
    ``"#FFFFFF"`` count as the one colour they are.
    """
    with style_context(style):
        return to_rgba(plt.rcParams["figure.facecolor"])


def make_page(
    size: tuple[float, float], grid: Grid, *, style: StyleLike = None
) -> tuple[Figure, list[SubplotSpec]]:
    """Create the figure of a page and return it with its cells in reading order.

    The figure is created under ``style``, which sets what belongs to the figure
    itself (its background); each plot then draws under its own style. Constrained
    layout pads the page as it pads a figure of its own, and places the plots of
    the cells independently of each other, so a plot is laid out as it would be
    on a figure of its cell's size. It pads the plots of neighbouring cells only
    by that much, so every other row and column of the page's grid is a gap of
    :data:`PAGE_GAP` inches.
    """
    width, height = cell_room(size, grid)
    with style_context(style):
        engine = PlotLayoutEngine(w_pad=LAYOUT_PAD, h_pad=LAYOUT_PAD)
        fig = plt.figure(figsize=size, layout=engine)
    rows, columns = grid
    outer = fig.add_gridspec(
        2 * rows - 1,
        2 * columns - 1,
        width_ratios=[width, PAGE_GAP] * (columns - 1) + [width],
        height_ratios=[height, PAGE_GAP] * (rows - 1) + [height],
    )
    return fig, [outer[2 * row, 2 * column] for row in range(rows) for column in range(columns)]


@contextmanager
def multipage_pdf(target: Path, *, metadata: Mapping[str, Any] | None = None) -> Iterator[PdfPages]:
    """Write a multipage PDF to ``target``, through a temporary file next to it.

    The block saves its pages with :meth:`PdfPages.savefig`. The temporary file
    is renamed onto ``target`` once the block has completed and the document is
    finalised, an atomic replacement on the same file system, so an existing
    ``target`` survives a failure part way through and no half-written document
    is left behind: on any exception, ``KeyboardInterrupt`` included, the
    writer is closed, the temporary file removed and the exception re-raised.
    The parent directory is created if needed.

    ``metadata`` is the document information dictionary (``{"Title": ..., "Author":
    ...}``). It belongs to the document, so it is given to the writer here rather
    than to a page: :meth:`PdfPages.savefig` reuses the document opened with the
    first page and drops a ``metadata`` of its own.

    Raises
    ------
    ValueError
        The block saved no page: an empty document is not written.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    pdf = PdfPages(temporary, metadata=metadata)  # opens the file at the first page saved
    try:
        yield pdf
        pdf.close()
        if not temporary.exists():
            msg = f"no page was written to {str(target)!r}"
            raise ValueError(msg)
        os.replace(temporary, target)
    except BaseException:
        with suppress(Exception):
            pdf.close()
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise
