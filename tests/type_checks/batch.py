"""Static checks for the public PlotBook contracts (run by mypy)."""

from collections.abc import Iterator
from pathlib import Path
from typing import assert_type

import rootfig as rf


def plot_book(sample: rf.Sample, group: rf.Group) -> None:
    book = rf.PlotBook(
        [sample, group],
        [rf.Variable("x"), "y"],
        selections={"sr": "x > 0", "loose": rf.Cut("x > -1"), "none": None},
        variants={"lin": {}, "log": {"logy": True}},
        plot_kwargs={"stack": True, "observed": group, "style": rf.Style(experiment="X")},
    )
    assert_type(rf.PlotBook(sample, "x").variables, tuple[rf.Variable, ...])
    assert_type(rf.PlotBook(sample, rf.Variable("x")).variables, tuple[rf.Variable, ...])
    assert_type(book.tasks(), tuple[rf.PlotTask, ...])
    assert_type(book.plots(), Iterator[tuple[rf.PlotTask, rf.Plot]])
    assert_type(book.save("plots", formats=["pdf", "png"], dpi=200), list[Path])
    assert_type(book.select(variables="x", selections=["sr"], variants=None), rf.PlotBook)
    task = book.tasks()[0]
    assert_type(task.variable, rf.Variable)
    assert_type(task.selection_name, str | None)
    assert_type(task.selection, rf.Cut | None)
    assert_type(task.variant_name, str | None)
    assert_type(task.stem, str)
    for _, result in book.plots():
        assert_type(result, rf.Plot)
    # A variant maps keywords to values; mypy reports an unused ignore if this loosens.
    rf.PlotBook(sample, "x", variants={"log": True})  # type: ignore[dict-item]
