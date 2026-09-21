"""PlotBook.save_pdf: one multipage PDF drawn through the book's batched preparation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.backends.backend_pdf as backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import rootfig as rf
from rootfig.api import _pdf, batch
from rootfig.plotting.pages import PAGE_GAP, Page, page_size
from test_batch import _reads, _spy, assert_same_plot

X = rf.Variable("x", bins=(5, 0, 1))
Y = rf.Variable("y", bins=(4, 0, 4))
CELL = (7.0, 5.6)
"""The default figure size, so the cell size of a page of plain plots."""


@pytest.fixture
def samples() -> list[rf.Sample]:
    a = rf.Sample({"x": [0.2, 0.3, 0.5, 0.7], "y": [1.0, 2.0, 3.0, 4.0]}, label="A")
    b = rf.Sample({"x": [0.4, 0.6], "y": [1.5, 2.5]}, label="B")
    return [a, b]


@pytest.fixture
def files(signal_file: Path, background_file: Path) -> list[rf.Sample]:
    return [
        rf.Sample(signal_file, tree="events", label="Signal", weight="weight"),
        rf.Sample(background_file, tree="events", label="Background", weight="weight"),
    ]


def _pages(monkeypatch: pytest.MonkeyPatch) -> list[Figure]:
    """Record the figure of every page written; the figures are closed afterwards."""
    written: list[Figure] = []
    original = backend_pdf.PdfPages.savefig

    def recording(self: Any, figure: Figure, **kwargs: Any) -> Any:
        written.append(figure)
        return original(self, figure, **kwargs)

    monkeypatch.setattr(backend_pdf.PdfPages, "savefig", recording)
    return written


def _drawn(monkeypatch: pytest.MonkeyPatch) -> list[tuple[rf.PlotTask, rf.Plot]]:
    """Record the task and result of every cell drawn, in drawing order."""
    results: list[tuple[rf.PlotTask, rf.Plot]] = []
    original = batch._PreparedTask.draw

    def recording(self: Any, **extra: Any) -> rf.Plot:
        result = original(self, **extra)
        results.append((self.task, result))
        return result

    monkeypatch.setattr(batch._PreparedTask, "draw", recording)
    return results


def _info_dict(target: Path) -> dict[str, str]:
    """The document information dictionary of ``target``, read back from the written file."""
    raw = target.read_bytes()
    return {key.decode(): value.decode() for key, value in re.findall(rb"/(\w+) \(([^)]*)\)", raw)}


def _temporaries(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.tmp-*"))


def _cell(ax: Axes) -> tuple[int, int]:
    """The (row, column) of the page cell ``ax`` was drawn in (every other one is a gap)."""
    spec = ax.get_subplotspec()
    assert spec is not None
    outer = spec.get_topmost_subplotspec()
    return outer.rowspan.start // 2, outer.colspan.start // 2


class TestBasics:
    def test_writes_one_pdf(self, samples: list[rf.Sample], tmp_path: Path) -> None:
        path = rf.PlotBook(samples, [X, Y]).save_pdf(tmp_path / "plots.pdf")
        assert path == tmp_path / "plots.pdf"
        assert path.read_bytes().startswith(b"%PDF-")
        assert plt.get_fignums() == []
        assert sorted(tmp_path.iterdir()) == [path]

    @pytest.mark.parametrize(
        ("given", "expected"),
        [("plots.pdf", "plots.pdf"), ("plots", "plots.pdf"), ("nested/plots", "nested/plots.pdf")],
    )
    def test_suffix_and_directories(
        self, samples: list[rf.Sample], tmp_path: Path, given: str, expected: str
    ) -> None:
        path = rf.PlotBook(samples, X).save_pdf(str(tmp_path / given))
        assert path == tmp_path / expected
        assert path.is_file()

    def test_upper_case_suffix_is_kept(self, samples: list[rf.Sample], tmp_path: Path) -> None:
        assert rf.PlotBook(samples, X).save_pdf(tmp_path / "plots.PDF") == tmp_path / "plots.PDF"

    def test_other_suffix_and_directories_fail_before_anything_runs(self, tmp_path: Path) -> None:
        # object() cannot be plotted: reaching the data would raise something else
        with pytest.raises(ValueError, match=r"ends in \.pdf"):
            rf.PlotBook(object(), X).save_pdf(tmp_path / "plots.png")
        with pytest.raises(IsADirectoryError):
            rf.PlotBook(object(), X).save_pdf(tmp_path)
        assert sorted(tmp_path.iterdir()) == []
        assert plt.get_fignums() == []

    @pytest.mark.parametrize("keyword", ["format", "fname", "figure", "backend"])
    def test_reserved_savefig_kwargs(self, tmp_path: Path, keyword: str) -> None:
        with pytest.raises(ValueError, match=f"does not take '{keyword}'"):
            rf.PlotBook(object(), X).save_pdf(tmp_path / "plots.pdf", **{keyword: "png"})
        assert sorted(tmp_path.iterdir()) == []

    def test_savefig_kwargs_reach_every_page(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []
        original = backend_pdf.PdfPages.savefig

        def recording(self: Any, figure: Figure, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return original(self, figure, **kwargs)

        monkeypatch.setattr(backend_pdf.PdfPages, "savefig", recording)
        rf.PlotBook(samples, [X, Y]).save_pdf(tmp_path / "a.pdf", layout=(1, 1), dpi=50)
        assert calls == [{"facecolor": "auto", "edgecolor": "auto", "dpi": 50}] * 2
        rf.PlotBook(samples, X).save_pdf(tmp_path / "b.pdf", facecolor="white")
        assert calls[-1] == {"facecolor": "white", "edgecolor": "auto"}

    def test_metadata_describes_the_document(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # PdfPages.savefig() reuses the document opened with the first page and drops a
        # metadata= of its own, so it has to reach the writer instead
        calls = _pages(monkeypatch)
        target = rf.PlotBook(samples, [X, Y]).save_pdf(
            tmp_path / "plots.pdf", layout=(1, 1), metadata={"Title": "Overview", "Author": "rf"}
        )
        assert len(calls) == 2
        info = _info_dict(target)
        assert info["Title"] == "Overview"
        assert info["Author"] == "rf"

    @pytest.mark.parametrize("layout", [(0, 2), (2, 0), (2,), ("2", 2), (True, 2), "dense"])
    def test_bad_layout_fails_before_anything_runs(self, tmp_path: Path, layout: Any) -> None:
        with pytest.raises(ValueError, match="layout must be"):
            rf.PlotBook(object(), X).save_pdf(tmp_path / "plots.pdf", layout=layout)
        assert sorted(tmp_path.iterdir()) == []
        assert plt.get_fignums() == []

    @pytest.mark.parametrize("figsize", [(0, 10), (-1, 5), (10,), ("10", 5), (float("inf"), 5)])
    def test_bad_figsize_fails_before_anything_runs(self, tmp_path: Path, figsize: Any) -> None:
        with pytest.raises(ValueError, match="figsize must be"):
            rf.PlotBook(object(), X).save_pdf(tmp_path / "plots.pdf", figsize=figsize)
        assert sorted(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        "kwargs",
        [{"plot_kwargs": {"figsize": (8, 6)}}, {"variants": {"large": {"figsize": (8, 6)}}}],
    )
    def test_task_figsize_is_rejected_before_reading(
        self,
        files: list[rf.Sample],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        kwargs: dict[str, Any],
    ) -> None:
        reads = _reads(monkeypatch)
        book = rf.PlotBook(files, ["MET"], **kwargs)
        with pytest.raises(ValueError, match="controls the page size") as info:
            book.save_pdf(tmp_path / "plots.pdf")
        where = "plot_kwargs" if "plot_kwargs" in kwargs else "variant 'large'"
        assert f"remove figsize= from {where}" in str(info.value)
        assert sorted(tmp_path.iterdir()) == []
        assert reads == []
        assert plt.get_fignums() == []
        # a task-level figsize stays valid where a task draws its own figure
        for _, result in book.plots():
            assert tuple(result.fig.get_size_inches()) == (8.0, 6.0)
            result.close()

    def test_explicit_figsize_skips_resolving_the_task_styles(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the cell size is what a task's style would give it, which an explicit page size
        # makes irrelevant: resolving every style for a result nothing reads is wasted work
        calls: list[int] = []
        original = _pdf.cell_size

        def recording(tasks: Any) -> Any:
            calls.append(len(tasks))
            return original(tasks)

        monkeypatch.setattr(_pdf, "cell_size", recording)
        rf.PlotBook(samples, [X, Y]).save_pdf(tmp_path / "auto.pdf", figsize=(9.0, 7.0))
        assert calls == []
        rf.PlotBook(samples, [X, Y]).save_pdf(tmp_path / "sized.pdf")
        assert calls == [2]

    @pytest.mark.parametrize("figsize", [None, (9.0, 7.0)])
    def test_a_style_that_cannot_be_resolved_names_its_task(
        self, samples: list[rf.Sample], tmp_path: Path, figsize: Any
    ) -> None:
        # without a figsize the style is resolved by cell_size(), with one by the page the
        # task opens; both are outside the drawing step that carries the task note itself
        book = rf.PlotBook(samples, [X], variants={"ok": {}, "bad": {"style": "no-such-style"}})
        with pytest.raises(ValueError, match="unknown style") as info:
            book.save_pdf(tmp_path / "plots.pdf", layout=(1, 1), figsize=figsize)
        assert any("variant='bad'" in note for note in info.value.__notes__)
        assert sorted(tmp_path.iterdir()) == []

    def test_figsize_overridden_away_by_every_variant_is_accepted(
        self, samples: list[rf.Sample], tmp_path: Path
    ) -> None:
        # no task runs with a figsize, so the page keeps the size save_pdf() gives it
        book = rf.PlotBook(
            samples,
            [X],
            plot_kwargs={"figsize": (8.0, 6.0)},
            variants={"a": {"figsize": None}, "b": {"figsize": None}},
        )
        assert book.save_pdf(tmp_path / "plots.pdf").is_file()

    def test_figsize_left_by_one_variant_names_where_it_comes_from(
        self, samples: list[rf.Sample], tmp_path: Path
    ) -> None:
        book = rf.PlotBook(
            samples,
            [X],
            plot_kwargs={"figsize": (8.0, 6.0)},
            variants={"a": {"figsize": None}, "b": {}},
        )
        with pytest.raises(ValueError, match="remove figsize= from plot_kwargs"):
            book.save_pdf(tmp_path / "plots.pdf")
        assert sorted(tmp_path.iterdir()) == []


class TestPages:
    def test_seven_plain_tasks_make_a_dense_page_and_a_single_one(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pages = _pages(monkeypatch)
        variables = [X.replace(name=f"x{i}") for i in range(7)]
        rf.PlotBook(samples, variables).save_pdf(tmp_path / "plots.pdf")
        assert [len(page.axes) for page in pages] == [6, 1]
        assert [tuple(page.get_size_inches()) for page in pages] == [
            pytest.approx((3 * CELL[0] + 2 * PAGE_GAP, 2 * CELL[1] + PAGE_GAP)),
            pytest.approx(CELL),
        ]
        assert not any(plt.fignum_exists(page.number) for page in pages)

    def test_explicit_layout_keeps_its_grid_and_leaves_unused_cells_empty(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pages = _pages(monkeypatch)
        drawn = _drawn(monkeypatch)
        variables = [X.replace(name=f"x{i}") for i in range(5)]
        book = rf.PlotBook(samples, variables)
        book.save_pdf(tmp_path / "plots.pdf", layout=(2, 2))
        assert [len(page.axes) for page in pages] == [4, 1]  # no axes for the three unused cells
        assert [tuple(page.get_size_inches()) for page in pages] == [
            pytest.approx(page_size((2, 2), CELL))
        ] * 2
        assert [task.stem for task, _ in drawn] == [task.stem for task in book.tasks()]
        assert [_cell(result.ax) for _, result in drawn] == [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0)]

    def test_complex_tasks_use_the_conservative_grid(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pages = _pages(monkeypatch)
        variables = [X.replace(name=f"x{i}") for i in range(5)]
        rf.PlotBook(samples, variables, plot_kwargs={"ratio": True}).save_pdf(tmp_path / "p.pdf")
        assert [len(page.axes) for page in pages] == [8, 2]  # main and ratio axes per cell
        cell_height = CELL[1] * (1 + 0.3 * 0.85)  # as make_figure enlarges a ratio figure
        assert tuple(pages[0].get_size_inches()) == pytest.approx(
            page_size((2, 2), (CELL[0], cell_height))
        )

    def test_tasks_stay_in_book_order_across_pages(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn = _drawn(monkeypatch)
        book = rf.PlotBook(
            samples,
            [X, Y],
            selections={"all": None, "hi": "y > 1"},
            variants={"lin": {}, "log": {"logy": True}},
        )
        book.save_pdf(tmp_path / "plots.pdf", layout=(1, 3))
        assert [task.stem for task, _ in drawn] == [task.stem for task in book.tasks()]
        assert [_cell(result.ax) for _, result in drawn] == [(0, 0), (0, 1), (0, 2)] * 2 + [
            (0, 0),
            (0, 1),
        ]
        figures = [result.fig for _, result in drawn]
        assert len({id(fig) for fig in figures}) == 3  # one figure per page
        assert figures[0] is figures[1] is figures[2]

    def test_page_figsize_is_the_whole_page(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pages = _pages(monkeypatch)
        rf.PlotBook(samples, [X, Y]).save_pdf(tmp_path / "plots.pdf", figsize=(14, 10))
        assert [tuple(page.get_size_inches()) for page in pages] == [(14.0, 10.0)]
        assert len(pages[0].axes) == 2

    def test_cells_are_sized_once_for_the_document(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # a ratio panel makes every cell taller, also on the page without one
        pages = _pages(monkeypatch)
        book = rf.PlotBook(samples, [X, Y], variants={"plain": {}, "ratio": {"ratio": True}})
        book.save_pdf(tmp_path / "plots.pdf", layout=(1, 2))
        cell_height = CELL[1] * (1 + 0.3 * 0.85)
        assert [tuple(page.get_size_inches()) for page in pages] == [
            pytest.approx(page_size((1, 2), (CELL[0], cell_height)))
        ] * 2
        wide = rf.Style(figsize=(9.0, 4.0))
        book = rf.PlotBook(samples, [X, Y], variants={"a": {}, "b": {"style": wide}})
        book.save_pdf(tmp_path / "wide.pdf", layout=(1, 1))
        assert [tuple(page.get_size_inches()) for page in pages[2:]] == [
            pytest.approx((9.0, 5.6))
        ] * 4

    def test_pages_are_written_one_at_a_time(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        open_at_save: list[int] = []
        original = backend_pdf.PdfPages.savefig

        def recording(self: Any, figure: Figure, **kwargs: Any) -> Any:
            open_at_save.append(len(plt.get_fignums()))
            return original(self, figure, **kwargs)

        monkeypatch.setattr(backend_pdf.PdfPages, "savefig", recording)
        variables = [X.replace(name=f"x{i}") for i in range(3)]
        rf.PlotBook(samples, variables).save_pdf(tmp_path / "plots.pdf", layout=(1, 1))
        assert open_at_save == [1, 1, 1]
        assert plt.get_fignums() == []


class TestCells:
    """Each cell holds what the standalone plot would, in its own axes."""

    def test_ratio_cells_have_their_own_panels(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn = _drawn(monkeypatch)
        kwargs: dict[str, Any] = {"ratio": True, "stack": False}
        rf.PlotBook(samples, [X, Y], plot_kwargs=kwargs).save_pdf(tmp_path / "plots.pdf")
        assert len(drawn) == 2
        axes = [ax for _, result in drawn for ax in (result.ax, result.ratio_ax)]
        assert all(isinstance(ax, Axes) for ax in axes)
        assert len({id(ax) for ax in axes}) == 4
        for task, result in drawn:
            assert result.ratio_ax is not None
            assert result.ratio_ax.get_shared_x_axes().joined(result.ax, result.ratio_ax)
            assert _cell(result.ax) == _cell(result.ratio_ax)
            assert result.ax.get_title() == ""  # no task caption is added
            assert_same_plot(result, rf.plot(samples, task.variable, **kwargs))

    def test_significance_cell(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn = _drawn(monkeypatch)
        kwargs: dict[str, Any] = {"ratio": ("s/sqrt(b)", "B"), "title": "Signal B"}
        rf.PlotBook(samples, [X, Y], plot_kwargs=kwargs).save_pdf(tmp_path / "plots.pdf")
        for task, result in drawn:
            assert result.ratios
            assert result.ratio_ax is not None
            assert result.ax.get_title() == "Signal B"
            assert_same_plot(result, rf.plot(samples, task.variable, **kwargs))

    @pytest.mark.parametrize("ratio", [False, True])
    def test_broken_axes_stay_within_their_cell(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ratio: bool
    ) -> None:
        drawn = _drawn(monkeypatch)
        positions: dict[str, list[tuple[float, float]]] = {}
        original = backend_pdf.PdfPages.savefig

        def measuring(self: Any, figure: Figure, **kwargs: Any) -> Any:
            figure.canvas.draw()
            for task, result in drawn:
                positions[task.stem] = [
                    (ax.get_position().x0, ax.get_position().x1) for ax in result.axes
                ]
            return original(self, figure, **kwargs)

        monkeypatch.setattr(backend_pdf.PdfPages, "savefig", measuring)
        kwargs: dict[str, Any] = {"xbreak": (0.4, 0.6), "ratio": ratio}
        book = rf.PlotBook(samples, [X, X.replace(name="x2")], plot_kwargs=kwargs)
        book.save_pdf(tmp_path / "plots.pdf")
        (first, left), (second, right) = drawn
        for result in (left, right):
            assert result.ax_right is not None
            assert (result.ratio_ax_right is not None) is ratio
            assert len(result.axes) == (4 if ratio else 2)
            assert result.ax_right.get_shared_y_axes().joined(result.ax, result.ax_right)
        assert len({id(ax) for r in (left, right) for ax in r.axes}) == (8 if ratio else 4)
        # every axes of the left cell ends before any axes of the right cell begins
        assert max(x1 for _, x1 in positions[first.stem]) < min(
            x0 for x0, _ in positions[second.stem]
        )
        assert_same_plot(left, rf.plot(samples, X, **kwargs))

    def test_mixed_plain_and_complex_cells_on_one_page(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pages = _pages(monkeypatch)
        drawn = _drawn(monkeypatch)
        book = rf.PlotBook(samples, [X, Y], variants={"plain": {}, "ratio": {"ratio": True}})
        book.save_pdf(tmp_path / "plots.pdf")
        assert len(pages) == 1
        assert len(pages[0].axes) == 6  # two plain cells plus two cells of main and ratio
        by_variant = {task.variant_name: result for task, result in drawn}
        assert by_variant["plain"].ratio_ax is None
        assert by_variant["ratio"].ratio_ax is not None
        assert [_cell(result.ax) for _, result in drawn] == [(0, 0), (0, 1), (1, 0), (1, 1)]
        for task, result in drawn:
            assert_same_plot(result, rf.plot(samples, task.variable, **task.kwargs))

    def test_styles_do_not_leak_between_cells(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn = _drawn(monkeypatch)
        serif = rf.Style(rc={"font.family": "serif", "xtick.labelsize": 20, "axes.labelsize": 22})
        sans = rf.Style(rc={"xtick.labelsize": 8, "axes.labelsize": 9})
        book = rf.PlotBook(
            samples, X, variants={"serif": {"style": serif}, "sans": {"style": sans}}
        )
        book.save_pdf(tmp_path / "plots.pdf")
        by_variant = {task.variant_name: result for task, result in drawn}
        first, second = by_variant["serif"].ax, by_variant["sans"].ax
        assert first.figure is second.figure
        assert first.xaxis.get_tick_params()["labelsize"] == 20
        assert second.xaxis.get_tick_params()["labelsize"] == 8
        # pinned to the concrete fonts of each style; the generic name may follow
        assert first.xaxis.get_tick_params()["labelfontfamily"][0] == "DejaVu Serif"
        assert second.xaxis.get_tick_params()["labelfontfamily"][0] == "DejaVu Sans"
        assert first.xaxis.label.get_fontfamily()[0] == "DejaVu Serif"
        assert second.xaxis.label.get_fontfamily()[0] == "DejaVu Sans"
        assert (first.xaxis.label.get_fontsize(), second.xaxis.label.get_fontsize()) == (22, 9)

    def test_page_background_is_the_one_its_plots_ask_for(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pages = _pages(monkeypatch)
        with rf.dark_theme():
            rf.PlotBook(samples, X).save_pdf(tmp_path / "dark.pdf")
        assert pages[0].get_facecolor() == (0.0, 0.0, 0.0, 0.0)
        red = rf.Style(rc={"figure.facecolor": "red"})
        rf.PlotBook(samples, X, variants={"r": {"style": red}}).save_pdf(tmp_path / "r.pdf")
        assert pages[1].get_facecolor() == (1.0, 0.0, 0.0, 1.0)

    def test_plots_wanting_different_backgrounds_do_not_share_a_page(
        self, samples: list[rf.Sample], tmp_path: Path
    ) -> None:
        # a cell's labels and legend are drawn outside its axes, on the page's background,
        # so a dark plot beside a light one would lose them; there is no per-cell figure
        red = rf.Style(rc={"figure.facecolor": "red"})
        book = rf.PlotBook(samples, X, variants={"r": {"style": red}, "d": {}})
        with pytest.raises(ValueError, match="different page backgrounds") as info:
            book.save_pdf(tmp_path / "plots.pdf")
        message = str(info.value)
        assert "variant='r'" in message
        assert "variant='d'" in message
        assert "select(variants='d')" in message
        assert "layout=(1, 1)" in message
        assert sorted(tmp_path.iterdir()) == []
        assert plt.get_fignums() == []
        # one per page is fine: each page takes the background of the plots on it
        assert book.save_pdf(tmp_path / "split.pdf", layout=(1, 1)).is_file()

    def test_the_first_clash_is_reported_before_a_later_style_is_resolved(
        self, samples: list[rf.Sample], tmp_path: Path
    ) -> None:
        # backgrounds are resolved only as far as the first disagreement, so a broken style
        # further along the page cannot mask a clash the earlier plots already have
        book = rf.PlotBook(
            samples,
            X,
            variants={
                "red": {"style": rf.Style(rc={"figure.facecolor": "red"})},
                "blue": {"style": rf.Style(rc={"figure.facecolor": "blue"})},
                "broken": {"style": "no-such-style"},
            },
        )
        with pytest.raises(ValueError, match="different page backgrounds") as info:
            book.save_pdf(tmp_path / "plots.pdf")
        message = str(info.value)
        assert "variant='red'" in message
        assert "variant='blue'" in message
        assert "broken" not in message

    def test_one_background_spelt_two_ways_is_not_a_clash(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ATLAS writes it '#FFFFFF' and CMS 'white'; resolved colours are compared, not names
        pages = _pages(monkeypatch)
        book = rf.PlotBook(
            samples,
            X,
            variants={
                "atlas": {"style": rf.Style(experiment="ATLAS")},
                "cms": {"style": rf.Style(experiment="CMS")},
            },
        )
        assert book.save_pdf(tmp_path / "plots.pdf").is_file()
        assert pages[0].get_facecolor() == (1.0, 1.0, 1.0, 1.0)


def _geometry(result: rf.Plot) -> dict[str, np.ndarray]:
    """Where everything of ``result`` sits, in inches from its top-left panel corner."""
    fig = result.fig
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    corners = np.array([ax.get_window_extent(renderer).get_points() for ax in result.axes])
    origin = np.array([corners[:, 0, 0].min(), corners[:, 1, 1].max()])

    def box(artist: Any) -> np.ndarray:
        (x0, y0), (x1, y1) = artist.get_window_extent(renderer).get_points()
        return np.array([x0 - origin[0], origin[1] - y1, x1 - x0, y1 - y0]) / fig.dpi

    found = {"ylim": np.array(result.ax.get_ylim())}
    for i, ax in enumerate(result.axes):
        found[f"axes {i}"] = box(ax)
        texts = [*ax.texts, ax.xaxis.label, ax.yaxis.label, ax.xaxis.get_offset_text(), ax.title]
        for text in texts:
            if text.get_visible() and text.get_text():
                found[f"axes {i} {text.get_text()!r}"] = np.r_[box(text), text.get_fontsize()]
        for j, artist in enumerate([*ax.artists, ax.get_legend()]):
            if artist is not None:
                found[f"axes {i} artist {j}"] = box(artist)
    return found


class TestLayoutIndependence:
    """A cell holds its plot exactly as a figure of its own does, at any layout, and a page
    costs as many layout passes as a page of one plot."""

    CMS = rf.Style(experiment="CMS", status="Preliminary", lumi=138, com=13.6)
    OWN = rf.Style(
        experiment="FCC-ee",  # no mplhep helper of its own
        status="Simulation",
        lumi="10.8 ab^-1",
        com="240 GeV",
        figsize=(5.0, 4.5),
        rc={"figure.subplot.left": 0.25, "font.size": 13, "axes.labelsize": 17},
    )

    @pytest.mark.parametrize("layout", [(1, 1), (3, 2)])
    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"ratio": True}, id="ratio"),
            pytest.param({"ratio": True, "style": CMS}, id="experiment-label"),
            pytest.param({"ratio": True, "style": OWN}, id="own-style"),
            pytest.param({"stats": True, "text": ["a line", "another"]}, id="stats-text"),
            pytest.param({"xbreak": (0.4, 0.6), "ratio": True}, id="xbreak"),
        ],
    )
    def test_cells_equal_their_own_figures(
        self,
        samples: list[rf.Sample],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        layout: tuple[int, int],
        kwargs: dict[str, Any],
    ) -> None:
        book = rf.PlotBook(samples, [X, X.replace(name="x2", bins=(3, 0, 1))], plot_kwargs=kwargs)
        alone = []
        for _, result in book.plots():
            alone.append(_geometry(result))
            result.close()
        cells: list[dict[str, np.ndarray]] = []
        drawn = _drawn(monkeypatch)
        original = backend_pdf.PdfPages.savefig

        def measuring(self: Any, figure: Figure, **options: Any) -> Any:
            cells.extend(_geometry(result) for _, result in drawn[len(cells) :])
            return original(self, figure, **options)

        monkeypatch.setattr(backend_pdf.PdfPages, "savefig", measuring)
        book.save_pdf(tmp_path / "plots.pdf", layout=layout)
        assert len(cells) == len(alone) == 2
        for cell, own in zip(cells, alone, strict=True):
            assert cell.keys() == own.keys()
            for name, where in own.items():
                np.testing.assert_allclose(cell[name], where, atol=2e-3, err_msg=name)

    @pytest.mark.parametrize("style", [None, CMS], ids=["plain", "experiment-label"])
    def test_a_page_is_laid_out_as_often_as_a_page_of_one_plot(
        self,
        samples: list[rf.Sample],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        style: rf.Style | None,
    ) -> None:
        draws: list[Figure] = []
        original = Figure.draw

        def counting(self: Figure, renderer: Any) -> None:
            draws.append(self)
            original(self, renderer)

        monkeypatch.setattr(Figure, "draw", counting)
        kwargs = {"ratio": True, "style": style}
        one = rf.PlotBook(samples, [X], plot_kwargs=kwargs)
        one.save_pdf(tmp_path / "one.pdf", layout=(1, 1))
        per_page = len(draws)
        draws.clear()
        four = rf.PlotBook(samples, [X.replace(name=f"x{i}") for i in range(4)], plot_kwargs=kwargs)
        four.save_pdf(tmp_path / "four.pdf", layout=(2, 2))
        assert len({id(fig) for fig in draws}) == 1
        assert len(draws) == per_page

    def test_a_page_too_small_for_its_grid_is_refused_before_drawing(
        self, samples: list[rf.Sample], tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match=r"no room for a 8 x 8 grid"):
            rf.PlotBook(samples, [X, Y]).save_pdf(tmp_path / "p.pdf", layout=(8, 8), figsize=(1, 1))
        assert sorted(tmp_path.iterdir()) == []
        assert plt.get_fignums() == []


class TestBatching:
    def test_reads_and_preparations_as_for_plots(
        self, files: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        book = rf.PlotBook(
            files,
            ["MET", "Muon_pt", "nMuon"],
            variants={"lin": {}, "log": {"logy": True}},
            plot_kwargs={"stack": True},
        )
        reads = _reads(monkeypatch)
        prepared = _spy(monkeypatch, "prepare_plot")
        drawn = _drawn(monkeypatch)
        book.save_pdf(tmp_path / "plots.pdf")
        assert len(reads) == 2  # one per sample for the whole batch
        assert all(set(call) == {"MET", "Muon_pt", "nMuon", "weight"} for call in reads)
        assert len(prepared) == 3  # the log variant draws from the lin preparation
        assert len(drawn) == 6
        del reads[:], prepared[:]
        for _, result in book.plots():
            result.close()
        assert (len(reads), len(prepared)) == (2, 3)
        for task, result in drawn:
            lin, log = result.histograms, rf.plot(files, task.variable, **task.kwargs).histograms
            for got, want in zip(lin, log, strict=True):
                np.testing.assert_array_equal(got.values(flow=True), want.values(flow=True))

    def test_variants_do_not_share_histograms(
        self, files: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn = _drawn(monkeypatch)
        book = rf.PlotBook(files, ["MET"], variants={"lin": {}, "log": {"logy": True}})
        book.save_pdf(tmp_path / "plots.pdf")
        (_, lin), (_, log) = drawn
        assert all(
            a.hist is not b.hist for a, b in zip(lin.histograms, log.histograms, strict=True)
        )

    def test_discovered_variables_in_order_without_a_second_discovery(
        self, signal_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sample = rf.Sample(signal_file, tree="events", weight="weight")
        book = rf.PlotBook(sample, rf.ALL, include=["MET", "Muon_pt", "nMuon"])
        assert [v.expression for v in book.variables] == ["MET", "Muon_pt", "nMuon"]
        discovered = _spy(monkeypatch, "discover_variables")
        reads = _reads(monkeypatch)
        drawn = _drawn(monkeypatch)
        book.save_pdf(tmp_path / "plots.pdf")
        assert discovered == []
        assert len(reads) == 1
        assert [task.variable for task, _ in drawn] == list(book.variables)

    def test_selected_book(
        self, signal_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn = _drawn(monkeypatch)
        book = rf.PlotBook(rf.Sample(signal_file, tree="events"), rf.ALL)
        book.select(variables=["Muon_pt", "Muon_eta"]).save_pdf(tmp_path / "muons.pdf")
        assert [task.variable.expression for task, _ in drawn] == ["Muon_eta", "Muon_pt"]

    @pytest.mark.parametrize("case", ["systematics", "group_observed_ratio", "stored"])
    def test_every_cell_equals_plot(
        self,
        case: str,
        data_dir: Path,
        stored_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from test_batch import _books

        drawn = _drawn(monkeypatch)
        book = _books(data_dir, stored_dir)[case]
        book.save_pdf(tmp_path / f"{case}.pdf")
        assert [task.stem for task, _ in drawn] == [task.stem for task in book.tasks()]
        for task, result in drawn:
            direct = rf.plot(book.data, task.variable, selection=task.selection, **task.kwargs)
            assert_same_plot(result, direct)
            direct.close()


class TestFailures:
    def test_deferred_mathtext_failure_names_the_page_and_preserves_output(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        existing = plt.figure()
        target = tmp_path / "plots.pdf"
        target.write_bytes(b"old")
        drawn = _drawn(monkeypatch)
        pages = _pages(monkeypatch)
        book = rf.PlotBook(
            samples,
            X,
            variants={"a": {}, "b": {}, "c": {}, "bad": {"title": r"$\notacommand$"}},
        )
        with pytest.raises(ValueError, match="Unknown symbol") as info:
            book.save_pdf(target, layout=(1, 2))
        assert len(drawn) == 4  # both draw calls on page 2 returned; finishing it failed
        assert len(pages) == 1
        assert info.value.__notes__ == [f"while rendering PlotBook PDF page 2 to {str(target)!r}"]
        assert plt.get_fignums() == [existing.number]
        assert target.read_bytes() == b"old"
        assert _temporaries(tmp_path) == []

    def test_drawing_failure_cleans_up(self, samples: list[rf.Sample], tmp_path: Path) -> None:
        existing = plt.figure()
        target = tmp_path / "plots.pdf"
        target.write_bytes(b"old")
        book = rf.PlotBook(samples, X, variants={"good": {}, "bad": {"ratio": "missing"}})
        with pytest.raises(ValueError, match="ratio reference 'missing'") as info:
            book.save_pdf(target)
        assert info.value.__notes__ == [
            "while running PlotBook task variable='x', selection='all', variant='bad'",
            f"while rendering PlotBook PDF page 1 to {str(target)!r}",
        ]
        assert plt.get_fignums() == [existing.number]
        assert target.read_bytes() == b"old"
        assert _temporaries(tmp_path) == []

    def test_preparation_failure_discards_the_pages_already_written(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pages = _pages(monkeypatch)
        target = tmp_path / "plots.pdf"
        book = rf.PlotBook(samples, [X, "nosuch"])
        with pytest.raises(rf.MissingBranchError, match="'nosuch'") as info:
            book.save_pdf(target, layout=(1, 1))
        assert len(pages) == 1  # the first page had been written to the temporary file
        assert info.value.__notes__ == [
            "while running PlotBook task variable='nosuch', selection='all', variant='default'",
            f"while rendering PlotBook PDF page 2 to {str(target)!r}",
        ]
        assert sorted(tmp_path.iterdir()) == []
        assert plt.get_fignums() == []

    def test_page_save_failure(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing(self: Any, figure: Figure, **kwargs: Any) -> Any:
            msg = "disk full"
            raise OSError(msg)

        monkeypatch.setattr(backend_pdf.PdfPages, "savefig", failing)
        target = tmp_path / "plots.pdf"
        target.write_bytes(b"old")
        with pytest.raises(OSError, match="disk full") as info:
            rf.PlotBook(samples, X).save_pdf(target)
        assert info.value.__notes__ == [f"while writing PlotBook PDF page 1 to {str(target)!r}"]
        assert plt.get_fignums() == []
        assert target.read_bytes() == b"old"
        assert _temporaries(tmp_path) == []

    def test_keyboard_interrupt_cleans_up(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        existing = plt.figure()

        def interrupted(*args: Any, **kwargs: Any) -> rf.Plot:
            raise KeyboardInterrupt

        monkeypatch.setattr(batch, "draw_plot", interrupted)
        target = tmp_path / "plots.pdf"
        with pytest.raises(KeyboardInterrupt):
            rf.PlotBook(samples, [X, Y]).save_pdf(target, layout=(1, 1))
        assert plt.get_fignums() == [existing.number]
        assert sorted(tmp_path.iterdir()) == []

    def test_existing_file_is_replaced_on_success(
        self, samples: list[rf.Sample], tmp_path: Path
    ) -> None:
        target = tmp_path / "plots.pdf"
        target.write_bytes(b"old")
        assert rf.PlotBook(samples, X).save_pdf(target) == target
        assert target.read_bytes().startswith(b"%PDF-")
        assert sorted(tmp_path.iterdir()) == [target]


class TestPlan:
    def test_complexity_follows_ratio_and_xbreak(self) -> None:
        assert not _pdf.is_complex({})
        assert not _pdf.is_complex({"ratio": False, "xbreak": None})
        assert _pdf.is_complex({"ratio": True})
        assert _pdf.is_complex({"ratio": "significance"})
        assert _pdf.is_complex({"xbreak": (1, 2)})

    def test_pages_of_a_book(self, samples: list[rf.Sample]) -> None:
        variables = [X.replace(name=f"x{i}") for i in range(4)]
        book = rf.PlotBook(samples, variables, variants={"a": {}, "b": {"xbreak": (0.4, 0.6)}})
        from rootfig.plotting.pages import plan_pages

        pages = plan_pages([_pdf.is_complex(task.kwargs) for task in book.tasks()], "auto")
        assert pages == [Page((2, 2), 4), Page((2, 2), 4)]
