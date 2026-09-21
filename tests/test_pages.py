"""Pages of plots: grid policy, validation, page figures and the multipage PDF writer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.layout_engine import ConstrainedLayoutEngine

import rootfig as rf
from rootfig.plotting.pages import (
    Page,
    auto_grid,
    check_figsize,
    check_layout,
    make_page,
    multipage_pdf,
    page_size,
    pdf_target,
    plan_pages,
)


def _temporaries(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.tmp-*"))


class TestAutoGrid:
    @pytest.mark.parametrize(
        ("count", "grid"),
        [(1, (1, 1)), (2, (1, 2)), (3, (2, 2)), (4, (2, 2)), (5, (2, 3)), (6, (2, 3)), (7, (2, 3))],
    )
    def test_plain(self, count: int, grid: tuple[int, int]) -> None:
        assert auto_grid(count, complex=False) == grid

    @pytest.mark.parametrize(
        ("count", "grid"), [(1, (1, 1)), (2, (1, 2)), (3, (2, 2)), (4, (2, 2)), (5, (2, 2))]
    )
    def test_complex_stops_at_two_by_two(self, count: int, grid: tuple[int, int]) -> None:
        assert auto_grid(count, complex=True) == grid


class TestPlanPages:
    @pytest.mark.parametrize(
        ("count", "pages"),
        [
            (1, [((1, 1), 1)]),
            (2, [((1, 2), 2)]),
            (3, [((2, 2), 3)]),
            (4, [((2, 2), 4)]),
            (5, [((2, 3), 5)]),
            (6, [((2, 3), 6)]),
            (7, [((2, 3), 6), ((1, 1), 1)]),
            (8, [((2, 3), 6), ((1, 2), 2)]),
            (13, [((2, 3), 6), ((2, 3), 6), ((1, 1), 1)]),
        ],
    )
    def test_plain_plots_adapt_the_last_page(
        self, count: int, pages: list[tuple[tuple[int, int], int]]
    ) -> None:
        assert plan_pages([False] * count, "auto") == [Page(grid, n) for grid, n in pages]

    @pytest.mark.parametrize(
        ("count", "pages"),
        [
            (1, [((1, 1), 1)]),
            (2, [((1, 2), 2)]),
            (3, [((2, 2), 3)]),
            (4, [((2, 2), 4)]),
            (5, [((2, 2), 4), ((1, 1), 1)]),
            (9, [((2, 2), 4), ((2, 2), 4), ((1, 1), 1)]),
        ],
    )
    def test_complex_plots_take_four_per_page(
        self, count: int, pages: list[tuple[tuple[int, int], int]]
    ) -> None:
        assert plan_pages([True] * count, "auto") == [Page(grid, n) for grid, n in pages]

    def test_one_complex_plot_among_the_next_six_makes_the_page_conservative(self) -> None:
        plain, complex_ = False, True
        assert plan_pages([plain] * 4 + [complex_] + [plain] * 2, "auto") == [
            Page((2, 2), 4),
            Page((2, 2), 3),
        ]
        # a complex plot that would not fit a full plain page leaves that page dense
        assert plan_pages([plain] * 6 + [complex_], "auto") == [Page((2, 3), 6), Page((1, 1), 1)]

    def test_explicit_grid_is_kept_on_the_last_page(self) -> None:
        assert plan_pages([False] * 8, (2, 3)) == [Page((2, 3), 6), Page((2, 3), 2)]
        assert plan_pages([True] * 5, (2, 2)) == [Page((2, 2), 4), Page((2, 2), 1)]
        assert plan_pages([False] * 4, (2, 2)) == [Page((2, 2), 4)]
        assert plan_pages([False] * 3, (1, 1)) == [Page((1, 1), 1)] * 3

    def test_no_plots_no_pages(self) -> None:
        assert plan_pages([], "auto") == []
        assert plan_pages([], (2, 2)) == []


class TestValidation:
    @pytest.mark.parametrize(
        ("layout", "expected"),
        [("auto", "auto"), ((2, 3), (2, 3)), ([1, 4], (1, 4)), ((np.int64(2), 2), (2, 2))],
    )
    def test_layouts_accepted(self, layout: Any, expected: Any) -> None:
        assert check_layout(layout) == expected

    @pytest.mark.parametrize(
        "layout",
        [(0, 2), (2, 0), (-1, 2), (2,), (2, 2, 2), ("2", 2), (True, 2), (2.0, 3), "dense", 2, None],
    )
    def test_layouts_rejected(self, layout: Any) -> None:
        with pytest.raises(ValueError, match="layout must be 'auto' or a") as info:
            check_layout(layout)
        assert repr(layout) in str(info.value)

    @pytest.mark.parametrize(
        ("figsize", "expected"),
        [
            (None, None),
            ((14, 10), (14.0, 10.0)),
            ([8.5, 11], (8.5, 11.0)),
            ((np.float64(3), 2), (3.0, 2.0)),
        ],
    )
    def test_figsizes_accepted(self, figsize: Any, expected: Any) -> None:
        assert check_figsize(figsize) == expected

    @pytest.mark.parametrize(
        "figsize",
        [(0, 10), (-1, 5), (10,), ("10", 5), (math.inf, 5), (5, math.nan), (True, 5), "10x5", 10],
    )
    def test_figsizes_rejected(self, figsize: Any) -> None:
        with pytest.raises(ValueError, match="figsize must be a"):
            check_figsize(figsize)

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("plots.pdf", "plots.pdf"),
            ("plots", "plots.pdf"),
            ("nested/plots", "nested/plots.pdf"),
            ("plots.PDF", "plots.PDF"),
            ("archive.v2.pdf", "archive.v2.pdf"),
        ],
    )
    def test_pdf_target_adds_the_suffix(self, tmp_path: Path, given: str, expected: str) -> None:
        assert pdf_target(tmp_path / given) == tmp_path / expected
        assert pdf_target(str(tmp_path / given)) == tmp_path / expected

    def test_pdf_target_rejects_other_suffixes(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"ends in \.pdf.*formats=\['png'\]"):
            pdf_target(tmp_path / "plots.png")

    def test_pdf_target_rejects_directories(self, tmp_path: Path) -> None:
        with pytest.raises(IsADirectoryError, match="is a directory"):
            pdf_target(tmp_path)
        with pytest.raises(IsADirectoryError, match="is a directory"):
            pdf_target(f"{tmp_path / 'plots'}/")
        (tmp_path / "plots.pdf").mkdir()
        with pytest.raises(IsADirectoryError, match="is a directory"):
            pdf_target(tmp_path / "plots")


class TestPageFigure:
    def test_page_size_multiplies_the_cell(self) -> None:
        assert page_size((2, 3), (7.0, 5.6)) == pytest.approx((21.0, 11.2))
        assert page_size((1, 1), (7.0, 7.0)) == (7.0, 7.0)

    def test_make_page_cells_in_reading_order(self) -> None:
        fig, cells = make_page((6.0, 4.0), (2, 3))
        try:
            assert tuple(fig.get_size_inches()) == (6.0, 4.0)
            assert isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)
            assert fig.axes == []  # cells are grid positions; the plots add their own axes
            assert [(c.rowspan.start, c.colspan.start) for c in cells] == [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
            ]
        finally:
            plt.close(fig)

    def test_make_page_is_created_under_the_style(self) -> None:
        fig, _ = make_page((4.0, 3.0), (1, 1), style=rf.Style(rc={"figure.facecolor": "red"}))
        assert fig.get_facecolor() == to_rgba("red")
        plt.close(fig)
        with rf.dark_theme():
            fig, _ = make_page((4.0, 3.0), (1, 1))
        assert fig.get_facecolor() == to_rgba("none")
        plt.close(fig)


class TestMultipagePdf:
    def test_writes_through_a_temporary_file_and_replaces(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "a.pdf"
        target.parent.mkdir()
        target.write_bytes(b"old")
        with multipage_pdf(target) as pdf:
            fig = plt.figure()
            pdf.savefig(fig)
            plt.close(fig)
            assert target.read_bytes() == b"old"  # replaced only once complete
            [temporary] = _temporaries(target.parent)
            assert temporary.name.startswith("a.pdf.tmp-")
        assert target.read_bytes().startswith(b"%PDF-")
        assert _temporaries(target.parent) == []

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "er" / "a.pdf"
        with multipage_pdf(target) as pdf:
            fig = plt.figure()
            pdf.savefig(fig)
            plt.close(fig)
        assert target.exists()

    @pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
    def test_failure_keeps_the_existing_file_and_removes_the_temporary(
        self, tmp_path: Path, failure: type[BaseException]
    ) -> None:
        target = tmp_path / "a.pdf"
        target.write_bytes(b"old")

        def write_then_fail() -> None:
            with multipage_pdf(target) as pdf:
                fig = plt.figure()
                pdf.savefig(fig)
                plt.close(fig)
                assert _temporaries(tmp_path)
                raise failure

        with pytest.raises(failure):
            write_then_fail()
        assert target.read_bytes() == b"old"
        assert _temporaries(tmp_path) == []

    def test_metadata_reaches_the_document(self, tmp_path: Path) -> None:
        target = tmp_path / "a.pdf"
        with multipage_pdf(target, metadata={"Title": "Pages"}) as pdf:
            fig = plt.figure()
            pdf.savefig(fig)
            plt.close(fig)
        assert b"/Title (Pages)" in target.read_bytes()

    def test_no_page_is_an_error(self, tmp_path: Path) -> None:
        target = tmp_path / "a.pdf"
        with pytest.raises(ValueError, match="no page was written"), multipage_pdf(target):
            pass
        assert sorted(tmp_path.iterdir()) == []
